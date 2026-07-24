"""Kalman filter processing module for Ring IMU data.

This module ports the Kalman filtering pipeline from the original
``imu_visualizer.py`` into a standalone, self-contained component. It processes
raw IMU sensor data (from the Ring Sound SDK's ``SensorDataBatch.samples``)
through a :class:`MotionIntegrator` and produces structured output with motion
state classification.

The pipeline (see :class:`MotionIntegrator`):
    unit conversion -> stationary bias calibration -> complementary attitude
    estimation -> gravity removal and rotation into a world frame -> per-axis
    constant-acceleration Kalman filter with zero-velocity updates (ZUPT).

Unlike the previous revision, this module has NO external dependency on
``imu_visualizer.py``; all filter math lives here.

Usage:
    from ring_IMU.kalman_filter import KalmanProcessor

    processor = KalmanProcessor()
    processor.start(sample_rate_hz=50, accel_range_g=2, gyro_range_dps=250)

    for sample in batch.samples:
        result = processor.process(sample, sequence=seq)
        print(result)
"""

from __future__ import annotations

from collections import deque
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any

__all__ = [
    "MOTION_THRESHOLDS",
    "Axis3",
    "ImuPoint",
    "MotionIntegrator",
    "axis_magnitude",
    "dominant_axis",
    "imu_point_payload",
    "HandoffImuCsvLogger",
    "KalmanProcessor",
]


# ---------------------------------------------------------------------------
# Filter tuning constants (ported verbatim from imu_visualizer.py)
# ---------------------------------------------------------------------------

GRAVITY_MPS2 = 9.80665
INT16_FULL_SCALE = 32768.0
DEFAULT_CALIBRATION_SECONDS = 1.0
DEFAULT_KALMAN_INITIAL_ERROR = 1.0
# Display smoothing for the acceleration and angular-velocity channels.
ACCEL_KALMAN_PROCESS_NOISE = 0.08
ACCEL_KALMAN_MEASUREMENT_NOISE = 0.75
GYRO_KALMAN_PROCESS_NOISE = 0.20
GYRO_KALMAN_MEASUREMENT_NOISE = 3.0
# Complementary attitude filter used to estimate and remove gravity.
ATTITUDE_TIME_CONSTANT_S = 0.5
ATTITUDE_ACCEL_TRUST_TOL = 0.2
# Per-axis constant-acceleration Kalman filter for velocity and position.
KINEMATIC_ACCEL_NOISE_MPS2 = 0.6
KINEMATIC_INITIAL_POS_VAR = 1e-4
KINEMATIC_INITIAL_VEL_VAR = 1e-4
# Zero-velocity update (ZUPT) that bounds integration drift when stationary.
ZUPT_ACCEL_TOL_MPS2 = 0.3
ZUPT_GYRO_TOL_DPS = 8.0
ZUPT_VELOCITY_MEAS_NOISE = 1e-3
# Runtime gyro-bias tracking: EMA rate applied per detected stationary sample.
GYRO_BIAS_ADAPT_RATE = 0.02
# Impulse rejection for the firmware batch-boundary glitch (raw-count domain).
GYRO_SPIKE_WINDOW = 7
GYRO_SPIKE_SIGMA = 5.0
GYRO_SPIKE_FLOOR_COUNTS = 12.0


Axis3 = tuple[float, float, float]


# ---------------------------------------------------------------------------
# Threshold constants for motion state classification
# ---------------------------------------------------------------------------

MOTION_THRESHOLDS = {
    # Stationary detection
    "stationary_accel_max_mps2": 0.75,
    "stationary_gyro_max_dps": 8.0,
    # Moving detection
    "moving_accel_min_mps2": 1.2,
    # Rotating detection
    "rotating_gyro_min_dps": 20.0,
    # Quality flags
    "quality_accel_unstable_mps2": 12.0,
    "quality_gyro_unstable_dps": 250.0,
    # Dominant axis thresholds
    "dominant_motion_axis_threshold_mps2": 1.2,
    "dominant_rotation_axis_threshold_dps": 20.0,
}


# ---------------------------------------------------------------------------
# Data point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImuPoint:
    """One processed IMU sample emitted by :class:`MotionIntegrator`."""

    host_time_s: float
    sequence: int
    timestamp_ms: int
    accel_mps2: Axis3
    gyro_dps: Axis3
    velocity_mps: Axis3
    position_m: Axis3
    raw_accel: tuple[int, int, int]
    raw_gyro: tuple[int, int, int]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def axis_magnitude(values: Any) -> float:
    """Compute the Euclidean magnitude of a 3-axis vector.

    Args:
        values: Iterable of 3 numeric values (x, y, z).

    Returns:
        The magnitude sqrt(x^2 + y^2 + z^2).
    """
    return math.sqrt(sum(float(v) * float(v) for v in values))


def dominant_axis(values: Any, *, threshold: float) -> str:
    """Determine which axis (x, y, z) has the largest absolute value.

    Args:
        values: Iterable of 3 numeric values.
        threshold: Minimum absolute value to declare a dominant axis.

    Returns:
        One of 'x', 'y', 'z', or 'none' if below threshold.
    """
    axes = ("x", "y", "z")
    magnitudes = [abs(float(v)) for v in values]
    best_index = max(range(3), key=lambda i: magnitudes[i])
    if magnitudes[best_index] < threshold:
        return "none"
    return axes[best_index]


def imu_point_payload(point: Any) -> dict[str, Any]:
    """Convert a :class:`ImuPoint` to a structured dict.

    Returns:
        dict with processed fields including motion state classification.
    """
    motion_intensity = axis_magnitude(point.accel_mps2)
    rotation_intensity = axis_magnitude(point.gyro_dps)

    t = MOTION_THRESHOLDS
    is_stationary = (
        motion_intensity < t["stationary_accel_max_mps2"]
        and rotation_intensity < t["stationary_gyro_max_dps"]
    )
    is_moving = motion_intensity >= t["moving_accel_min_mps2"]
    is_rotating = rotation_intensity >= t["rotating_gyro_min_dps"]

    quality = "ok"
    if motion_intensity > t["quality_accel_unstable_mps2"]:
        quality = "gravity_or_motion_unstable"
    if rotation_intensity > t["quality_gyro_unstable_dps"]:
        quality = "gyro_unstable"

    return {
        "host_time_s": point.host_time_s,
        "sequence": point.sequence,
        "timestamp_ms": point.timestamp_ms,
        "kalman_accel_x_mps2": point.accel_mps2[0],
        "kalman_accel_y_mps2": point.accel_mps2[1],
        "kalman_accel_z_mps2": point.accel_mps2[2],
        "kalman_gyro_x_dps": point.gyro_dps[0],
        "kalman_gyro_y_dps": point.gyro_dps[1],
        "kalman_gyro_z_dps": point.gyro_dps[2],
        "raw_accel_x": point.raw_accel[0],
        "raw_accel_y": point.raw_accel[1],
        "raw_accel_z": point.raw_accel[2],
        "raw_gyro_x": point.raw_gyro[0],
        "raw_gyro_y": point.raw_gyro[1],
        "raw_gyro_z": point.raw_gyro[2],
        "motion_intensity_mps2": motion_intensity,
        "rotation_intensity_dps": rotation_intensity,
        "is_stationary": is_stationary,
        "is_moving": is_moving,
        "is_rotating": is_rotating,
        "dominant_motion_axis": dominant_axis(
            point.accel_mps2, threshold=t["dominant_motion_axis_threshold_mps2"]
        ),
        "dominant_rotation_axis": dominant_axis(
            point.gyro_dps, threshold=t["dominant_rotation_axis_threshold_dps"]
        ),
        "quality": quality,
    }


# ---------------------------------------------------------------------------
# Scalar / triple Kalman smoothers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KalmanTuning:
    process_noise: float
    measurement_noise: float
    initial_error: float = DEFAULT_KALMAN_INITIAL_ERROR


class ScalarKalmanFilter:
    """Simple one-dimensional Kalman smoother for streaming sensor channels."""

    def __init__(self, tuning: KalmanTuning) -> None:
        self.tuning = tuning
        self.reset()

    def reset(self) -> None:
        self.estimate = 0.0
        self.error = max(1e-9, self.tuning.initial_error)
        self.initialized = False

    def update(self, measurement: float, dt: float) -> float:
        if not self.initialized:
            self.estimate = measurement
            self.initialized = True
            return self.estimate

        process_noise = max(0.0, self.tuning.process_noise)
        measurement_noise = max(1e-9, self.tuning.measurement_noise)
        self.error += process_noise * max(dt, 1e-6)
        gain = self.error / (self.error + measurement_noise)
        self.estimate += gain * (measurement - self.estimate)
        self.error = (1.0 - gain) * self.error
        return self.estimate


class TripleKalmanFilter:
    """Apply the same scalar Kalman tuning independently to X/Y/Z axes."""

    def __init__(self, tuning: KalmanTuning) -> None:
        self.filters = tuple(ScalarKalmanFilter(tuning) for _ in range(3))

    def reset(self) -> None:
        for filter_ in self.filters:
            filter_.reset()

    def update(self, values: Axis3, dt: float) -> Axis3:
        return tuple(
            filter_.update(value, dt)
            for filter_, value in zip(self.filters, values)
        )


class HampelSpikeGate:
    """Reject isolated single-sample impulses on each axis (median/MAD gate).

    Runs in the raw-count domain so the threshold is independent of the sensor
    range. A sample is replaced by the running median only when it deviates from
    the window median by more than ``n_sigma`` robust standard deviations (or a
    fixed count floor). This removes the firmware batch-boundary glitch (a lone
    spike every batch on one axis) while leaving genuine multi-sample motion,
    whose median tracks the trend, untouched.
    """

    def __init__(self, *, window: int, n_sigma: float, floor: float) -> None:
        self.window = max(3, window)
        self.n_sigma = max(1.0, n_sigma)
        self.floor = max(0.0, floor)
        self.reset()

    def reset(self) -> None:
        self._history = tuple(deque(maxlen=self.window) for _ in range(3))

    def filter(self, values: tuple[int, int, int]) -> Axis3:
        result: list[float] = []
        for axis, value in enumerate(values):
            history = self._history[axis]
            if len(history) >= 3:
                ordered = sorted(history)
                median = ordered[len(ordered) // 2]
                mad = sorted(abs(item - median) for item in ordered)[len(ordered) // 2]
                threshold = max(self.floor, self.n_sigma * 1.4826 * mad)
                if abs(value - median) > threshold:
                    # Isolated impulse: substitute the median and do not let the
                    # glitch enter the window, so the next sample stays clean.
                    result.append(float(median))
                    continue
            history.append(value)
            result.append(float(value))
        return tuple(result)


class ComplementaryAttitudeFilter:
    """Fuse gyro integration with the accelerometer gravity direction.

    Roll/pitch are observable from gravity and stay drift-free; yaw relies on
    gyro integration only (no magnetometer) and slowly drifts. The estimate is
    used to remove gravity from the accelerometer and rotate motion into a
    world frame.
    """

    def __init__(self, *, time_constant_s: float, accel_trust_tol: float) -> None:
        self.time_constant_s = max(1e-3, time_constant_s)
        self.accel_trust_tol = max(0.0, accel_trust_tol)
        self.reset()

    def reset(self) -> None:
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.initialized = False

    def initialize_from_accel(self, accel: Axis3) -> None:
        self.roll, self.pitch = self._accel_angles(accel)
        self.yaw = 0.0
        self.initialized = True

    @staticmethod
    def _accel_angles(accel: Axis3) -> tuple[float, float]:
        ax, ay, az = accel
        roll = math.atan2(ay, az)
        pitch = math.atan2(-ax, math.hypot(ay, az) or 1e-9)
        return roll, pitch

    def update(self, accel: Axis3, gyro_dps: Axis3, dt: float) -> tuple[float, float, float]:
        if not self.initialized:
            self.initialize_from_accel(accel)
            return self.roll, self.pitch, self.yaw

        self.roll += math.radians(gyro_dps[0]) * dt
        self.pitch += math.radians(gyro_dps[1]) * dt
        self.yaw = self._wrap(self.yaw + math.radians(gyro_dps[2]) * dt)

        magnitude = math.sqrt(sum(value * value for value in accel))
        if magnitude > 1e-6:
            deviation = abs(magnitude - GRAVITY_MPS2) / GRAVITY_MPS2
            if deviation <= self.accel_trust_tol:
                roll_acc, pitch_acc = self._accel_angles(accel)
                alpha = self.time_constant_s / (self.time_constant_s + dt)
                self.roll = alpha * self.roll + (1.0 - alpha) * roll_acc
                self.pitch = alpha * self.pitch + (1.0 - alpha) * pitch_acc

        return self.roll, self.pitch, self.yaw

    @staticmethod
    def _wrap(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def gravity_sensor(self) -> Axis3:
        """Gravity reaction vector expressed in the current sensor frame."""
        cr, sr = math.cos(self.roll), math.sin(self.roll)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        return (
            -sp * GRAVITY_MPS2,
            cp * sr * GRAVITY_MPS2,
            cp * cr * GRAVITY_MPS2,
        )

    def rotate_to_world(self, vec: Axis3) -> Axis3:
        """Rotate a body-frame vector into the world frame (ZYX Euler)."""
        cr, sr = math.cos(self.roll), math.sin(self.roll)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        x, y, z = vec
        return (
            (cy * cp) * x + (cy * sp * sr - sy * cr) * y + (cy * sp * cr + sy * sr) * z,
            (sy * cp) * x + (sy * sp * sr + cy * cr) * y + (sy * sp * cr - cy * sr) * z,
            (-sp) * x + (cp * sr) * y + (cp * cr) * z,
        )


class AxisKinematicKalman:
    """Constant-acceleration Kalman filter for one world axis.

    State is [position, velocity]; the measured linear acceleration enters as a
    control input. A zero-velocity update (ZUPT) may be applied when the ring is
    detected stationary to bound integration drift.
    """

    def __init__(
        self,
        *,
        accel_noise: float,
        initial_pos_var: float,
        initial_vel_var: float,
    ) -> None:
        self.accel_noise = max(1e-9, accel_noise)
        self.initial_pos_var = max(0.0, initial_pos_var)
        self.initial_vel_var = max(0.0, initial_vel_var)
        self.reset()

    def reset(self) -> None:
        self.pos = 0.0
        self.vel = 0.0
        self.p00 = self.initial_pos_var
        self.p01 = 0.0
        self.p10 = 0.0
        self.p11 = self.initial_vel_var

    def predict(self, accel: float, dt: float) -> None:
        dt = max(dt, 1e-6)
        # State propagation: x = F x + B u, with F = [[1, dt], [0, 1]].
        self.pos += self.vel * dt + 0.5 * accel * dt * dt
        self.vel += accel * dt

        # Covariance propagation: P = F P F^T + Q.
        fp00 = self.p00 + dt * self.p10
        fp01 = self.p01 + dt * self.p11
        fp10 = self.p10
        fp11 = self.p11
        p00 = fp00 + dt * fp01
        p01 = fp01
        p10 = fp10 + dt * fp11
        p11 = fp11

        variance = self.accel_noise * self.accel_noise
        g0 = 0.5 * dt * dt
        g1 = dt
        self.p00 = p00 + g0 * g0 * variance
        self.p01 = p01 + g0 * g1 * variance
        self.p10 = p10 + g1 * g0 * variance
        self.p11 = p11 + g1 * g1 * variance

    def update_zero_velocity(self, measurement_noise: float) -> None:
        # Measurement z = 0 on velocity only, so H = [0, 1].
        innovation_cov = self.p11 + max(1e-12, measurement_noise)
        gain_pos = self.p01 / innovation_cov
        gain_vel = self.p11 / innovation_cov
        residual = -self.vel
        self.pos += gain_pos * residual
        self.vel += gain_vel * residual

        p00 = self.p00 - gain_pos * self.p10
        p01 = self.p01 - gain_pos * self.p11
        p10 = (1.0 - gain_vel) * self.p10
        p11 = (1.0 - gain_vel) * self.p11
        self.p00, self.p01, self.p10, self.p11 = p00, p01, p10, p11


# ---------------------------------------------------------------------------
# MotionIntegrator — the full pipeline
# ---------------------------------------------------------------------------


class MotionIntegrator:
    """Convert raw six-axis samples into gravity-compensated motion estimates.

    Pipeline: unit conversion -> stationary bias calibration -> complementary
    attitude estimation -> gravity removal and rotation into a world frame ->
    per-axis constant-acceleration Kalman filter with zero-velocity updates.
    """

    def __init__(
        self,
        *,
        sample_rate_hz: int,
        accel_range_g: int,
        gyro_range_dps: int,
        calibration_seconds: float = DEFAULT_CALIBRATION_SECONDS,
    ) -> None:
        self.sample_rate_hz = max(1, sample_rate_hz)
        self.accel_scale = (max(1, accel_range_g) * GRAVITY_MPS2) / INT16_FULL_SCALE
        self.gyro_scale = max(1, gyro_range_dps) / INT16_FULL_SCALE
        self.calibration_sample_target = max(
            1,
            int(self.sample_rate_hz * max(0.0, calibration_seconds)),
        )
        self._outbox: list[tuple[str, Any]] = []
        self.reset()

    @property
    def is_calibrated(self) -> bool:
        return self._calibration_count >= self.calibration_sample_target

    @property
    def calibration_progress(self) -> float:
        return min(1.0, self._calibration_count / self.calibration_sample_target)

    def reset(self) -> None:
        self._accel_bias = (0.0, 0.0, 0.0)
        self._gyro_bias = (0.0, 0.0, 0.0)
        self._accel_sum = [0.0, 0.0, 0.0]
        self._gyro_sum = [0.0, 0.0, 0.0]
        self._calibration_count = 0
        self._last_timestamp_ms: int | None = None
        self._gyro_gate = HampelSpikeGate(
            window=GYRO_SPIKE_WINDOW,
            n_sigma=GYRO_SPIKE_SIGMA,
            floor=GYRO_SPIKE_FLOOR_COUNTS,
        )
        self._reset_filters()

    def process(self, sample: Any, sequence: int) -> ImuPoint:
        raw_accel = (sample.accel_x, sample.accel_y, sample.accel_z)
        raw_gyro = (sample.gyro_x, sample.gyro_y, sample.gyro_z)
        # De-spike the gyro before it reaches calibration, attitude and display;
        # raw_gyro is preserved unchanged for logging/diagnostics.
        gyro_counts = self._gyro_gate.filter(raw_gyro)
        measured_accel = tuple(value * self.accel_scale for value in raw_accel)
        measured_gyro = tuple(value * self.gyro_scale for value in gyro_counts)

        dt = self._sample_dt(sample.timestamp_ms)

        if not self.is_calibrated:
            self._add_calibration_sample(measured_accel, measured_gyro)
            return self._zero_point(sample, sequence, raw_accel, raw_gyro)

        corrected_accel = tuple(
            measured_accel[index] - self._accel_bias[index] for index in range(3)
        )
        corrected_gyro = tuple(
            measured_gyro[index] - self._gyro_bias[index] for index in range(3)
        )

        stationary = self._is_stationary(corrected_accel, corrected_gyro)
        if stationary:
            # When still the true angular rate is ~0, so the measured gyro is
            # essentially bias. Nudge the bias estimate toward it (EMA) to track
            # slow zero-offset drift, then refresh the corrected gyro.
            self._gyro_bias = tuple(
                self._gyro_bias[index]
                + GYRO_BIAS_ADAPT_RATE * (measured_gyro[index] - self._gyro_bias[index])
                for index in range(3)
            )
            corrected_gyro = tuple(
                measured_gyro[index] - self._gyro_bias[index] for index in range(3)
            )

        self._attitude.update(corrected_accel, corrected_gyro, dt)
        gravity = self._attitude.gravity_sensor()
        linear_sensor = tuple(
            corrected_accel[index] - gravity[index] for index in range(3)
        )
        linear_world = self._attitude.rotate_to_world(linear_sensor)

        position: list[float] = []
        velocity: list[float] = []
        for index, axis_filter in enumerate(self._axis_filters):
            axis_filter.predict(linear_world[index], dt)
            if stationary:
                axis_filter.update_zero_velocity(ZUPT_VELOCITY_MEAS_NOISE)
            position.append(axis_filter.pos)
            velocity.append(axis_filter.vel)

        filtered_accel = self._accel_filter.update(linear_world, dt)
        filtered_gyro = self._gyro_filter.update(corrected_gyro, dt)

        return ImuPoint(
            host_time_s=time.monotonic(),
            sequence=sequence,
            timestamp_ms=sample.timestamp_ms,
            accel_mps2=filtered_accel,
            gyro_dps=filtered_gyro,
            velocity_mps=tuple(velocity),
            position_m=tuple(position),
            raw_accel=raw_accel,
            raw_gyro=raw_gyro,
        )

    def drain_messages(self) -> list[tuple[str, Any]]:
        if not self._outbox:
            return []
        messages = self._outbox
        self._outbox = []
        return messages

    def _zero_point(
        self,
        sample: Any,
        sequence: int,
        raw_accel: tuple[int, int, int],
        raw_gyro: tuple[int, int, int],
    ) -> ImuPoint:
        zero = (0.0, 0.0, 0.0)
        return ImuPoint(
            host_time_s=time.monotonic(),
            sequence=sequence,
            timestamp_ms=sample.timestamp_ms,
            accel_mps2=zero,
            gyro_dps=zero,
            velocity_mps=zero,
            position_m=zero,
            raw_accel=raw_accel,
            raw_gyro=raw_gyro,
        )

    def _sample_dt(self, timestamp_ms: int) -> float:
        fallback = 1.0 / self.sample_rate_hz
        if self._last_timestamp_ms is None:
            self._last_timestamp_ms = timestamp_ms
            return fallback

        dt = (timestamp_ms - self._last_timestamp_ms) / 1000.0
        if dt <= 0.0 or dt > 1.0:
            # Backwards/jumped device timestamp (batch-boundary glitch): keep a
            # monotonic clock by advancing one nominal step instead of trusting
            # the reported value.
            self._last_timestamp_ms += round(fallback * 1000.0)
            return fallback
        self._last_timestamp_ms = timestamp_ms
        return dt

    def _add_calibration_sample(self, accel: Axis3, gyro: Axis3) -> None:
        for index in range(3):
            self._accel_sum[index] += accel[index]
            self._gyro_sum[index] += gyro[index]
        self._calibration_count += 1

        if self._calibration_count >= self.calibration_sample_target:
            self._finalize_calibration()

    def _finalize_calibration(self) -> None:
        count = float(self._calibration_count)
        mean_accel = tuple(value / count for value in self._accel_sum)
        self._gyro_bias = tuple(value / count for value in self._gyro_sum)
        # Fresh filters, then seed attitude from the average rest orientation.
        self._reset_filters()
        self._attitude.initialize_from_accel(mean_accel)
        gravity = self._attitude.gravity_sensor()
        # Residual after removing modelled gravity is the accelerometer bias.
        self._accel_bias = tuple(
            mean_accel[index] - gravity[index] for index in range(3)
        )

    def _is_stationary(self, accel: Axis3, gyro: Axis3) -> bool:
        accel_magnitude = math.sqrt(sum(value * value for value in accel))
        gyro_magnitude = math.sqrt(sum(value * value for value in gyro))
        return (
            abs(accel_magnitude - GRAVITY_MPS2) < ZUPT_ACCEL_TOL_MPS2
            and gyro_magnitude < ZUPT_GYRO_TOL_DPS
        )

    def _reset_filters(self) -> None:
        self._accel_filter = TripleKalmanFilter(
            KalmanTuning(
                process_noise=ACCEL_KALMAN_PROCESS_NOISE,
                measurement_noise=ACCEL_KALMAN_MEASUREMENT_NOISE,
            )
        )
        self._gyro_filter = TripleKalmanFilter(
            KalmanTuning(
                process_noise=GYRO_KALMAN_PROCESS_NOISE,
                measurement_noise=GYRO_KALMAN_MEASUREMENT_NOISE,
            )
        )
        self._attitude = ComplementaryAttitudeFilter(
            time_constant_s=ATTITUDE_TIME_CONSTANT_S,
            accel_trust_tol=ATTITUDE_ACCEL_TRUST_TOL,
        )
        self._axis_filters = tuple(
            AxisKinematicKalman(
                accel_noise=KINEMATIC_ACCEL_NOISE_MPS2,
                initial_pos_var=KINEMATIC_INITIAL_POS_VAR,
                initial_vel_var=KINEMATIC_INITIAL_VEL_VAR,
            )
            for _ in range(3)
        )


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------


class HandoffImuCsvLogger:
    """Writes processed IMU data points to a CSV file.

    The CSV schema matches the structured output of imu_point_payload() plus
    a sample_index_in_run counter.

    Usage:
        logger = HandoffImuCsvLogger(Path("output.csv"))
        logger.write(payload_dict)
        logger.close()
    """

    COLUMNS = [
        "host_time_s",
        "sequence",
        "timestamp_ms",
        "kalman_accel_x_mps2",
        "kalman_accel_y_mps2",
        "kalman_accel_z_mps2",
        "kalman_gyro_x_dps",
        "kalman_gyro_y_dps",
        "kalman_gyro_z_dps",
        "raw_accel_x",
        "raw_accel_y",
        "raw_accel_z",
        "raw_gyro_x",
        "raw_gyro_y",
        "raw_gyro_z",
        "motion_intensity_mps2",
        "rotation_intensity_dps",
        "is_stationary",
        "is_moving",
        "is_rotating",
        "dominant_motion_axis",
        "dominant_rotation_axis",
        "quality",
        "sample_index_in_run",
    ]

    def __init__(self, path: Path) -> None:
        """Open a CSV file for writing IMU data.

        Args:
            path: Destination file path. Parent directories are created automatically.
        """
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file, fieldnames=self.COLUMNS, extrasaction="ignore"
        )
        self._writer.writeheader()
        print(f"[KalmanFilter] CSV logger opened: {self.path}")

    def write(self, payload: dict[str, Any]) -> None:
        """Write one processed IMU sample row."""
        self._writer.writerow(payload)

    def flush(self) -> None:
        """Flush buffered data to disk."""
        self._file.flush()

    def close(self) -> None:
        """Flush and close the CSV file."""
        self._file.flush()
        self._file.close()
        print(f"[KalmanFilter] CSV logger closed: {self.path}")


# ---------------------------------------------------------------------------
# KalmanProcessor — high-level processing pipeline
# ---------------------------------------------------------------------------


class KalmanProcessor:
    """Encapsulates the Kalman IMU processing pipeline.

    This class manages the lifecycle of :class:`MotionIntegrator`: initialization,
    per-sample processing, and structured output generation. It operates
    independently of BLE connections and visualization.

    Typical usage:
        processor = KalmanProcessor()
        processor.start(sample_rate_hz=50, accel_range_g=2, gyro_range_dps=250)

        for sample in batch.samples:
            result = processor.process(sample, sequence=seq_num)
            # result is a dict with all Kalman-processed fields

        messages = processor.drain_messages()
        processor.reset()

    Args:
        calibration_seconds: Duration of initial calibration phase in seconds.
    """

    def __init__(self, calibration_seconds: float = DEFAULT_CALIBRATION_SECONDS) -> None:
        self._calibration_seconds = calibration_seconds
        self._integrator: MotionIntegrator | None = None
        self._sample_index: int = 0
        self._started: bool = False

    @property
    def is_started(self) -> bool:
        """Whether the processor has been initialized and is ready to process."""
        return self._started

    @property
    def sample_count(self) -> int:
        """Number of samples processed since last start/reset."""
        return self._sample_index

    def start(
        self,
        sample_rate_hz: int,
        accel_range_g: int,
        gyro_range_dps: int,
        calibration_seconds: float | None = None,
    ) -> None:
        """Initialize the MotionIntegrator for a new collection session.

        Must be called before process(). Can be called again to reset the
        integrator for a new session.

        Args:
            sample_rate_hz: IMU sample rate reported by the ring.
            accel_range_g: Accelerometer full-scale range in g.
            gyro_range_dps: Gyroscope full-scale range in deg/s.
            calibration_seconds: Override the default calibration duration.
        """
        cal = (
            calibration_seconds
            if calibration_seconds is not None
            else self._calibration_seconds
        )
        self._integrator = MotionIntegrator(
            sample_rate_hz=sample_rate_hz,
            accel_range_g=accel_range_g,
            gyro_range_dps=gyro_range_dps,
            calibration_seconds=cal,
        )
        self._sample_index = 0
        self._started = True
        print(
            f"[KalmanFilter] Processor started: "
            f"{sample_rate_hz} Hz, ±{accel_range_g} g, ±{gyro_range_dps} dps, "
            f"calibration={cal}s"
        )

    def process(self, sample: Any, sequence: int) -> dict[str, Any]:
        """Process a single raw IMU sample through the Kalman filter.

        Args:
            sample: A raw sensor sample object from SDK's SensorDataBatch.samples.
            sequence: The monotonic sequence number for this sample.

        Returns:
            Structured dict containing all Kalman-filtered values, motion state
            classification, and quality indicators. Includes 'sample_index_in_run'.

        Raises:
            RuntimeError: If start() has not been called.
        """
        if not self._started or self._integrator is None:
            raise RuntimeError(
                "KalmanProcessor.start() must be called before process()"
            )

        self._sample_index += 1
        point = self._integrator.process(sample, sequence=sequence)
        payload = imu_point_payload(point)
        payload["sample_index_in_run"] = self._sample_index
        return payload

    def process_batch(self, batch: Any) -> list[dict[str, Any]]:
        """Process an entire SensorDataBatch through the Kalman filter.

        Args:
            batch: A SensorDataBatch object with .samples list and
                   .sequence_start int.

        Returns:
            List of structured dicts, one per sample in the batch.
        """
        results = []
        for index, sample in enumerate(batch.samples):
            result = self.process(sample, sequence=batch.sequence_start + index)
            results.append(result)
        return results

    def drain_messages(self) -> list[tuple[str, str]]:
        """Drain any queued messages from the MotionIntegrator.

        Returns:
            List of (kind, text) tuples. Empty if no messages or not started.
        """
        if self._integrator is None:
            return []
        return list(self._integrator.drain_messages())

    def reset(self) -> None:
        """Reset the processor state, discarding the current integrator.

        Call start() again before processing new data.
        """
        self._integrator = None
        self._sample_index = 0
        self._started = False
        print("[KalmanFilter] Processor reset.")


# ---------------------------------------------------------------------------
# Standalone entry point for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[KalmanFilter] Module loaded successfully.")
    print(f"  Thresholds: {MOTION_THRESHOLDS}")
    print(f"  Exported symbols: {__all__}")
    print("  To use: instantiate KalmanProcessor and call .start() then .process()")
