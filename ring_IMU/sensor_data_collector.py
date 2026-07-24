"""BLE IMU 数据采集模块：连接戒指设备、启动传感器上报、读取原始 IMU 数据。

将 Ring Sound SDK 的 BLE 连接与 IMU 传感器数据采集封装为独立模块，
不依赖手势识别或可视化功能。

用法::

    from ring_IMU.sensor_data_collector import SensorDataCollector

    async with SensorDataCollector(ring_address="AA:BB:CC:DD:EE:FF") as collector:
        async for batch in collector.read_stream(duration_s=10.0):
            print(batch)
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Callable

from utils.sdk_loader import get_sdk
from utils.ble_connection import (
    DEFAULT_CONNECT_ATTEMPTS,
    DEFAULT_CONNECT_RETRY_DELAY,
    connect_ring,
    disconnect_ring,
)
from ring_IMU.kalman_filter import (
    DEFAULT_CALIBRATION_SECONDS,
    ImuPoint,
    MotionIntegrator,
)


# IMU 上报默认参数（连接/重试相关默认值统一由 utils.ble_connection 提供）
DEFAULT_PACKET_TIMEOUT = 5.0


class SensorDataCollector:
    """BLE IMU 数据采集器（ring_talking 风格）。"""

    def __init__(
        self,
        ring_address: str | None = None,
        *,
        connect_attempts: int = DEFAULT_CONNECT_ATTEMPTS,
        retry_delay: float = DEFAULT_CONNECT_RETRY_DELAY,
        packet_timeout: float = DEFAULT_PACKET_TIMEOUT,
        auto_prepare: bool = False,
        calibration_seconds: float = DEFAULT_CALIBRATION_SECONDS,
        on_status: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化 IMU 数据采集器。

        Parameters
        ----------
        ring_address : str | None
            戒指 BLE MAC 地址，格式如 "AA:BB:CC:DD:EE:FF"。
        connect_attempts : int
            最大连接重试次数。
        retry_delay : float
            重试间隔秒数。
        packet_timeout : float
            等待数据包的超时秒数。
        auto_prepare : bool
            戒指忙碌（DEVICE_BUSY）时是否自动等待用户单击进入手势模式后重试上报。
        calibration_seconds : float
            Kalman 处理的初始静止校准时长（秒），用于 :meth:`read_points`。
        on_status : Callable[[str], None] | None
            可选状态回调，用于把连接/校准等提示同步给上层（如 GUI）。
            本模块只做“数据采集 + Kalman 处理”，不关心提示如何呈现。
        **kwargs
            兼容扩展参数（忽略）。
        """
        self.ring_address = ring_address
        self._connect_attempts = connect_attempts
        self._retry_delay = retry_delay
        self._packet_timeout = packet_timeout
        self._auto_prepare = auto_prepare
        self._calibration_seconds = calibration_seconds
        self._on_status = on_status
        self._ring: Any = None
        self._reporting = False
        self._start_info: Any = None
        self._integrator: MotionIntegrator | None = None

    # ------------------------------------------------------------------
    # BLE 连接管理
    # ------------------------------------------------------------------

    def _notify(self, message: str) -> None:
        """打印并（如提供）回调一条状态提示。"""
        print(f"[SensorDataCollector] {message}")
        if self._on_status is not None:
            try:
                self._on_status(message)
            except Exception:
                pass

    async def _start_sensor_report(self, ring: Any) -> Any:
        """启动 IMU 上报；若 auto_prepare 且设备忙碌则等待单击后重试。"""
        sdk = get_sdk()
        try:
            return await sdk.start_sensor_report(ring, timeout_s=5.0)
        except sdk.DeviceError as exc:
            if (
                not self._auto_prepare
                or exc.error_code != int(sdk.ErrorCode.DEVICE_BUSY)
            ):
                raise
        self._notify("戒指忙碌，请单击戒指进入手势模式 ...")
        await sdk.wait_sensor_key_single_press_event(ring, timeout_s=30.0)
        self._notify("已检测到单击，正在启动 IMU ...")
        return await sdk.start_sensor_report(ring, timeout_s=5.0)

    async def connect(self) -> None:
        """连接戒指设备，启动 IMU 数据上报。

        带重试机制，适应戒指低频广播场景（Windows 下尤其需要）。
        """
        if self._ring is not None:
            print("[SensorDataCollector] 已连接，跳过重复连接")
            return

        ring = await connect_ring(
            self.ring_address,
            connect_attempts=self._connect_attempts,
            retry_delay=self._retry_delay,
            log_prefix="[SensorDataCollector]",
        )
        self._ring = ring

        # 启动 IMU 传感器上报
        try:
            start_info = await self._start_sensor_report(ring)
            self._reporting = True
            self._start_info = start_info
            self._integrator = MotionIntegrator(
                sample_rate_hz=start_info.sample_rate_hz,
                accel_range_g=start_info.accel_range_g,
                gyro_range_dps=start_info.gyro_range_dps,
                calibration_seconds=self._calibration_seconds,
            )
            print(
                f"[SensorDataCollector] IMU 上报已启动 "
                f"(sample_rate={start_info.sample_rate_hz}Hz, "
                f"accel_range={start_info.accel_range_g}g, "
                f"gyro_range={start_info.gyro_range_dps}dps)"
            )
        except Exception as exc:
            print(f"[SensorDataCollector] 启动 IMU 上报失败: {exc!r}")
            await disconnect_ring(ring, log_prefix="[SensorDataCollector]")
            self._ring = None
            self._integrator = None
            self._start_info = None
            raise

    async def disconnect(self) -> None:
        """停止 IMU 上报并断开连接。"""
        if self._ring is None:
            return

        sdk = get_sdk()
        try:
            if self._reporting:
                await sdk.stop_sensor_report(self._ring)
                self._reporting = False
                print("[SensorDataCollector] IMU 上报已停止")
        except Exception as exc:
            print(f"[SensorDataCollector] 停止上报时出错: {exc!r}")
        finally:
            await disconnect_ring(self._ring, log_prefix="[SensorDataCollector]")
            self._ring = None
            self._integrator = None
            self._start_info = None

    # ------------------------------------------------------------------
    # 处理状态（供上层只读查询）
    # ------------------------------------------------------------------

    @property
    def start_info(self) -> Any:
        """上报启动信息（sample_rate_hz / accel_range_g / gyro_range_dps）。"""
        return self._start_info

    @property
    def is_calibrated(self) -> bool:
        """Kalman 静止校准是否完成。"""
        return self._integrator is not None and self._integrator.is_calibrated

    @property
    def calibration_progress(self) -> float:
        """校准进度 0.0-1.0。"""
        if self._integrator is None:
            return 0.0
        return self._integrator.calibration_progress

    def drain_messages(self) -> list[tuple[str, Any]]:
        """取出 Kalman 处理器排队的提示消息（如坐标对齐提示）。"""
        if self._integrator is None:
            return []
        return self._integrator.drain_messages()

    # ------------------------------------------------------------------
    # 数据读取
    # ------------------------------------------------------------------

    async def read_batch(self) -> list[dict[str, Any]]:
        """读取一批原始 IMU 数据。

        Returns
        -------
        list[dict[str, Any]]
            本批次所有 IMU 采样点，每个点包含 accel、gyro 等原始数据。
        """
        if self._ring is None or not self._reporting:
            raise RuntimeError("[SensorDataCollector] 未连接或未启动上报")

        sdk = get_sdk()
        batch = await sdk.wait_sensor_data(self._ring, timeout_s=self._packet_timeout)
        samples = []
        for idx, sample in enumerate(batch.samples):
            samples.append({
                "sequence": batch.sequence_start + idx,
                "host_time_ms": int(time.time() * 1000),
                **sample.__dict__,
            })
        return samples

    async def read_stream(
        self, duration_s: float | None = None
    ) -> AsyncIterator[list[dict[str, Any]]]:
        """异步生成器，连续产生 IMU 数据批次。

        Parameters
        ----------
        duration_s : float | None
            采集持续时间（秒）。为 None 时无限采集直到手动停止。

        Yields
        ------
        list[dict[str, Any]]
            每批次的 IMU 采样点列表。
        """
        if self._ring is None or not self._reporting:
            raise RuntimeError("[SensorDataCollector] 未连接或未启动上报")

        deadline = (time.monotonic() + duration_s) if duration_s else None
        samples_total = 0

        print(
            f"[SensorDataCollector] 开始流式采集"
            + (f" (duration={duration_s}s)" if duration_s else " (无限)")
        )

        try:
            while True:
                if deadline and time.monotonic() >= deadline:
                    break
                batch = await self.read_batch()
                samples_total += len(batch)
                yield batch
        finally:
            print(f"[SensorDataCollector] 流式采集结束，共 {samples_total} 个采样点")

    async def read_points(
        self, duration_s: float | None = None
    ) -> AsyncIterator[ImuPoint]:
        """异步生成器，连续产出经 Kalman 处理的 :class:`ImuPoint`。

        内部用 :class:`MotionIntegrator` 逐样本处理（单位换算、静止校准、
        重力移除、姿态估计与逐轴 Kalman 滤波）。校准进度/提示消息可通过
        :attr:`is_calibrated` / :attr:`calibration_progress` / :meth:`drain_messages`
        查询，供上层（如可视化）在逐点消费时读取。

        Parameters
        ----------
        duration_s : float | None
            采集持续时间（秒）。为 None 时无限采集直到手动停止。

        Yields
        ------
        ImuPoint
            逐个处理后的 IMU 采样点。
        """
        if self._ring is None or not self._reporting or self._integrator is None:
            raise RuntimeError("[SensorDataCollector] 未连接或未启动上报")

        sdk = get_sdk()
        deadline = (time.monotonic() + duration_s) if duration_s else None
        points_total = 0

        print(
            "[SensorDataCollector] 开始处理点流采集"
            + (f" (duration={duration_s}s)" if duration_s else " (无限)")
        )

        try:
            while True:
                if deadline and time.monotonic() >= deadline:
                    break
                try:
                    batch = await sdk.wait_sensor_data(
                        self._ring, timeout_s=self._packet_timeout
                    )
                except sdk.TimeoutError:
                    self._notify("等待 IMU 数据 ...")
                    continue
                for index, sample in enumerate(batch.samples):
                    point = self._integrator.process(
                        sample, sequence=batch.sequence_start + index
                    )
                    points_total += 1
                    yield point
        finally:
            print(f"[SensorDataCollector] 点流采集结束，共 {points_total} 个处理点")

    # ------------------------------------------------------------------
    # 上下文管理器
    # ------------------------------------------------------------------

    async def __aenter__(self) -> SensorDataCollector:
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.disconnect()


__all__ = ["SensorDataCollector", "ImuPoint"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="从戒指采集 IMU 传感器数据")
    parser.add_argument("address", help="戒指 BLE MAC 地址 (AA:BB:CC:DD:EE:FF)")
    parser.add_argument(
        "-d", "--duration", type=float, default=10.0,
        help="采集时长（秒），默认 10",
    )
    args = parser.parse_args()

    async def _main() -> None:
        async with SensorDataCollector(ring_address=args.address) as collector:
            async for batch in collector.read_stream(duration_s=args.duration):
                for sample in batch:
                    print(f"  seq={sample['sequence']} data={sample}")

    asyncio.run(_main())
