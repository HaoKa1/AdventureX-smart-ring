"""IMU real-time visualization module for Ring sensor data.

This module is visualization-only. All BLE connection, sensor-report control and
Kalman processing are delegated to
:class:`~ring_IMU.sensor_data_collector.SensorDataCollector`; this file merely
drives that collector from a background thread and renders the resulting
:class:`~ring_IMU.kalman_filter.ImuPoint` stream as live charts.

Components:
    - ``_RingWorker``: background thread owning an asyncio loop; drives a
      ``SensorDataCollector`` and forwards processed points/state to a
      thread-safe queue. It contains no BLE/SDK/Kalman logic of its own.
    - ``ChartView``: a Tkinter widget rendering a real-time 3-axis line chart.
    - ``ImuVisualizer``: orchestrates the Tkinter window, worker and charts.

Usage:
    from ring_IMU.visualizer import ImuVisualizer

    viz = ImuVisualizer(address="AA:BB:CC:DD:EE:FF")
    viz.run()  # Blocking — opens the Tkinter window

Run as a script:
    python -m ring_IMU.visualizer --address "AA:BB:CC:DD:EE:FF"
"""

from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import Future
import math
import queue
import threading
from typing import Any

from ring_IMU.sensor_data_collector import SensorDataCollector
from ring_IMU.kalman_filter import ImuPoint, imu_point_payload

__all__ = [
    "ImuVisualizer",
    "ChartView",
]


DEFAULT_WINDOW_SECONDS = 10.0
DEFAULT_PACKET_TIMEOUT_S = 5.0
DEFAULT_CALIBRATION_SECONDS = 1.0


Axis3 = tuple[float, float, float]


# ---------------------------------------------------------------------------
# BLE worker — owns an asyncio loop on a background thread
# ---------------------------------------------------------------------------


class _RingWorker:
    """Own the asyncio loop on a background thread and drive a SensorDataCollector.

    All BLE connection, sensor-report control and Kalman processing live in
    :class:`~ring_IMU.sensor_data_collector.SensorDataCollector`; this worker only
    drives it and forwards results to a thread-safe queue for the Tkinter GUI:
      - ("status", str) / ("error", str): messages
      - ("connected", bool): connection state
      - ("sensor_started", dict) / ("sensor_stopped", None): stream lifecycle
      - ("calibration", float): calibration progress 0.0-1.0
      - ("align", str): messages drained from the collector
      - ("point", ImuPoint): a processed IMU sample
    """

    def __init__(
        self,
        events: "queue.Queue[tuple[str, Any]]",
        *,
        auto_prepare: bool,
        packet_timeout_s: float,
        calibration_seconds: float,
    ) -> None:
        self.events = events
        self._auto_prepare = auto_prepare
        self._packet_timeout_s = packet_timeout_s
        self._calibration_seconds = calibration_seconds
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.collector: SensorDataCollector | None = None
        self.stream_task: asyncio.Task[None] | None = None
        self.stream_stop: asyncio.Event | None = None
        self._closed = False
        self.thread.start()

    # -- public API (thread-safe, called from the GUI thread) ---------------

    def connect(self, address: str) -> None:
        self._submit(self._connect(address))

    def disconnect(self) -> None:
        self._submit(self._disconnect())

    def start_stream(self) -> None:
        self._submit(self._start_stream())

    def stop_stream(self) -> None:
        self._submit(self._stop_stream())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        future = self._submit(self._disconnect())
        try:
            future.result(timeout=5.0)
        except Exception:
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5.0)

    # -- loop plumbing ------------------------------------------------------

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
        pending = asyncio.all_tasks(self.loop)
        for task in pending:
            task.cancel()
        if pending:
            self.loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        self.loop.close()

    def _submit(self, coro: Any) -> "Future[Any]":
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def _post(self, kind: str, payload: Any) -> None:
        self.events.put((kind, payload))

    # -- coroutines (run on the worker loop) --------------------------------

    async def _connect(self, address: str) -> None:
        if not address:
            self._post("error", "BLE address is required.")
            return
        await self._disconnect()
        self._post("status", f"Connecting to {address}...")
        # 全部 BLE/上报/Kalman 交给 SensorDataCollector；GUI 只保留单次连接
        # 语义（connect_attempts=1）与自有的事件回报流。状态提示通过 on_status
        # 回调转发到 GUI 队列。
        collector = SensorDataCollector(
            ring_address=address,
            connect_attempts=1,
            packet_timeout=self._packet_timeout_s,
            auto_prepare=self._auto_prepare,
            calibration_seconds=self._calibration_seconds,
            on_status=lambda message: self._post("status", message),
        )
        try:
            await collector.connect()
        except Exception as exc:
            self._post("connected", False)
            self._post("error", f"Connect failed: {exc}")
            return
        self.collector = collector
        info = collector.start_info
        self._post("connected", True)
        if info is not None:
            self._post(
                "sensor_started",
                {
                    "sample_rate_hz": info.sample_rate_hz,
                    "accel_range_g": info.accel_range_g,
                    "gyro_range_dps": info.gyro_range_dps,
                },
            )
        self._post("status", "Connected.")

    async def _disconnect(self) -> None:
        await self._stop_stream()
        if self.collector is not None:
            try:
                await self.collector.disconnect()
            except Exception as exc:
                self._post("error", f"Disconnect warning: {exc}")
            finally:
                self.collector = None
        self._post("connected", False)

    async def _start_stream(self) -> None:
        if self.collector is None:
            self._post("error", "Connect to a ring before starting IMU.")
            return
        if self.stream_task and not self.stream_task.done():
            self._post("status", "IMU stream is already running.")
            return
        self.stream_stop = asyncio.Event()
        self.stream_task = asyncio.create_task(self._consume_points())

    async def _stop_stream(self) -> None:
        if self.stream_stop is not None:
            self.stream_stop.set()
        task = self.stream_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.gather(task, return_exceptions=True),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                pass
        self.stream_task = None
        self.stream_stop = None

    async def _consume_points(self) -> None:
        """Consume processed points from the collector and forward to the GUI."""
        assert self.collector is not None
        assert self.stream_stop is not None
        self._post("status", "IMU running. Hold still during calibration.")
        calibration_done = False
        try:
            async for point in self.collector.read_points():
                if self.stream_stop.is_set():
                    break
                self._post("point", point)
                for kind, text in self.collector.drain_messages():
                    self._post(kind, text)
                if not self.collector.is_calibrated:
                    self._post("calibration", self.collector.calibration_progress)
                elif not calibration_done:
                    calibration_done = True
                    self._post(
                        "status", "Calibration complete. Kalman filtering active."
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._post("error", f"IMU stream stopped: {exc}")
        finally:
            self._post("sensor_stopped", None)
            self._post("status", "IMU stopped.")


# ---------------------------------------------------------------------------
# ChartView — Tkinter real-time 3-axis line chart
# ---------------------------------------------------------------------------


def _make_chart_view() -> type:
    """Build the ChartView class lazily (imports tkinter only when needed)."""
    import tkinter as tk
    from tkinter import ttk

    class ChartView(ttk.Frame):
        COLORS = ("#d62728", "#2ca02c", "#1f77b4")
        AXES = ("X", "Y", "Z")

        def __init__(
            self,
            master: "tk.Widget",
            *,
            title: str,
            unit: str,
            window_seconds: float = DEFAULT_WINDOW_SECONDS,
        ) -> None:
            super().__init__(master)
            self.window_seconds = window_seconds
            self.title = title
            self.unit = unit
            self._start_time: float | None = None
            self._points: "deque[tuple[float, Axis3]]" = deque()

            header = ttk.Frame(self)
            header.pack(fill=tk.X)
            ttk.Label(header, text=title, font=("", 10, "bold")).pack(side=tk.LEFT)
            ttk.Label(header, text=unit).pack(side=tk.RIGHT)

            self.canvas = tk.Canvas(
                self, height=160, background="white", highlightthickness=1
            )
            self.canvas.pack(fill=tk.BOTH, expand=True, pady=(3, 0))
            self.canvas.bind("<Configure>", lambda _event: self.draw())

        def clear(self) -> None:
            self._start_time = None
            self._points.clear()
            self.draw()

        def add(self, host_time_s: float, values: Axis3) -> None:
            if self._start_time is None:
                self._start_time = host_time_s
            relative_time = host_time_s - self._start_time
            self._points.append((relative_time, values))
            min_time = relative_time - self.window_seconds
            while self._points and self._points[0][0] < min_time:
                self._points.popleft()

        def draw(self) -> None:
            canvas = self.canvas
            canvas.delete("all")
            width = max(1, canvas.winfo_width())
            height = max(1, canvas.winfo_height())
            left, right, top, bottom = 42, 10, 10, 24
            plot_width = max(1, width - left - right)
            plot_height = max(1, height - top - bottom)

            x0 = left
            y0 = top + plot_height
            canvas.create_line(x0, top, x0, y0, fill="#bbbbbb")
            canvas.create_line(x0, y0, width - right, y0, fill="#bbbbbb")

            if not self._points:
                canvas.create_text(
                    width / 2, height / 2, text="No data", fill="#777777"
                )
                return

            last_time = self._points[-1][0]
            first_time = max(0.0, last_time - self.window_seconds)
            values = [value for _t, triple in self._points for value in triple]
            low = min(values)
            high = max(values)
            if math.isclose(low, high, abs_tol=1e-9):
                pad = max(1.0, abs(low) * 0.1)
                low -= pad
                high += pad
            else:
                pad = (high - low) * 0.12
                low -= pad
                high += pad

            zero_y = self._scale_y(0.0, low, high, top, plot_height)
            if top <= zero_y <= y0:
                canvas.create_line(x0, zero_y, width - right, zero_y, fill="#eeeeee")

            canvas.create_text(
                3, top + 4, text=f"{high:.2f}", anchor="nw", fill="#777777"
            )
            canvas.create_text(
                3, y0 - 14, text=f"{low:.2f}", anchor="nw", fill="#777777"
            )

            for axis_index, color in enumerate(self.COLORS):
                coords: list[float] = []
                for t_s, triple in self._points:
                    x = x0 + ((t_s - first_time) / self.window_seconds) * plot_width
                    y = self._scale_y(triple[axis_index], low, high, top, plot_height)
                    coords.extend([x, y])
                if len(coords) >= 4:
                    canvas.create_line(*coords, fill=color, width=2)

            legend_x = left + 8
            for axis, color in zip(self.AXES, self.COLORS):
                canvas.create_text(
                    legend_x, height - 12, text=axis, fill=color, anchor="w"
                )
                legend_x += 24

        @staticmethod
        def _scale_y(
            value: float, low: float, high: float, top: int, plot_height: int
        ) -> float:
            return top + (high - value) / (high - low) * plot_height

    return ChartView


# Exposed lazily; assigned on first ImuVisualizer.run() so importing this module
# does not require a display / tkinter to be available.
ChartView: type | None = None


# ---------------------------------------------------------------------------
# Status text helper
# ---------------------------------------------------------------------------


def _status_text_from_point(point: ImuPoint, sample_count: int) -> str:
    """Build a human-readable status string from a processed IMU point."""
    payload = imu_point_payload(point)

    def fmt(values: Axis3) -> str:
        return f"X {values[0]: .3f}, Y {values[1]: .3f}, Z {values[2]: .3f}"

    return (
        f"Samples: {sample_count} | Seq: {point.sequence} | "
        f"Device ms: {point.timestamp_ms}\n"
        f"Accel {fmt(point.accel_mps2)} | Gyro {fmt(point.gyro_dps)}\n"
        f"Motion {payload['motion_intensity_mps2']:.3f} m/s^2 | "
        f"Rotation {payload['rotation_intensity_dps']:.3f} deg/s | "
        f"State stationary={payload['is_stationary']} "
        f"moving={payload['is_moving']} rotating={payload['is_rotating']} | "
        f"Quality {payload['quality']}"
    )


# ---------------------------------------------------------------------------
# ImuVisualizer — main visualization class
# ---------------------------------------------------------------------------


class ImuVisualizer:
    """Real-time Tkinter visualization for Ring IMU Kalman-filtered data.

    Creates a window with two line charts (linear acceleration and angular
    velocity) and a status bar showing live motion state. All BLE, sensor-report
    and Kalman processing are delegated to
    :class:`~ring_IMU.sensor_data_collector.SensorDataCollector` via a background
    worker thread; this class only renders the resulting point stream.

    Args:
        address: BLE MAC address of the ring device.
        auto_prepare: If True, auto-handle busy state by waiting for a click.
        packet_timeout: Timeout for waiting on IMU data packets (seconds).
        calibration_seconds: Kalman filter calibration duration.
        title: Window title.
    """

    def __init__(
        self,
        address: str,
        *,
        auto_prepare: bool = True,
        packet_timeout: float = DEFAULT_PACKET_TIMEOUT_S,
        calibration_seconds: float = DEFAULT_CALIBRATION_SECONDS,
        title: str = "Ring IMU Lines",
    ) -> None:
        self._address = address
        self._auto_prepare = auto_prepare
        self._packet_timeout = packet_timeout
        self._calibration_seconds = calibration_seconds
        self._title = title

        self._root: Any = None
        self._worker: _RingWorker | None = None
        self._sample_count: int = 0

    def run(self) -> None:
        """Run the visualizer (blocking).

        Opens the Tkinter window and enters the main loop. The window closes
        when the user clicks the close button or the connection is lost.
        """
        global ChartView
        import tkinter as tk
        from tkinter import ttk

        if ChartView is None:
            ChartView = _make_chart_view()

        root = tk.Tk()
        root.title(self._title)
        self._root = root
        try:
            ttk.Style().theme_use("clam")
        except tk.TclError:
            pass

        events: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        worker = _RingWorker(
            events,
            auto_prepare=self._auto_prepare,
            packet_timeout_s=self._packet_timeout,
            calibration_seconds=self._calibration_seconds,
        )
        self._worker = worker

        state = {
            "connected": False,
            "stream_started": False,
        }
        self._sample_count = 0

        main = ttk.Frame(root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=1)
        for row in range(2):
            main.rowconfigure(row + 1, weight=1)

        status_var = tk.StringVar(value=f"Connecting to {self._address} ...")
        stats_var = tk.StringVar(value="No samples yet.")
        ttk.Label(main, textvariable=status_var).grid(row=0, column=0, sticky="ew")
        ttk.Label(main, textvariable=stats_var).grid(
            row=3, column=0, sticky="ew", pady=(6, 0)
        )

        accel_chart = ChartView(
            main,
            title="Linear acceleration (gravity removed)",
            unit="m/s^2",
        )
        gyro_chart = ChartView(main, title="Kalman angular velocity", unit="deg/s")
        charts = (accel_chart, gyro_chart)
        for row_idx, chart in enumerate(charts, start=1):
            chart.grid(row=row_idx, column=0, sticky="nsew", pady=4)

        # --- Event handlers ---

        def start_stream_if_ready() -> None:
            if state["connected"] and not state["stream_started"]:
                state["stream_started"] = True
                worker.start_stream()

        def handle_event(kind: str, payload: Any) -> None:
            if kind == "status":
                status_var.set(str(payload))
            elif kind == "error":
                status_var.set(str(payload))
            elif kind == "connected":
                state["connected"] = bool(payload)
                status_var.set(
                    "Connected. Starting IMU stream..."
                    if payload
                    else "Disconnected."
                )
                start_stream_if_ready()
            elif kind == "sensor_started":
                status_var.set(
                    "IMU running: "
                    f"{payload['sample_rate_hz']} Hz, "
                    f"+/-{payload['accel_range_g']} g, "
                    f"+/-{payload['gyro_range_dps']} dps."
                )
            elif kind == "sensor_stopped":
                status_var.set("IMU stopped.")
                state["stream_started"] = False
            elif kind == "calibration":
                status_var.set(f"Calibrating: {float(payload) * 100:.0f}%")
            elif kind == "align":
                status_var.set(str(payload))
            elif kind == "point":
                self._sample_count += 1
                point = payload
                accel_chart.add(point.host_time_s, point.accel_mps2)
                gyro_chart.add(point.host_time_s, point.gyro_dps)
                stats_var.set(
                    _status_text_from_point(point, self._sample_count)
                )

        def process_events() -> None:
            handled = 0
            while handled < 500:
                try:
                    kind, payload = events.get_nowait()
                except queue.Empty:
                    break
                handled += 1
                handle_event(kind, payload)
            root.after(50, process_events)

        def draw_charts() -> None:
            for chart in charts:
                chart.draw()
            root.after(100, draw_charts)

        def on_close() -> None:
            print("[Visualizer] Window closed by user.")
            try:
                worker.close()
            finally:
                root.destroy()

        # --- Start ---
        root.protocol("WM_DELETE_WINDOW", on_close)
        worker.connect(self._address)
        root.after(50, process_events)
        root.after(100, draw_charts)

        print(f"[Visualizer] Starting GUI for {self._address}")
        root.mainloop()
        print(
            f"[Visualizer] GUI closed. Total samples displayed: {self._sample_count}"
        )


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ring IMU real-time Kalman visualization (standalone)."
    )
    parser.add_argument(
        "--address", required=True, help="BLE MAC address of the ring."
    )
    parser.add_argument(
        "--packet-timeout",
        type=float,
        default=DEFAULT_PACKET_TIMEOUT_S,
        help="Timeout for IMU data packets (seconds).",
    )
    parser.add_argument(
        "--calibration-seconds",
        type=float,
        default=DEFAULT_CALIBRATION_SECONDS,
        help="Kalman filter calibration duration.",
    )
    parser.add_argument(
        "--no-auto-prepare",
        action="store_true",
        help="Disable auto-prepare (do not wait for click on busy state).",
    )

    args = parser.parse_args()

    viz = ImuVisualizer(
        address=args.address,
        auto_prepare=not args.no_auto_prepare,
        packet_timeout=args.packet_timeout,
        calibration_seconds=args.calibration_seconds,
    )
    viz.run()
