"""手势/按键事件识别模块：监听戒指的手势、双击、按键等事件。

将 Ring Sound SDK 的事件监听能力封装为独立模块，
支持并发监听多种事件类型，不依赖 IMU 数据采集功能。

用法::

    from ring_IMU.gesture_recognizer import GestureRecognizer

    recognizer = GestureRecognizer(ring_address="AA:BB:CC:DD:EE:FF")
    event = await recognizer.wait_any_event(timeout_s=30.0)
    print(event)
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator

from utils.sdk_loader import get_sdk
from utils.ble_connection import (
    DEFAULT_CONNECT_ATTEMPTS,
    DEFAULT_CONNECT_RETRY_DELAY,
    connect_ring,
    disconnect_ring,
)


# 支持的事件类型
EVENT_KINDS = ["key_single_press", "key_double_press", "double_tap", "gesture"]


class GestureRecognizer:
    """手势/按键事件识别器。"""

    def __init__(
        self,
        ring_address: str | None = None,
        *,
        connect_attempts: int = DEFAULT_CONNECT_ATTEMPTS,
        retry_delay: float = DEFAULT_CONNECT_RETRY_DELAY,
    ) -> None:
        """初始化事件识别器。

        Parameters
        ----------
        ring_address : str | None
            戒指 BLE MAC 地址，格式如 "AA:BB:CC:DD:EE:FF"。
        connect_attempts : int
            最大连接重试次数。
        retry_delay : float
            重试间隔秒数。
        """
        self.ring_address = ring_address
        self._connect_attempts = connect_attempts
        self._retry_delay = retry_delay
        self._ring: Any = None

    # ------------------------------------------------------------------
    # BLE 连接管理
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """连接戒指设备。

        带重试机制，适应戒指低频广播场景。
        """
        if self._ring is not None:
            print("[GestureRecognizer] 已连接，跳过重复连接")
            return

        self._ring = await connect_ring(
            self.ring_address,
            connect_attempts=self._connect_attempts,
            retry_delay=self._retry_delay,
            log_prefix="[GestureRecognizer]",
        )

    async def disconnect(self) -> None:
        """断开连接。"""
        if self._ring is None:
            return
        await disconnect_ring(self._ring, log_prefix="[GestureRecognizer]")
        self._ring = None

    # ------------------------------------------------------------------
    # 内部辅助：等待单一事件类型
    # ------------------------------------------------------------------

    async def _wait_one_event(
        self, kind: str, timeout_s: float | None
    ) -> dict[str, Any]:
        """等待指定类型的单个事件并返回标准化 dict。"""
        sdk = get_sdk()
        ring = self._ring
        timestamp = int(time.time() * 1000)

        if kind == "key_single_press":
            event = await sdk.wait_sensor_key_single_press_event(
                ring, timeout_s=timeout_s
            )
            return {
                "type": "key_single_press",
                "timestamp": timestamp,
                "details": event.__dict__,
            }
        elif kind == "key_double_press":
            event = await sdk.wait_sensor_key_double_press_event(
                ring, timeout_s=timeout_s
            )
            return {
                "type": "key_double_press",
                "timestamp": timestamp,
                "details": event.__dict__,
            }
        elif kind == "double_tap":
            event = await sdk.wait_sensor_double_tap_event(
                ring, timeout_s=timeout_s
            )
            return {
                "type": "double_tap",
                "timestamp": timestamp,
                "details": event.__dict__,
            }
        elif kind == "gesture":
            event = await sdk.wait_sensor_gesture_event(
                ring, timeout_s=timeout_s
            )
            details = event.__dict__.copy()
            details["gesture_name"] = sdk.sensor_gesture_name(event.gesture_id)
            return {
                "type": "gesture",
                "timestamp": timestamp,
                "details": details,
            }
        else:
            raise ValueError(f"[GestureRecognizer] 未知事件类型: {kind}")

    # ------------------------------------------------------------------
    # 公开 API：等待事件
    # ------------------------------------------------------------------

    async def wait_gesture(self, timeout_s: float | None = None) -> dict[str, Any]:
        """等待手势事件（旋转/挥手等）。

        Parameters
        ----------
        timeout_s : float | None
            超时秒数，None 表示无限等待。

        Returns
        -------
        dict[str, Any]
            标准化事件 dict，包含 type、timestamp、details。
        """
        if self._ring is None:
            raise RuntimeError("[GestureRecognizer] 未连接")

        print("[GestureRecognizer] 等待手势事件 ...")
        result = await self._wait_one_event("gesture", timeout_s)
        print(
            f"[GestureRecognizer] 检测到手势: "
            f"{result['details'].get('gesture_name', 'unknown')}"
        )
        return result

    async def wait_double_tap(self, timeout_s: float | None = None) -> dict[str, Any]:
        """等待双击事件。

        Parameters
        ----------
        timeout_s : float | None
            超时秒数，None 表示无限等待。

        Returns
        -------
        dict[str, Any]
            标准化事件 dict，包含 type、timestamp、details。
        """
        if self._ring is None:
            raise RuntimeError("[GestureRecognizer] 未连接")

        print("[GestureRecognizer] 等待双击事件 ...")
        result = await self._wait_one_event("double_tap", timeout_s)
        print("[GestureRecognizer] 检测到双击")
        return result

    async def wait_any_event(self, timeout_s: float | None = None) -> dict[str, Any]:
        """等待任意事件（并发监听多种事件类型）。

        使用 asyncio.wait 并发监听所有支持的事件类型，
        返回最先触发的事件。

        Parameters
        ----------
        timeout_s : float | None
            超时秒数，None 表示无限等待。

        Returns
        -------
        dict[str, Any]
            标准化事件 dict，包含 type、timestamp、details。

        Raises
        ------
        TimeoutError
            超时未收到任何事件。
        """
        if self._ring is None:
            raise RuntimeError("[GestureRecognizer] 未连接")

        print("[GestureRecognizer] 等待任意事件 ...")

        tasks = {
            asyncio.create_task(
                self._wait_one_event(kind, timeout_s)
            ): kind
            for kind in EVENT_KINDS
        }

        try:
            done, pending = await asyncio.wait(
                tasks.keys(),
                timeout=timeout_s,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if not done:
                raise TimeoutError(
                    f"[GestureRecognizer] 等待事件超时 ({timeout_s}s)"
                )

            # 取第一个完成的任务结果
            task = next(iter(done))
            result = task.result()
            print(f"[GestureRecognizer] 检测到事件: {result['type']}")
            return result

        finally:
            # 取消所有未完成的任务
            for task in tasks:
                if not task.done():
                    task.cancel()

    async def event_stream(
        self, timeout_s: float | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """异步生成器，连续产生事件。

        Parameters
        ----------
        timeout_s : float | None
            总监听时长（秒）。为 None 时无限监听。

        Yields
        ------
        dict[str, Any]
            标准化事件 dict。
        """
        if self._ring is None:
            raise RuntimeError("[GestureRecognizer] 未连接")

        deadline = (time.monotonic() + timeout_s) if timeout_s else None
        event_count = 0

        print(
            f"[GestureRecognizer] 开始事件流"
            + (f" (timeout={timeout_s}s)" if timeout_s else " (无限)")
        )

        tasks: dict[asyncio.Task, str] = {}
        try:
            # 初始化：为每种事件类型创建监听任务
            remaining = (
                max(0.0, deadline - time.monotonic()) if deadline else None
            )
            for kind in EVENT_KINDS:
                tasks[
                    asyncio.create_task(self._wait_one_event(kind, remaining))
                ] = kind

            while tasks:
                if deadline and time.monotonic() >= deadline:
                    break

                wait_timeout = (
                    max(0.0, deadline - time.monotonic()) if deadline else None
                )
                done, _pending = await asyncio.wait(
                    tasks.keys(),
                    timeout=wait_timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if not done:
                    break

                for task in done:
                    kind = tasks.pop(task)
                    try:
                        result = task.result()
                        event_count += 1
                        yield result
                    except Exception as exc:
                        print(
                            f"[GestureRecognizer] 事件监听异常 "
                            f"({kind}): {exc!r}"
                        )

                    # 重新创建该类型的监听任务
                    if deadline is None or time.monotonic() < deadline:
                        remaining = (
                            max(0.1, deadline - time.monotonic())
                            if deadline
                            else None
                        )
                        tasks[
                            asyncio.create_task(
                                self._wait_one_event(kind, remaining)
                            )
                        ] = kind
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            print(
                f"[GestureRecognizer] 事件流结束，共 {event_count} 个事件"
            )

    # ------------------------------------------------------------------
    # 上下文管理器
    # ------------------------------------------------------------------

    async def __aenter__(self) -> GestureRecognizer:
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.disconnect()


__all__ = ["GestureRecognizer"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="监听戒指手势/按键事件")
    parser.add_argument("address", help="戒指 BLE MAC 地址 (AA:BB:CC:DD:EE:FF)")
    parser.add_argument(
        "-t", "--timeout", type=float, default=30.0,
        help="监听时长（秒），默认 30",
    )
    args = parser.parse_args()

    async def _main() -> None:
        recognizer = GestureRecognizer(ring_address=args.address)
        await recognizer.connect()
        try:
            async for event in recognizer.event_stream(timeout_s=args.timeout):
                print(f"  事件: {event}")
        finally:
            await recognizer.disconnect()

    asyncio.run(_main())
