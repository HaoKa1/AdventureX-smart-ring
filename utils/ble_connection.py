"""可复用的 Ring 蓝牙(BLE/NUS)连接工具。

统一封装 Ring Sound SDK 的官方连接入口 ``sdk.connect_ring()``，在其之上补充
项目级的连接重试、安全断开与统一日志，供所有功能模块复用。本模块只负责
“连接管理”，不涉及 IMU 上报、事件监听、音频下载、可视化等具体业务，避免功能耦合。

设计要点：
- 连接实现完全委托给 SDK 官方入口 ``sdk.connect_ring()``，不再手动构造
  ``RingSoundClient`` / ``NusClient``；扫描超时等底层细节交由 SDK 决定。
- 本模块只在 SDK 连接之上叠加“重试 + 日志 + 安全断开”这层薄封装。

用法::

    from utils.ble_connection import connect_ring, disconnect_ring

    ring = await connect_ring("AA:BB:CC:DD:EE:FF")
    try:
        ...  # 使用 ring 做上报/事件/音频/可视化
    finally:
        await disconnect_ring(ring)
"""

from __future__ import annotations

import asyncio
from typing import Any

from utils.sdk_loader import get_sdk

# BLE 连接重试默认参数（适应戒指低频广播场景，Windows 下尤其需要多次重试）。
DEFAULT_CONNECT_ATTEMPTS = 10
DEFAULT_CONNECT_RETRY_DELAY = 1.0


async def connect_ring(
    address: str,
    *,
    connect_attempts: int = DEFAULT_CONNECT_ATTEMPTS,
    retry_delay: float = DEFAULT_CONNECT_RETRY_DELAY,
    auto_time_sync: bool = False,
    command_timeout_s: float | None = None,
    log_prefix: str = "[BLE]",
) -> Any:
    """连接戒指设备，带重试机制，返回已连接的 client。

    连接实现委托 SDK 官方入口 ``sdk.connect_ring()``；本函数只负责项目级的
    重试、失败日志与最终异常抛出。

    Parameters
    ----------
    address : str
        戒指 BLE MAC 地址。为空时抛出 ``RuntimeError``。
    connect_attempts : int
        最大连接尝试次数。
    retry_delay : float
        相邻两次尝试之间的等待秒数。
    auto_time_sync : bool
        是否在连接成功后自动启用时间同步（透传给 ``sdk.connect_ring``）。
    command_timeout_s : float | None
        命令超时秒数；为 None 时使用 SDK 默认值。
    log_prefix : str
        日志前缀，便于区分调用方（如 "[SensorDataCollector]"）。

    Returns
    -------
    Any
        已连接的 `RingSoundClient` 实例。

    Raises
    ------
    RuntimeError
        未指定 address，或未执行任何连接尝试。
    Exception
        所有尝试均失败时，抛出最后一次捕获的异常。
    """
    if not address:
        raise RuntimeError(f"{log_prefix} 未指定 ring_address")

    sdk = get_sdk()
    attempts = max(1, connect_attempts)
    last_error: Exception | None = None

    kwargs: dict[str, Any] = {"address": address, "auto_time_sync": auto_time_sync}
    if command_timeout_s is not None:
        kwargs["command_timeout_s"] = command_timeout_s

    for attempt in range(1, attempts + 1):
        try:
            client = await sdk.connect_ring(**kwargs)
        except Exception as exc:
            last_error = exc
            print(f"{log_prefix} 连接失败 (尝试 {attempt}/{attempts}): {exc!r}")
            if attempt < attempts:
                await asyncio.sleep(retry_delay)
            continue

        print(f"{log_prefix} 连接成功 (尝试 {attempt}/{attempts})")
        return client

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{log_prefix} 未执行任何连接尝试")


async def disconnect_ring(ring: Any, *, log_prefix: str = "[BLE]") -> None:
    """安全断开连接；对 ``None`` 或已断开的 client 均可安全调用。"""
    if ring is None:
        return
    try:
        await ring.disconnect()
    except Exception as exc:
        print(f"{log_prefix} 断开连接时出错: {exc!r}")
    else:
        print(f"{log_prefix} 已断开连接")


__all__ = [
    "DEFAULT_CONNECT_ATTEMPTS",
    "DEFAULT_CONNECT_RETRY_DELAY",
    "connect_ring",
    "disconnect_ring",
]
