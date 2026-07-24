"""IMU 阈值动作触发器：按命名规则对六轴通道做边沿触发判定。

这是一个**纯检测器**模块：它不拥有 BLE 连接、不涉及 GUI，只接收调用方喂入的
:class:`ring_IMU.kalman_filter.ImuPoint` 或 :func:`imu_point_payload` 产出的
dict，按“多命名规则 + 六通道有符号阈值 + 边沿触发 + 冷却窗口”判定动作是否触发，
并通过 **Python 回调** 与 **异步事件流** 两种接口把触发事件发出去。

六个可设阈值的通道均为**有符号**量（保留方向，不取绝对值）：

    accel_x/y/z -> kalman_accel_{x,y,z}_mps2 / point.accel_mps2[i]   (m/s^2)
    gyro_x/y/z  -> kalman_gyro_{x,y,z}_dps  / point.gyro_dps[i]      (deg/s)

方向由阈值正负 + 比较方向表达，例如 ``gyro_z >= +20`` 与 ``gyro_z <= -20`` 表示
左右两个方向的不同动作。

用法::

    from ring_IMU.action_trigger import ActionTrigger, ActionRule, DEFAULT_RULES

    def on_trigger(ev):
        print(ev.action, ev.channel_values)

    trigger = ActionTrigger(DEFAULT_RULES, on_trigger=on_trigger)
    for sample in batch.samples:
        point = integrator.process(sample, sequence=seq)
        trigger.process(point)   # 返回本样本新触发的事件列表

    # 或异步消费事件流：
    async for ev in trigger.stream():
        print(ev.as_dict())
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, AsyncIterator, Callable, Iterable

__all__ = [
    "CHANNELS",
    "COMPARATORS",
    "DEFAULT_COOLDOWN_S",
    "ChannelCondition",
    "ActionRule",
    "TriggerEvent",
    "ActionTrigger",
    "DEFAULT_RULES",
    "load_rules_from_json",
]


# 可用于设阈值的六个有符号通道。
CHANNELS = ("accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z")
# 仅支持这两种比较方向；阈值正负决定方向。
COMPARATORS = (">=", "<=")
# 边沿触发后的默认冷却时间（秒），期间同一规则不重复触发。
DEFAULT_COOLDOWN_S = 0.3


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelCondition:
    """单个通道的有符号阈值条件。

    Parameters
    ----------
    channel : str
        六个通道之一，见 :data:`CHANNELS`。
    comparator : str
        ``">="`` 或 ``"<="``，见 :data:`COMPARATORS`。
    threshold : float
        有符号阈值。方向由阈值正负 + 比较方向共同表达。
    """

    channel: str
    comparator: str
    threshold: float

    def __post_init__(self) -> None:
        if self.channel not in CHANNELS:
            raise ValueError(
                f"[ActionTrigger] 未知通道 {self.channel!r}，可用: {CHANNELS}"
            )
        if self.comparator not in COMPARATORS:
            raise ValueError(
                f"[ActionTrigger] 不支持的比较方向 {self.comparator!r}，"
                f"仅支持 {COMPARATORS}"
            )

    def evaluate(self, values: dict[str, float]) -> bool:
        """判断当前六通道快照是否满足该条件。"""
        value = values[self.channel]
        if self.comparator == ">=":
            return value >= self.threshold
        return value <= self.threshold

    def as_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "comparator": self.comparator,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class ActionRule:
    """一条命名动作规则。

    规则命中 = 其所有 ``conditions`` 同时成立（AND）。当前常见场景每条规则
    只放 1 个条件，但 ``conditions`` 用 tuple 承载，为后续“多通道复合叠加动作”
    预留扩展且不破坏接口。
    """

    name: str
    conditions: tuple[ChannelCondition, ...]
    cooldown_s: float = DEFAULT_COOLDOWN_S

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("[ActionTrigger] 规则 name 不能为空")
        if not self.conditions:
            raise ValueError(
                f"[ActionTrigger] 规则 {self.name!r} 至少需要一个 condition"
            )

    @classmethod
    def single(
        cls,
        name: str,
        channel: str,
        comparator: str,
        threshold: float,
        *,
        cooldown_s: float = DEFAULT_COOLDOWN_S,
    ) -> "ActionRule":
        """便捷构造单通道条件规则。"""
        return cls(
            name=name,
            conditions=(ChannelCondition(channel, comparator, threshold),),
            cooldown_s=cooldown_s,
        )

    def matches(self, values: dict[str, float]) -> bool:
        """所有条件同时成立时命中。"""
        return all(cond.evaluate(values) for cond in self.conditions)


@dataclass(frozen=True)
class TriggerEvent:
    """一次动作触发事件。"""

    action: str
    triggered: bool
    timestamp_ms: int
    host_time_s: float
    sequence: int
    channel_values: dict[str, float]
    matched: tuple[ChannelCondition, ...]

    def as_dict(self) -> dict[str, Any]:
        """转为 JSON 可序列化 dict，供后续 WebSocket/文件等接口直接复用。"""
        return {
            "action": self.action,
            "triggered": self.triggered,
            "timestamp_ms": self.timestamp_ms,
            "host_time_s": self.host_time_s,
            "sequence": self.sequence,
            "channel_values": dict(self.channel_values),
            "matched": [cond.as_dict() for cond in self.matched],
        }


# ---------------------------------------------------------------------------
# 通道取值归一化
# ---------------------------------------------------------------------------


def _channel_values(point_or_payload: Any) -> dict[str, float]:
    """从 ImuPoint 或 imu_point_payload dict 提取六通道有符号值。"""
    # payload dict 分支：读 kalman_* 键。
    if isinstance(point_or_payload, dict):
        payload = point_or_payload
        return {
            "accel_x": float(payload["kalman_accel_x_mps2"]),
            "accel_y": float(payload["kalman_accel_y_mps2"]),
            "accel_z": float(payload["kalman_accel_z_mps2"]),
            "gyro_x": float(payload["kalman_gyro_x_dps"]),
            "gyro_y": float(payload["kalman_gyro_y_dps"]),
            "gyro_z": float(payload["kalman_gyro_z_dps"]),
        }
    # ImuPoint 分支：读 accel_mps2 / gyro_dps 三元组。
    point = point_or_payload
    accel = point.accel_mps2
    gyro = point.gyro_dps
    return {
        "accel_x": float(accel[0]),
        "accel_y": float(accel[1]),
        "accel_z": float(accel[2]),
        "gyro_x": float(gyro[0]),
        "gyro_y": float(gyro[1]),
        "gyro_z": float(gyro[2]),
    }


def _meta(point_or_payload: Any) -> tuple[int, float, int]:
    """提取 (timestamp_ms, host_time_s, sequence)，缺失字段用合理回退。"""
    if isinstance(point_or_payload, dict):
        payload = point_or_payload
        timestamp_ms = int(payload.get("timestamp_ms", 0))
        host_time_s = float(payload.get("host_time_s", time.monotonic()))
        sequence = int(payload.get("sequence", -1))
    else:
        point = point_or_payload
        timestamp_ms = int(getattr(point, "timestamp_ms", 0))
        host_time_s = float(getattr(point, "host_time_s", time.monotonic()))
        sequence = int(getattr(point, "sequence", -1))
    return timestamp_ms, host_time_s, sequence


# ---------------------------------------------------------------------------
# ActionTrigger — 检测器与输出接口
# ---------------------------------------------------------------------------


class ActionTrigger:
    """按命名规则做边沿触发 + 冷却判定的纯检测器。

    Parameters
    ----------
    rules : Iterable[ActionRule]
        初始规则集合。
    on_trigger : Callable[[TriggerEvent], None] | None
        触发时的同步回调（可选）。
    stream_maxsize : int
        内部异步事件流队列的最大长度；满时丢弃最旧事件以避免阻塞采集线程。
    """

    def __init__(
        self,
        rules: Iterable[ActionRule] = (),
        *,
        on_trigger: Callable[[TriggerEvent], None] | None = None,
        stream_maxsize: int = 256,
    ) -> None:
        self._rules: dict[str, ActionRule] = {}
        self._on_trigger = on_trigger
        self._stream_maxsize = max(1, stream_maxsize)

        # 边沿 / 冷却状态
        self._armed: dict[str, bool] = {}
        self._last_fire_s: dict[str, float] = {}

        # 异步事件流：首次进入 stream() 时惰性创建。
        self._queue: "asyncio.Queue[TriggerEvent]" | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        for rule in rules:
            self.add_rule(rule)

    # -- 规则管理 -----------------------------------------------------------

    def add_rule(self, rule: ActionRule) -> None:
        """新增/覆盖一条规则。"""
        self._rules[rule.name] = rule
        self._armed.pop(rule.name, None)
        self._last_fire_s.pop(rule.name, None)

    def remove_rule(self, name: str) -> None:
        """按名称移除规则（不存在则忽略）。"""
        self._rules.pop(name, None)
        self._armed.pop(name, None)
        self._last_fire_s.pop(name, None)

    @property
    def rules(self) -> tuple[ActionRule, ...]:
        return tuple(self._rules.values())

    def reset(self) -> None:
        """清空边沿与冷却状态（规则保留）。"""
        self._armed.clear()
        self._last_fire_s.clear()

    # -- 处理 ---------------------------------------------------------------

    def process(self, point_or_payload: Any) -> list[TriggerEvent]:
        """处理一个样本，返回本样本**新触发**的事件列表。

        逐规则做边沿 + 冷却判定；对新触发的规则构造 :class:`TriggerEvent`，
        依次调用 ``on_trigger``、推入异步事件流队列、追加到返回列表。
        同步返回，可在任意线程调用。
        """
        values = _channel_values(point_or_payload)
        timestamp_ms, host_time_s, sequence = _meta(point_or_payload)

        fired: list[TriggerEvent] = []
        for name, rule in self._rules.items():
            satisfied = rule.matches(values)
            was_armed = self._armed.get(name, False)

            if satisfied and not was_armed:
                # 不满足 -> 满足 的上升沿，检查冷却窗口。
                last_fire = self._last_fire_s.get(name)
                if last_fire is None or (host_time_s - last_fire) >= rule.cooldown_s:
                    event = TriggerEvent(
                        action=name,
                        triggered=True,
                        timestamp_ms=timestamp_ms,
                        host_time_s=host_time_s,
                        sequence=sequence,
                        channel_values=dict(values),
                        matched=rule.conditions,
                    )
                    self._last_fire_s[name] = host_time_s
                    fired.append(event)

            # 更新武装状态：条件跌回不满足时重新武装，供下一次跨越触发。
            self._armed[name] = satisfied

        for event in fired:
            self._emit(event)
        return fired

    def _emit(self, event: TriggerEvent) -> None:
        """派发事件到回调与异步事件流队列。"""
        if self._on_trigger is not None:
            try:
                self._on_trigger(event)
            except Exception as exc:  # 回调异常不应影响采集
                print(f"[ActionTrigger] on_trigger 回调异常: {exc!r}")

        queue = self._queue
        if queue is None:
            return
        loop = self._loop
        if loop is not None and loop.is_running():
            # 从任意线程安全入队（兼容 BLE 后台线程喂数据）。
            loop.call_soon_threadsafe(self._enqueue_nowait, queue, event)
        else:
            self._enqueue_nowait(queue, event)

    @staticmethod
    def _enqueue_nowait(
        queue: "asyncio.Queue[TriggerEvent]", event: TriggerEvent
    ) -> None:
        """非阻塞入队；队列满时丢弃最旧事件后再入队。"""
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    # -- 异步事件流 ---------------------------------------------------------

    async def stream(self) -> AsyncIterator[TriggerEvent]:
        """异步事件流：``async for ev in trigger.stream(): ...``。

        首次进入时捕获当前运行的 event loop 与队列；:meth:`process` 会经
        ``loop.call_soon_threadsafe`` 把事件安全推入，以兼容“BLE 后台线程喂
        数据、异步协程消费”的场景。
        """
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self._stream_maxsize)
        self._loop = asyncio.get_running_loop()
        queue = self._queue
        while True:
            event = await queue.get()
            yield event


# ---------------------------------------------------------------------------
# 默认示例规则 & JSON 配置加载
# ---------------------------------------------------------------------------


# 仅作默认演示，调用方可完全替换。方向由阈值正负 + 比较方向表达。
DEFAULT_RULES: tuple[ActionRule, ...] = (
    ActionRule.single("push_forward", "accel_x", ">=", 8.0),
    ActionRule.single("pull_back", "accel_x", "<=", -8.0),
    ActionRule.single("turn_right", "gyro_z", ">=", 20.0),
    ActionRule.single("turn_left", "gyro_z", "<=", -20.0),
)


def load_rules_from_json(path: Any) -> list[ActionRule]:
    """从 JSON 文件加载规则列表。

    期望结构（字段与 :class:`ActionRule` / :class:`ChannelCondition` 对应）::

        [
          {
            "name": "push_forward",
            "cooldown_s": 0.3,
            "conditions": [
              {"channel": "accel_x", "comparator": ">=", "threshold": 8.0}
            ]
          },
          ...
        ]

    也支持单条件简写（省略 ``conditions``，直接给 channel/comparator/threshold）。
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("[ActionTrigger] 规则 JSON 顶层必须是数组")

    rules: list[ActionRule] = []
    for entry in data:
        name = entry["name"]
        cooldown_s = float(entry.get("cooldown_s", DEFAULT_COOLDOWN_S))
        if "conditions" in entry:
            conditions = tuple(
                ChannelCondition(
                    channel=cond["channel"],
                    comparator=cond["comparator"],
                    threshold=float(cond["threshold"]),
                )
                for cond in entry["conditions"]
            )
            rules.append(
                ActionRule(name=name, conditions=conditions, cooldown_s=cooldown_s)
            )
        else:
            # 单条件简写。
            rules.append(
                ActionRule.single(
                    name,
                    entry["channel"],
                    entry["comparator"],
                    float(entry["threshold"]),
                    cooldown_s=cooldown_s,
                )
            )
    return rules


# ---------------------------------------------------------------------------
# 轻量自测入口（不连 BLE）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio as _asyncio
    from types import SimpleNamespace

    print("[ActionTrigger] 自测开始")

    # 合成六通道样本：0.02s 间隔（50 Hz），构造跨越阈值与保持超阈两种情形。
    def make_point(seq: int, host_s: float, accel_x: float, gyro_z: float) -> Any:
        return SimpleNamespace(
            host_time_s=host_s,
            sequence=seq,
            timestamp_ms=seq * 20,
            accel_mps2=(accel_x, 0.0, 0.0),
            gyro_dps=(0.0, 0.0, gyro_z),
        )

    # accel_x 序列：低 -> 超阈保持3样本 -> 跌回 -> 再超阈。
    # 期望：边沿只触发一次；保持超阈期间不重复；跌回后再超阈再触发一次。
    script = [
        (0.00, 0.0, 0.0),
        (0.02, 9.0, 0.0),   # 上升沿 -> push_forward 触发 #1
        (0.04, 9.5, 0.0),   # 保持超阈 -> 不重复
        (0.06, 9.2, 0.0),   # 保持超阈 -> 不重复
        (0.08, 0.0, 0.0),   # 跌回 -> 重新武装
        (0.60, 8.5, 0.0),   # 再次上升沿（已过冷却）-> push_forward 触发 #2
        (0.62, 0.0, 25.0),  # gyro_z 上升沿 -> turn_right 触发
        (0.64, 0.0, -30.0), # gyro_z 反向上升沿 -> turn_left 触发
    ]

    collected: list[TriggerEvent] = []
    trigger = ActionTrigger(DEFAULT_RULES, on_trigger=collected.append)

    for index, (host_s, accel_x, gyro_z) in enumerate(script):
        point = make_point(index, host_s, accel_x, gyro_z)
        trigger.process(point)

    actions = [ev.action for ev in collected]
    print(f"  回调收到动作序列: {actions}")
    assert actions == [
        "push_forward",
        "push_forward",
        "turn_right",
        "turn_left",
    ], f"边沿/冷却语义不符: {actions}"

    # 验证 as_dict() 可 JSON 序列化。
    sample_json = json.dumps(collected[0].as_dict())
    print(f"  首个事件 JSON: {sample_json}")

    # 验证冷却窗口：紧接上升沿再来一次上升沿（未过冷却）不应触发。
    trig2 = ActionTrigger([ActionRule.single("tap", "accel_x", ">=", 5.0, cooldown_s=0.3)])
    fired_a = trig2.process(make_point(0, 0.00, 6.0, 0.0))   # 触发
    _idle = trig2.process(make_point(1, 0.02, 0.0, 0.0))     # 跌回
    fired_b = trig2.process(make_point(2, 0.10, 6.0, 0.0))   # 上升沿但在冷却内 -> 不触发
    _idle2 = trig2.process(make_point(3, 0.12, 0.0, 0.0))    # 跌回
    fired_c = trig2.process(make_point(4, 0.50, 6.0, 0.0))   # 过冷却 -> 触发
    assert len(fired_a) == 1 and len(fired_b) == 0 and len(fired_c) == 1, (
        f"冷却窗口不符: {len(fired_a)}, {len(fired_b)}, {len(fired_c)}"
    )
    print("  冷却窗口验证通过")

    # 验证异步事件流。
    async def _stream_test() -> list[str]:
        trig3 = ActionTrigger(DEFAULT_RULES)
        received: list[str] = []

        async def consumer() -> None:
            async for ev in trig3.stream():
                received.append(ev.action)
                if len(received) >= 2:
                    break

        task = _asyncio.create_task(consumer())
        await _asyncio.sleep(0.05)  # 让 consumer 进入 stream() 捕获 loop/queue
        trig3.process(make_point(0, 0.0, 9.0, 0.0))    # push_forward
        trig3.process(make_point(1, 0.02, 0.0, 0.0))
        trig3.process(make_point(2, 0.6, 0.0, 25.0))   # turn_right
        await _asyncio.wait_for(task, timeout=2.0)
        return received

    stream_actions = _asyncio.run(_stream_test())
    print(f"  异步流收到动作序列: {stream_actions}")
    assert stream_actions == ["push_forward", "turn_right"], (
        f"异步流不符: {stream_actions}"
    )

    print("[ActionTrigger] 自测全部通过 ✓")
