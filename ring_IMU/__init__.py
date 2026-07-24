"""ring_IMU — 语音戒指 IMU 传感器功能模块。

提供 BLE IMU 数据采集、手势/按键事件识别、Kalman 滤波处理与实时可视化能力，
设计风格与 ring_talking 模块保持一致。

用法::

    from ring_IMU import SensorDataCollector, GestureRecognizer, KalmanProcessor, ImuVisualizer
"""

from ring_IMU.sensor_data_collector import SensorDataCollector
from ring_IMU.gesture_recognizer import GestureRecognizer
from ring_IMU.kalman_filter import KalmanProcessor
from ring_IMU.visualizer import ImuVisualizer
from ring_IMU.action_trigger import (
    ActionTrigger,
    ActionRule,
    ChannelCondition,
    TriggerEvent,
)

__all__ = [
    "SensorDataCollector",
    "GestureRecognizer",
    "KalmanProcessor",
    "ImuVisualizer",
    "ActionTrigger",
    "ActionRule",
    "ChannelCondition",
    "TriggerEvent",
]
