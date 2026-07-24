# AdventureX Smart Ring

AdventureX 语音戒指开发套件——基于 BLE 通信的多模态智能戒指，集成音频录制/解码、IMU 传感器数据采集、卡尔曼滤波、手势识别与语音转文字能力。

## 项目结构

```
AdventureX-smart-ring/
├── ring_sound_SDK/    # 底层 SDK（BLE 通信、协议收发、Speex 解码、IMU/动作事件）
├── ring_talking/      # 音频处理（录音下载解码、语音转文字）
├── ring_IMU/          # IMU 传感器（数据采集、卡尔曼滤波、手势识别、实时可视化）
├── utils/             # 共享工具（SDK 加载、ffmpeg 路径）
├── test_mic_to_text.py  # 麦克风录音→语音转文字 测试脚本
└── requirements.txt
```

| 模块 | 职责 |
|------|------|
| `ring_sound_SDK/ring_sound.py` | 单文件 Python SDK：BLE 扫描连接、v4 协议收发、录音下载、Speex→WAV 解码、IMU 数据、动作事件、CLI |
| `ring_talking/audio_decoder.py` | 封装 SDK 的 BLE 录音下载 + Speex→WAV 解码 |
| `ring_talking/speech_to_text.py` | DashScope Qwen3-ASR-Flash 语音转文字 |
| `ring_IMU/sensor_data_collector.py` | BLE IMU 数据流式采集（带重连） |
| `ring_IMU/gesture_recognizer.py` | 手势/按键事件并发监听 |
| `ring_IMU/kalman_filter.py` | 卡尔曼滤波处理管线 + 运动状态分类 + CSV 记录 |
| `ring_IMU/visualizer.py` | Tkinter 实时三轴折线图可视化 |
| `utils/sdk_loader.py` | 统一加载 `ring_sound.py` 到 `sys.path` |
| `utils/ffmpeg_tools.py` | 获取 ffmpeg 路径（优先 imageio-ffmpeg） |

## 安装依赖

```bash
pip install -r requirements.txt
```

| 依赖 | 用途 |
|------|------|
| `bleak>=0.21.0` | BLE 蓝牙通信（扫描、连接戒指） |
| `dashscope>=1.20.0` | 阿里云百炼 API（语音转文字） |
| `imageio-ffmpeg>=0.4.8` | 内置 ffmpeg 二进制（Speex→WAV 解码） |
| `sounddevice>=0.4.6` | 麦克风录音 |
| `soundfile>=0.12.1` | WAV 文件读写 |
| `numpy>=1.24.0` | 音频数据处理 |

## 环境配置

语音转文字功能需要设置阿里云百炼 API Key：

```powershell
# Windows
set DASHSCOPE_API_KEY=your_api_key

# Linux/macOS
export DASHSCOPE_API_KEY=your_api_key
```

## 快速开始

### ring_talking：录音下载 + 语音转文字

```python
import asyncio
from ring_talking import AudioDecoder, SpeechToText

async def main():
    # 从戒指下载最新录音并解码为 WAV
    decoder = AudioDecoder(ring_address="AA:BB:CC:DD:EE:FF")
    wav_path = await decoder.download_and_decode()

    # 语音转文字
    stt = SpeechToText(language="zh")
    text = await stt.transcribe(wav_path)
    print(f"识别结果: {text}")

asyncio.run(main())
```

### ring_IMU：数据采集 + 手势监听

```python
import asyncio
from ring_IMU import SensorDataCollector, GestureRecognizer

async def collect_imu():
    """采集 10 秒 IMU 数据"""
    async with SensorDataCollector(ring_address="AA:BB:CC:DD:EE:FF") as collector:
        async for batch in collector.read_stream(duration_s=10.0):
            for sample in batch:
                print(f"seq={sample['sequence']} accel=({sample['accel_x']},{sample['accel_y']},{sample['accel_z']})")

async def listen_gestures():
    """监听手势事件"""
    async with GestureRecognizer(ring_address="AA:BB:CC:DD:EE:FF") as recognizer:
        async for event in recognizer.event_stream(timeout_s=30.0):
            print(f"事件: {event['type']} -> {event['details']}")

asyncio.run(collect_imu())
```

### ring_IMU：卡尔曼滤波处理

```python
from pathlib import Path
from ring_IMU.kalman_filter import KalmanProcessor

processor = KalmanProcessor(
    visualizer_path=Path("path/to/imu_visualizer.py"),
    sdk_dir=Path("ring_sound_SDK"),
)
processor.start(sample_rate_hz=50, accel_range_g=2, gyro_range_dps=250)

# 对每个 IMU 采样执行卡尔曼滤波
result = processor.process(sample, sequence=seq_num)
# result 包含去重力加速度、滤波角速度、运动状态分类等
```

## ring_IMU 模块详细说明

### 核心类

| 类 | 说明 |
|----|------|
| `SensorDataCollector` | BLE IMU 数据采集器，支持 async context manager，带自动重连 |
| `GestureRecognizer` | 并发监听 4 种事件：`key_single_press`、`key_double_press`、`double_tap`、`gesture` |
| `KalmanProcessor` | 卡尔曼滤波管线，动态加载外部 `imu_visualizer.py` 中的 `MotionIntegrator` |
| `ImuVisualizer` | Tkinter 实时可视化，显示加速度和角速度三轴折线图 + 运动状态 |

### 数据输出格式

#### CSV 输出（24 列）

`HandoffImuCsvLogger` 产生的 CSV 包含以下字段：

| 列 | 说明 |
|----|------|
| `host_time_s` | 主机时间戳（秒） |
| `sequence` | 序列号 |
| `timestamp_ms` | 设备时间戳（ms） |
| `kalman_accel_x/y/z_mps2` | 卡尔曼滤波后加速度（去重力，m/s²） |
| `kalman_gyro_x/y/z_dps` | 卡尔曼滤波后角速度（°/s） |
| `raw_accel_x/y/z` | 原始加速度计读数 |
| `raw_gyro_x/y/z` | 原始陀螺仪读数 |
| `motion_intensity_mps2` | 加速度合量（m/s²） |
| `rotation_intensity_dps` | 角速度合量（°/s） |
| `is_stationary` | 是否静止 |
| `is_moving` | 是否运动 |
| `is_rotating` | 是否旋转 |
| `dominant_motion_axis` | 主运动轴（x/y/z/none） |
| `dominant_rotation_axis` | 主旋转轴（x/y/z/none） |
| `quality` | 数据质量标记 |
| `sample_index_in_run` | 本次采集内样本序号 |

#### JSONL 事件格式

参见 `ring_IMU/sample_ring_events.jsonl`，每行一个 JSON 对象：

```jsonl
{"type":"device-state","connected":true,"address":"EB:CD:C4:47:F4:8C","battery_percent":84,...}
{"type":"raw-imu","timestamp":...,"ax":120,"ay":-30,"az":980,"gx":5,"gy":2,"gz":-1}
{"type":"imu-state","timestamp":...,"motion_intensity":0.73,"is_stationary":false,...}
{"type":"gesture","gesture":"rotate_front","timestamp":...}
{"type":"voice-status","status":"recognizing","timestamp":...}
{"type":"voice-text","text":"识别文本","timestamp":...}
```

### 卡尔曼滤波阈值参数

```python
MOTION_THRESHOLDS = {
    "stationary_accel_max_mps2": 0.75,      # 静止判定：加速度 < 0.75 m/s²
    "stationary_gyro_max_dps": 8.0,         # 静止判定：角速度 < 8°/s
    "moving_accel_min_mps2": 1.2,           # 运动判定：加速度 ≥ 1.2 m/s²
    "rotating_gyro_min_dps": 20.0,          # 旋转判定：角速度 ≥ 20°/s
    "quality_accel_unstable_mps2": 12.0,    # 加速度异常标记阈值
    "quality_gyro_unstable_dps": 250.0,     # 角速度异常标记阈值
}
```

### 可视化用法

```python
from pathlib import Path
from ring_IMU.visualizer import ImuVisualizer

viz = ImuVisualizer(
    address="AA:BB:CC:DD:EE:FF",
    visualizer_path=Path("path/to/imu_visualizer.py"),
    sdk_dir=Path("ring_sound_SDK"),
)
viz.run()  # 阻塞，打开 Tkinter 窗口显示实时数据
```

或通过命令行：

```bash
python -m ring_IMU.visualizer --address AA:BB:CC:DD:EE:FF
```

## 硬件注意事项

### 设备模式互斥

戒指有两种工作模式，**不能同时使用**：

| 模式 | 功能 | 切换方式 |
|------|------|----------|
| 录音模式（默认） | 长按录音，录音数据通过 BLE 传输 | 单击按键切换 |
| 手势/IMU 模式 | IMU 数据采集、HMM 手势识别 | 单击按键切换 |

- 录音模式下无法获取 IMU 数据
- 手势模式下无法录音
- 单击切换不一定成功（设备忙碌时），`0x0704` 事件不代表切换成功

### BLE 连接注意事项

- **Windows 扫描慢**：戒指低频广播，Windows 下扫描可能需要较长时间，SDK 默认扫描超时 25 秒
- **MAC 地址可能变化**：部分场景下设备地址会变更，连接失败时确认地址
- **避免多客户端同连**：同时只能有一个 BLE 客户端连接戒指
- **连接重试**：`SensorDataCollector` / `GestureRecognizer` 内置最多 10 次重连

### 设备信息

| 项目 | 值 |
|------|-----|
| BLE 广播名 | `ring` |
| NUS Service UUID | `6E400001-B5A3-F393-E0A9-E50E24DCCA9E` |
| TX Characteristic | `6E400003-B5A3-F393-E0A9-E50E24DCCA9E`（设备→主机） |
| RX Characteristic | `6E400002-B5A3-F393-E0A9-E50E24DCCA9E`（主机→设备） |
| 音频采样率 | 16000 Hz, 16bit, 单声道 |
| 录音编码 | Speex Wideband, quality 3 |

## ring_talking 模块说明

### AudioDecoder

BLE 录音下载与 Speex→WAV 解码器：

```python
from ring_talking import AudioDecoder

decoder = AudioDecoder(ring_address="AA:BB:CC:DD:EE:FF")
wav_path = await decoder.download_and_decode()          # 下载最新录音
wav_path = await decoder.download_and_decode(file_index=0)  # 指定录音编号
```

流程：连接戒指 → 查询录音数量 → 下载 Speex 数据 → 通过 ffmpeg 解码为 WAV → 断开连接。

### SpeechToText

基于阿里云百炼 DashScope 的 Qwen3-ASR-Flash 语音转文字：

```python
from ring_talking import SpeechToText

stt = SpeechToText(language="zh")          # 支持 zh/en 等
text = await stt.transcribe("recording.wav")
```

- 模型：`qwen3-asr-flash`（在线 API，无需本地模型）
- 鉴权：环境变量 `DASHSCOPE_API_KEY`
- 输入：16kHz 16bit 单声道 WAV

## 测试脚本

`test_mic_to_text.py` 通过电脑麦克风录音并调用语音转文字：

```bash
# 默认录音 5 秒
python test_mic_to_text.py

# 录音 10 秒
python test_mic_to_text.py -d 10

# 手动模式（按 Enter 停止录音）
python test_mic_to_text.py --manual

# 指定语言
python test_mic_to_text.py -l en
```

录音文件和转写结果保存在 `ring_talking/data/` 目录中。

## License

[MIT](LICENSE)
