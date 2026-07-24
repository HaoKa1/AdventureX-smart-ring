"""麦克风录音 → 语音转文字 测试脚本。

通过电脑麦克风录音，调用 ring_talking 模块的 SpeechToText 进行转写，
将录音文件和转写结果保存在 ring_talking/data/ 目录中。

前提条件：设置环境变量 DASHSCOPE_API_KEY（阿里云百炼 API Key）。

用法::

    python test_mic_to_text.py              # 默认录音 5 秒
    python test_mic_to_text.py -d 10        # 录音 10 秒
    python test_mic_to_text.py --manual     # 手动模式：按 Enter 停止录音
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

import sounddevice as sd
import soundfile as sf
import numpy as np

from ring_talking.speech_to_text import SpeechToText

# 录音参数（与 SDK 音频规格匹配）
SAMPLE_RATE = 16000  # 16kHz
CHANNELS = 1         # 单声道
DTYPE = "int16"      # 16-bit


def record_fixed_duration(duration: float) -> np.ndarray:
    """录制固定时长音频。"""
    print(f"\n🎙️  正在录音（{duration} 秒）...")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
    )
    sd.wait()
    print("✅  录音结束。")
    return audio


def record_manual_stop() -> np.ndarray:
    """手动模式录音，按 Enter 停止。"""
    frames: list[np.ndarray] = []
    stop_event = threading.Event()

    def callback(indata, frame_count, time_info, status):
        if status:
            print(f"  [警告] {status}")
        frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=callback,
    )

    print("\n🎙️  正在录音... 按 Enter 停止。")
    with stream:
        input()  # 阻塞等待用户按 Enter
    print("✅  录音结束。")

    if not frames:
        return np.zeros((0, CHANNELS), dtype=DTYPE)
    return np.concatenate(frames, axis=0)


def save_wav(audio: np.ndarray, wav_path: Path) -> None:
    """保存音频为 WAV 文件。"""
    sf.write(str(wav_path), audio, SAMPLE_RATE, subtype="PCM_16")
    print(f"💾  WAV 已保存: {wav_path}")


def save_result(txt_path: Path, wav_path: Path, text: str, timestamp: str) -> None:
    """保存转写结果到文本文件。"""
    content = (
        f"时间: {timestamp}\n"
        f"音频文件: {wav_path}\n"
        f"转写文本: {text}\n"
    )
    txt_path.write_text(content, encoding="utf-8")
    print(f"📄  转写结果已保存: {txt_path}")


async def main(args: argparse.Namespace) -> None:
    # 确保 ring_talking/data/ 目录存在
    data_dir = Path(__file__).parent / "ring_talking" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # --- 录音 ---
    input("按 Enter 开始录音...")

    if args.manual:
        audio = record_manual_stop()
    else:
        audio = record_fixed_duration(args.duration)

    if len(audio) == 0:
        print("⚠️  未录到任何音频，退出。")
        return

    # --- 保存 WAV ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_filename = f"recording_{timestamp}.wav"
    wav_path = data_dir / wav_filename
    save_wav(audio, wav_path)

    # --- 语音转文字 ---
    print("\n🔄  正在进行语音转文字...")
    stt = SpeechToText(language=args.language)
    text = await stt.transcribe(wav_path)
    print(f"\n📝  转写结果: {text}")

    # --- 保存结果 ---
    txt_filename = f"recording_{timestamp}.txt"
    txt_path = data_dir / txt_filename
    readable_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_result(txt_path, wav_path, text, readable_time)

    print("\n✨  完成！")


if __name__ == "__main__":
    # 检查 API Key 是否已设置
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("⚠️  未检测到 DASHSCOPE_API_KEY 环境变量！")
        print("   请先设置阿里云百炼 API Key：")
        print("   Windows:  set DASHSCOPE_API_KEY=your_api_key")
        print("   Linux:    export DASHSCOPE_API_KEY=your_api_key")
        print()
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="麦克风录音 → 语音转文字 测试脚本（DashScope）"
    )
    parser.add_argument(
        "-d", "--duration", type=float, default=5.0,
        help="录音时长（秒），默认 5 秒。使用 --manual 时忽略此参数",
    )
    parser.add_argument(
        "--manual", action="store_true",
        help="手动模式：按 Enter 停止录音",
    )
    parser.add_argument(
        "-l", "--language", default="zh",
        help="目标语言代码，默认 zh",
    )
    parsed_args = parser.parse_args()
    asyncio.run(main(parsed_args))
