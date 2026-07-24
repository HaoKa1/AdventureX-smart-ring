"""音频解码模块：连接戒指设备、下载录音、解码为 WAV。

将 Ring Sound SDK 的 BLE 音频下载与 Speex→WAV 解码能力封装为独立模块，
不依赖语音转文字功能。

用法::

    from ring_talking.audio_decoder import AudioDecoder

    decoder = AudioDecoder(ring_address="AA:BB:CC:DD:EE:FF")
    wav_path = await decoder.download_and_decode()
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from utils.sdk_loader import get_sdk
from utils.ffmpeg_tools import get_ffmpeg_path
from utils.ble_connection import connect_ring, disconnect_ring


class AudioDecoder:
    """BLE 录音下载与 Speex→WAV 解码器。"""

    def __init__(self, ring_address: str) -> None:
        """初始化解码器。

        Parameters
        ----------
        ring_address : str
            戒指 BLE MAC 地址，格式如 "AA:BB:CC:DD:EE:FF"。
        """
        self.ring_address = ring_address

    async def download_and_decode(self, file_index: int | None = None) -> Path:
        """从戒指下载录音并解码为 WAV 文件。

        Parameters
        ----------
        file_index : int | None
            要下载的录音编号（0-based）。为 None 时自动下载最新一条。

        Returns
        -------
        Path
            解码后的 WAV 文件路径。
        """
        sdk = get_sdk()
        ffmpeg = get_ffmpeg_path()

        print(f"[AudioDecoder] 正在连接戒指 {self.ring_address} ...")
        client = await connect_ring(self.ring_address, log_prefix="[AudioDecoder]")

        try:
            # 确定文件索引
            if file_index is None:
                count = await sdk.get_audio_file_count(client)
                if count <= 0:
                    raise RuntimeError("戒指中没有录音文件")
                file_index = count - 1
                print(
                    f"[AudioDecoder] 设备共 {count} 条录音，"
                    f"将下载最新一条 (index={file_index})"
                )

            # 下载原始 Speex 数据
            print(f"[AudioDecoder] 正在下载录音 index={file_index} ...")
            info, raw_data = await sdk.download_audio_file(client, file_index)
            print(
                f"[AudioDecoder] 下载完成: "
                f"size={len(raw_data)} bytes, "
                f"record_time={info.record_time}"
            )

        finally:
            await disconnect_ring(client, log_prefix="[AudioDecoder]")

        # 解码并保存为 WAV
        print("[AudioDecoder] 正在解码为 WAV ...")
        bundle = sdk.save_audio_bundle(
            file_index=file_index,
            data=raw_data,
            ffmpeg_path=ffmpeg,
        )
        print(
            f"[AudioDecoder] WAV 已保存: {bundle.play_path} "
            f"({bundle.play_size} bytes)"
        )
        return bundle.play_path


__all__ = ["AudioDecoder"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="从戒指下载录音并解码为 WAV")
    parser.add_argument("address", help="戒指 BLE MAC 地址 (AA:BB:CC:DD:EE:FF)")
    parser.add_argument(
        "-i", "--index", type=int, default=None,
        help="录音编号 (0-based)，默认下载最新一条",
    )
    args = parser.parse_args()

    decoder = AudioDecoder(ring_address=args.address)
    wav = asyncio.run(decoder.download_and_decode(file_index=args.index))
    print(f"输出文件: {wav}")
