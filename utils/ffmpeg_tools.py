"""获取 ffmpeg 可执行文件路径。

ring_sound SDK 在把 Speex 录音解码成 WAV 时需要调用 ffmpeg。为了避免依赖系统
安装或手动配置，这里优先使用 `imageio-ffmpeg` 附带的静态 ffmpeg 二进制；若不
可用则回退到系统 PATH 中的 `ffmpeg`。
"""

from __future__ import annotations

import shutil


def get_ffmpeg_path() -> str:
    """返回可用的 ffmpeg 可执行文件路径。

    优先返回 imageio-ffmpeg 内置二进制；找不到时回退到系统 PATH 中的 ffmpeg。
    """
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        # imageio-ffmpeg 未安装或获取失败，回退到系统 ffmpeg。
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg:
            return system_ffmpeg
        # 都没有时返回默认名，交由上层 SDK 抛出更明确的解码不可用异常。
        return "ffmpeg"


__all__ = ["get_ffmpeg_path"]
