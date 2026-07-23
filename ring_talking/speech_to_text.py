"""语音转文字模块：将 WAV 音频文件转写为文本。

使用阿里云百炼（DashScope）Qwen3-ASR-Flash 多模态语音识别 API 实现语音转写，
支持本地 WAV 文件直接识别，无需下载模型。

依赖环境变量 DASHSCOPE_API_KEY 进行鉴权。

用法::

    from ring_talking.speech_to_text import SpeechToText

    stt = SpeechToText(language="zh")
    text = await stt.transcribe("recording.wav")
"""

from __future__ import annotations

import asyncio
import os
import time
from http import HTTPStatus
from pathlib import Path
from typing import Any


class SpeechToText:
    """基于阿里云百炼 DashScope 的语音转文字引擎。"""

    def __init__(
        self,
        model: str = "qwen3-asr-flash",
        language: str = "zh",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化 STT 引擎。

        Parameters
        ----------
        model : str
            DashScope 语音识别模型名，默认 qwen3-asr-flash。
        language : str
            目标语言代码，如 "zh"、"en"。
        api_key : str | None
            DashScope API Key。为 None 时从环境变量 DASHSCOPE_API_KEY 读取。
        **kwargs
            兼容旧接口的多余参数（如 model_size、device 等），会被忽略。
        """
        self.model = model
        self.language = language
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self._initialized = False

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """验证 API Key 并标记初始化完成。

        DashScope 无需预加载模型，此方法主要用于兼容旧接口并提前校验配置。
        """
        if self._initialized:
            return

        if not self._api_key:
            raise RuntimeError(
                "[SpeechToText] 未设置 DASHSCOPE_API_KEY 环境变量，"
                "请先设置: set DASHSCOPE_API_KEY=your_api_key"
            )

        # 设置 dashscope API Key
        import dashscope
        dashscope.api_key = self._api_key

        self._initialized = True
        print(
            f"[SpeechToText] DashScope 引擎就绪 "
            f"(model={self.model}, language={self.language})"
        )

    # ------------------------------------------------------------------
    # 转写
    # ------------------------------------------------------------------

    async def transcribe(self, wav_path: Path | str) -> str:
        """将 WAV 文件转写为文字。

        首次调用时会自动验证配置。

        Parameters
        ----------
        wav_path : Path | str
            WAV 文件路径（16kHz, 16bit, mono）。

        Returns
        -------
        str
            转写文本。
        """
        await self.initialize()

        wav_path = Path(wav_path)
        if not wav_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {wav_path}")

        print(f"[SpeechToText] 正在转写: {wav_path}")
        t0 = time.perf_counter()

        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(
            None,
            self._transcribe_sync,
            str(wav_path),
        )

        elapsed = time.perf_counter() - t0
        print(f"[SpeechToText] 转写完成 ({elapsed:.1f}s)")
        return text

    def _transcribe_sync(self, audio_path: str) -> str:
        """同步执行转写（在线程池中调用）。

        使用 DashScope MultiModalConversation 类的 call 方法对本地文件进行同步识别。
        """
        import dashscope
        from dashscope import MultiModalConversation

        messages = [
            {
                "role": "user",
                "content": [{"audio": audio_path}],
            }
        ]

        asr_options: dict = {"enable_itn": True}
        if self.language:
            asr_options["language"] = self.language

        response = MultiModalConversation.call(
            model=self.model,
            messages=messages,
            result_format="message",
            asr_options=asr_options,
        )

        if response.status_code != HTTPStatus.OK:
            raise RuntimeError(
                f"[SpeechToText] DashScope 识别失败: "
                f"status={response.status_code}, message={response.message}"
            )

        # 从 MultiModalConversation 响应中提取文本
        try:
            content = response.output.choices[0].message.content
            if isinstance(content, list):
                texts = [item.get("text", "") for item in content if isinstance(item, dict)]
                return "".join(texts).strip()
            elif isinstance(content, str):
                return content.strip()
            return ""
        except (IndexError, AttributeError, KeyError):
            return ""


__all__ = ["SpeechToText"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="将 WAV 文件转写为文字（DashScope）")
    parser.add_argument("wav", help="WAV 文件路径")
    parser.add_argument(
        "-l", "--language", default="zh",
        help="目标语言代码，默认 zh",
    )
    parser.add_argument(
        "--model", default="qwen3-asr-flash",
        help="DashScope 模型名，默认 qwen3-asr-flash",
    )
    args = parser.parse_args()

    stt = SpeechToText(
        model=args.model,
        language=args.language,
    )
    result = asyncio.run(stt.transcribe(args.wav))
    print(f"转写结果: {result}")
