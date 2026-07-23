"""定位并导入 ring_sound SDK。

`ring_sound.py` 是位于 `ring_sound_SDK/` 目录下的单文件 SDK。为了让项目里任意
位置的脚本都能以 `import ring_sound as sdk` 的方式使用它，这里统一负责把 SDK
目录加入 `sys.path` 并缓存导入结果。

用法::

    from utils import get_sdk

    sdk = get_sdk()
    info = await sdk.get_system_info(ring)
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

# 项目根目录：utils/ 的上一级。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# SDK 所在目录与模块名。
SDK_DIR = PROJECT_ROOT / "ring_sound_SDK"
SDK_MODULE_NAME = "ring_sound"

_sdk_cache: ModuleType | None = None


def get_sdk() -> ModuleType:
    """导入并返回 ring_sound SDK 模块（结果会被缓存）。"""
    global _sdk_cache
    if _sdk_cache is not None:
        return _sdk_cache

    if not (SDK_DIR / f"{SDK_MODULE_NAME}.py").exists():
        raise FileNotFoundError(
            f"未找到 SDK 文件：{SDK_DIR / (SDK_MODULE_NAME + '.py')}。"
            "请确认 ring_sound_SDK/ring_sound.py 存在。"
        )

    if str(SDK_DIR) not in sys.path:
        sys.path.insert(0, str(SDK_DIR))

    _sdk_cache = importlib.import_module(SDK_MODULE_NAME)
    return _sdk_cache


__all__ = ["get_sdk", "PROJECT_ROOT", "SDK_DIR", "SDK_MODULE_NAME"]
