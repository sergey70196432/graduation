"""
Утилиты выбора устройства (cuda / mps / cpu) для PyTorch.

Эта функция используется в нескольких training-скриптах, поэтому живёт в общем модуле.
"""

from __future__ import annotations

import torch


def pick_device() -> str:
    """
    Выбираем "лучшее доступное" устройство:
    - CUDA (NVIDIA) — если доступно
    - MPS (Apple Silicon) — если доступно (macOS)
    - иначе CPU
    """

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

