from __future__ import annotations

"""
Модель классификатора значения скорости.

Сейчас используется MobileNetV3-Small, потому что:
- она лёгкая и быстрая
- достаточно точная для задачи "распознать число на знаке"

Важно про вход:
- в PyTorch обучении мы работаем с NCHW (batch, channels, height, width)
- в мобильном приложении чаще всего удобнее работать с HWC (height, width, channels)
  и подавать данные "плоским" массивом float.

Класс NhwcExportWrapper нужен только для экспорта (если нужно),
чтобы PyTorch-модель принимала NHWC и внутри делала permute -> NCHW.
"""

import torch
from torch import nn
from torchvision import models


def create_model(num_classes: int, *, pretrained: bool = False) -> nn.Module:
    """
    Создаёт модель MobileNetV3-Small под нужное число классов.

    Параметры:
    - num_classes: количество классов (папок) в датасете
    - pretrained: использовать ли предобученные веса ImageNet

    Возвращает:
    - nn.Module, которая выдаёт logits размера [N, num_classes]

    Важно:
    - на обучении вход = NCHW float после ToTensor()+Normalize в dataset.py
    """
    if num_classes <= 1:
        raise ValueError(f"num_classes must be > 1, got {num_classes}")

    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.mobilenet_v3_small(weights=weights)

    # Replace classifier: Sequential([Linear(...), Hardswish, Dropout, Linear(...)]), last is Linear
    if not isinstance(m.classifier, nn.Sequential) or len(m.classifier) < 1:
        raise RuntimeError("Unexpected MobileNetV3 classifier structure.")

    last = m.classifier[-1]
    if not isinstance(last, nn.Linear):
        raise RuntimeError("Unexpected MobileNetV3 last classifier layer.")

    in_features = int(last.in_features)
    m.classifier[-1] = nn.Linear(in_features, num_classes, bias=True)
    return m


class NhwcExportWrapper(nn.Module):
    """
    Обёртка для экспорта: принимает NHWC и внутри делает permute -> NCHW.

    Это бывает полезно, если вы хотите экспортировать ONNX/TFLite так,
    чтобы вход совпадал с тем, как вы подаёте данные в мобильном приложении.

    Вход:
    - x_nhwc: float32 tensor [N, H, W, 3]

    Выход:
    - logits [N, num_classes]
    """

    def __init__(self, nchw_model: nn.Module):
        super().__init__()
        self.model = nchw_model

    def forward(self, x_nhwc: torch.Tensor) -> torch.Tensor:
        """
        Перекладываем оси: NHWC -> NCHW и прогоняем через базовую модель.
        """
        x = x_nhwc.permute(0, 3, 1, 2).contiguous()
        return self.model(x)

