"""MobileNet V2/V3 classification model builders via timm.

Supported (in1k, various improved training recipes):
    mobilenet-v2, mobilenet-v3-small, mobilenet-v3-large
"""

from __future__ import annotations

import torch.nn as nn
import timm

MOBILENET_MODELS = {"mobilenet-v2", "mobilenet-v3-small", "mobilenet-v3-large"}

_TIMM_NAMES: dict[str, str] = {
    "mobilenet-v2": "mobilenetv2_100.ra_in1k",
    "mobilenet-v3-small": "mobilenetv3_small_100.lamb_in1k",
    "mobilenet-v3-large": "mobilenetv3_large_100.ra_in1k",
}


def build_mobilenet_model(model_name: str, num_classes: int) -> nn.Module:
    """Build a pretrained MobileNet model with a replaced classifier head.

    Args:
        model_name: One of the names in MOBILENET_MODELS.
        num_classes: Number of output classes.

    Returns:
        An nn.Module ready for fine-tuning.

    Raises:
        ValueError: If model_name is not recognised.
    """
    name = model_name.lower()
    if name not in _TIMM_NAMES:
        raise ValueError(
            f"Unknown MobileNet model: '{model_name}'. "
            f"Valid names: {sorted(MOBILENET_MODELS)}"
        )
    return timm.create_model(
        _TIMM_NAMES[name], pretrained=True, num_classes=num_classes
    )
