"""EfficientNet classification model builders via timm.

Supported (in1k, various improved training recipes):
    efficientnet-b0, efficientnet-b1, efficientnet-b2, efficientnet-b3
"""

from __future__ import annotations

import torch.nn as nn
import timm

EFFICIENTNET_MODELS = {
    "efficientnet-b0",
    "efficientnet-b1",
    "efficientnet-b2",
    "efficientnet-b3",
}

_TIMM_NAMES: dict[str, str] = {
    "efficientnet-b0": "efficientnet_b0.ra_in1k",
    "efficientnet-b1": "efficientnet_b1.ft_in1k",
    "efficientnet-b2": "efficientnet_b2.ra_in1k",
    "efficientnet-b3": "efficientnet_b3.ra2_in1k",
}


def build_efficientnet_model(model_name: str, num_classes: int) -> nn.Module:
    """Build a pretrained EfficientNet model with a replaced classifier head.

    Args:
        model_name: One of the names in EFFICIENTNET_MODELS.
        num_classes: Number of output classes.

    Returns:
        An nn.Module ready for fine-tuning.

    Raises:
        ValueError: If model_name is not recognised.
    """
    name = model_name.lower()
    if name not in _TIMM_NAMES:
        raise ValueError(
            f"Unknown EfficientNet model: '{model_name}'. "
            f"Valid names: {sorted(EFFICIENTNET_MODELS)}"
        )
    return timm.create_model(
        _TIMM_NAMES[name], pretrained=True, num_classes=num_classes
    )
