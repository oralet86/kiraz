"""ResNet classification model builders via timm.

Supported (in21k→in1k fine-tuned, improved training recipe):
    resnet50, resnet101
"""

from __future__ import annotations

import torch.nn as nn
import timm


RESNET_MODELS = {"resnet50", "resnet101"}

_TIMM_NAMES: dict[str, str] = {
    "resnet50": "resnet50.a1_in1k",
    "resnet101": "resnet101.a1_in1k",
}


def build_resnet_model(model_name: str, num_classes: int) -> nn.Module:
    """Build a pretrained ResNet model with a replaced classifier head.

    Args:
        model_name: One of the names in RESNET_MODELS.
        num_classes: Number of output classes.

    Returns:
        An nn.Module ready for fine-tuning.

    Raises:
        ValueError: If model_name is not recognised.
    """
    name = model_name.lower()
    if name not in _TIMM_NAMES:
        raise ValueError(
            f"Unknown ResNet model: '{model_name}'. "
            f"Valid names: {sorted(RESNET_MODELS)}"
        )
    return timm.create_model(
        _TIMM_NAMES[name], pretrained=True, num_classes=num_classes
    )
