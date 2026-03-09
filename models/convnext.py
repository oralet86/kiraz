"""ConvNeXt V1 and V2 classification model builders.

Supported V1 (torchvision):  convnext-tiny, convnext-small, convnext-base
Supported V2 (timm/FCMAE):   convnextv2-atto, convnextv2-femto, convnextv2-pico,
                              convnextv2-nano, convnextv2-tiny, convnextv2-base
"""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as tv_models
import timm

CONVNEXT_V1_MODELS = {"convnext-tiny", "convnext-small", "convnext-base"}

CONVNEXT_V2_MODELS = {
    "convnextv2-atto",
    "convnextv2-femto",
    "convnextv2-pico",
    "convnextv2-nano",
    "convnextv2-tiny",
    "convnextv2-base",
}

CONVNEXT_MODELS = CONVNEXT_V1_MODELS | CONVNEXT_V2_MODELS

# Use in22k→in1k fine-tuned weights where available (better transfer learning),
# fall back to in1k for smaller variants without in22k checkpoints.
_V2_TIMM_NAMES: dict[str, str] = {
    "convnextv2-atto": "convnextv2_atto.fcmae_ft_in1k",
    "convnextv2-femto": "convnextv2_femto.fcmae_ft_in1k",
    "convnextv2-pico": "convnextv2_pico.fcmae_ft_in1k",
    "convnextv2-nano": "convnextv2_nano.fcmae_ft_in22k_in1k",
    "convnextv2-tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "convnextv2-base": "convnextv2_base.fcmae_ft_in22k_in1k",
}


def build_convnext_model(model_name: str, num_classes: int) -> nn.Module:
    """Build a pretrained ConvNeXt V1 or V2 model with a replaced classifier head.

    Args:
        model_name: One of the names in CONVNEXT_MODELS.
        num_classes: Number of output classes.

    Returns:
        An nn.Module ready for fine-tuning.
    """
    name = model_name.lower()

    if name == "convnext-tiny":
        m = tv_models.convnext_tiny(weights="DEFAULT")
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    elif name == "convnext-small":
        m = tv_models.convnext_small(weights="DEFAULT")
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    elif name == "convnext-base":
        m = tv_models.convnext_base(weights="DEFAULT")
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    elif name in _V2_TIMM_NAMES:
        timm_name = _V2_TIMM_NAMES[name]
        m = timm.create_model(timm_name, pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(
            f"Unknown ConvNeXt model: '{model_name}'. "
            f"Valid names: {sorted(CONVNEXT_MODELS)}"
        )
    return m
