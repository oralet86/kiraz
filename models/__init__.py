"""Model builders for classification and detection.

Classification:
    ResNet:        resnet50, resnet101
    EfficientNet:  efficientnet-b{0..3}
    MobileNet:     mobilenet-v2, mobilenet-v3-{small,large}
    ConvNeXt V1:   convnext-{tiny,small,base}
    ConvNeXt V2:   convnextv2-{atto,femto,pico,nano,tiny,base}
    ViT:           vit-{small,base}
    DeiT:          deit-{small,base}
    MobileViT V1:  mobilevit-{xxs,xs,s}
    MobileViT V2:  mobilevitv2-{050,075,100}
    Swin:          swin-tiny

Detection:
    Faster R-CNN:  faster-rcnn-r{50,101}
"""

from __future__ import annotations

from models.convnext import (
    CONVNEXT_MODELS,
    CONVNEXT_V1_MODELS,
    CONVNEXT_V2_MODELS,
    build_convnext_model,
)
from models.frcnn import (
    FRCNN_MODELS,
    build_frcnn_model,
)
from models.efficientnet import EFFICIENTNET_MODELS, build_efficientnet_model
from models.mobilenet import MOBILENET_MODELS, build_mobilenet_model
from models.resnet import RESNET_MODELS, build_resnet_model
from models.vit import (
    DEIT_MODELS,
    MOBILEVIT_V1_MODELS,
    MOBILEVIT_V2_MODELS,
    SWIN_MODELS,
    TRANSFORMER_MODELS,
    VIT_MODELS,
    build_transformer_model,
)

import torch.nn as nn

CLS_MODELS = (
    RESNET_MODELS
    | EFFICIENTNET_MODELS
    | MOBILENET_MODELS
    | CONVNEXT_MODELS
    | TRANSFORMER_MODELS
)

DETECT_MODELS = FRCNN_MODELS  # Add other detection models here


def build_cls_model(model_name: str, num_classes: int) -> nn.Module:
    """Build a pretrained classification model with a replaced head.

    Dispatches to the appropriate family builder based on model_name.

    Args:
        model_name: Any name from CLS_MODELS.
        num_classes: Number of output classes.

    Returns:
        An nn.Module ready for fine-tuning.

    Raises:
        ValueError: If model_name is not recognised.
    """
    name = model_name.lower()
    if name in RESNET_MODELS:
        return build_resnet_model(name, num_classes)
    elif name in EFFICIENTNET_MODELS:
        return build_efficientnet_model(name, num_classes)
    elif name in MOBILENET_MODELS:
        return build_mobilenet_model(name, num_classes)
    elif name in CONVNEXT_MODELS:
        return build_convnext_model(name, num_classes)
    elif name in TRANSFORMER_MODELS:
        return build_transformer_model(name, num_classes)
    else:
        raise ValueError(
            f"Unknown classification model: '{model_name}'. "
            f"Valid names: {sorted(CLS_MODELS)}"
        )


__all__ = [
    # Sets
    "CLS_MODELS",
    "CONVNEXT_MODELS",
    "CONVNEXT_V1_MODELS",
    "CONVNEXT_V2_MODELS",
    "DEIT_MODELS",
    "EFFICIENTNET_MODELS",
    "FRCNN_MODELS",
    "MOBILENET_MODELS",
    "MOBILEVIT_V1_MODELS",
    "MOBILEVIT_V2_MODELS",
    "RESNET_MODELS",
    "SWIN_MODELS",
    "TRANSFORMER_MODELS",
    "VIT_MODELS",
    # Builders
    "build_cls_model",
    "build_convnext_model",
    "build_efficientnet_model",
    "build_frcnn_model",
    "build_mobilenet_model",
    "build_resnet_model",
    "build_transformer_model",
]
