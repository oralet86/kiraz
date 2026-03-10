"""Faster R-CNN detection model builders.

Supported Faster R-CNN (torchvision):
    faster-rcnn-r50, faster-rcnn-r101
"""

from __future__ import annotations

import torch.nn as nn
import torchvision.models.detection as tv_detect
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

FRCNN_MODELS = {"faster-rcnn-r50", "faster-rcnn-r101"}


def build_frcnn_model(model_name: str, num_classes: int) -> nn.Module:
    """Build a pretrained Faster R-CNN with a replaced prediction head.

    Args:
        model_name: One of the names in FRCNN_MODELS.
        num_classes: Number of output classes (including background at index 0).

    Returns:
        An nn.Module ready for fine-tuning.

    Raises:
        ValueError: If model_name is not recognised.
    """
    name = model_name.lower()
    if name == "faster-rcnn-r50":
        m = tv_detect.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
        in_f = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes)
    elif name == "faster-rcnn-r101":
        backbone = resnet_fpn_backbone(
            "resnet101", weights="DEFAULT", trainable_layers=3
        )
        m = FasterRCNN(backbone, num_classes=num_classes)
    else:
        raise ValueError(
            f"Unknown Faster R-CNN model: '{model_name}'. "
            f"Valid names: {sorted(FRCNN_MODELS)}"
        )
    return m
