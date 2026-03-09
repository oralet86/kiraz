"""Faster R-CNN and DETR detection model builders.

Supported Faster R-CNN (torchvision):
    faster-rcnn-r50, faster-rcnn-r101

Supported DETR (HuggingFace):
    detr-r50, detr-r101
"""

from __future__ import annotations

from typing import Tuple

import torch.nn as nn
import torchvision.models.detection as tv_detect
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from transformers import DetrForObjectDetection, DetrImageProcessor

FRCNN_MODELS = {"faster-rcnn-r50", "faster-rcnn-r101"}
DETR_MODELS = {"detr-r50", "detr-r101"}
DETECT_MODELS = FRCNN_MODELS | DETR_MODELS


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


def build_detr_model(
    model_name: str, num_labels: int
) -> Tuple[nn.Module, DetrImageProcessor]:
    """Build a pretrained DETR model with a replaced classification head.

    Args:
        model_name: One of the names in DETR_MODELS.
        num_labels: Number of object classes (excluding no-object).

    Returns:
        A tuple of (model, processor) ready for fine-tuning.

    Raises:
        ValueError: If model_name is not recognised.
    """
    hf_ids = {
        "detr-r50": "facebook/detr-resnet-50",
        "detr-r101": "facebook/detr-resnet-101",
    }
    name = model_name.lower()
    if name not in hf_ids:
        raise ValueError(
            f"Unknown DETR model: '{model_name}'. Valid names: {sorted(DETR_MODELS)}"
        )
    hf_id = hf_ids[name]
    model = DetrForObjectDetection.from_pretrained(
        hf_id, num_labels=num_labels, ignore_mismatched_sizes=True
    )
    processor = DetrImageProcessor.from_pretrained(
        hf_id, size={"shortest_edge": 480, "longest_edge": 640}
    )
    return model, processor
