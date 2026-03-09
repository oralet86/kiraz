"""ViT-family classification model builders.

All models load pretrained weights via timm.

Supported ViT (Dosovitskiy et al., in21k→in1k fine-tuned):
    vit-small, vit-base

Supported DeiT (Touvron et al., Facebook):
    deit-small, deit-base

Supported MobileViT V1 (Mehta & Rastegari, Apple — edge/mobile optimised):
    mobilevit-xxs, mobilevit-xs, mobilevit-s

Supported MobileViT V2 (Mehta & Rastegari, Apple — edge/mobile optimised):
    mobilevitv2-050, mobilevitv2-075, mobilevitv2-100

Supported Swin Transformer (Liu et al., Microsoft — tiny variant only):
    swin-tiny
"""

from __future__ import annotations

import torch.nn as nn
import timm

VIT_MODELS = {"vit-small", "vit-base"}
DEIT_MODELS = {"deit-small", "deit-base"}

# MobileViT V1: designed for 256×256 input; compact hybrid CNN-transformer for edge
MOBILEVIT_V1_MODELS = {"mobilevit-xxs", "mobilevit-xs", "mobilevit-s"}

# MobileViT V2: separable self-attention — faster and more accurate than V1
MOBILEVIT_V2_MODELS = {"mobilevitv2-050", "mobilevitv2-075", "mobilevitv2-100"}

# Swin-Tiny: hierarchical shifted-window attention; ~28 M params, reasonable on Pi 5
SWIN_MODELS = {"swin-tiny"}

TRANSFORMER_MODELS = (
    VIT_MODELS | DEIT_MODELS | MOBILEVIT_V1_MODELS | MOBILEVIT_V2_MODELS | SWIN_MODELS
)

# Fully-qualified timm checkpoint names pinned to specific pretrained weights.
# in21k→in1k fine-tuning is used where available for higher transfer accuracy.
_TIMM_NAMES: dict[str, str] = {
    # ViT — in21k pre-training fine-tuned on in1k (AugReg recipe)
    "vit-small": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "vit-base": "vit_base_patch16_224.augreg_in21k_ft_in1k",
    # DeiT — knowledge-distilled, in1k only
    "deit-small": "deit_small_patch16_224.fb_in1k",
    "deit-base": "deit_base_patch16_224.fb_in1k",
    # MobileViT V1 — CVNets in1k weights (native input 256×256)
    "mobilevit-xxs": "mobilevit_xxs.cvnets_in1k",
    "mobilevit-xs": "mobilevit_xs.cvnets_in1k",
    "mobilevit-s": "mobilevit_s.cvnets_in1k",
    # MobileViT V2 — CVNets in1k weights (native input 256×256)
    "mobilevitv2-050": "mobilevitv2_050.cvnets_in1k",
    "mobilevitv2-075": "mobilevitv2_075.cvnets_in1k",
    "mobilevitv2-100": "mobilevitv2_100.cvnets_in1k",
    # Swin-Tiny — Microsoft in1k weights
    "swin-tiny": "swin_tiny_patch4_window7_224.ms_in1k",
}


def build_transformer_model(model_name: str, num_classes: int) -> nn.Module:
    """Build a pretrained ViT-family model with a replaced classifier head.

    The head replacement is handled by timm's ``num_classes`` parameter, so no
    manual layer surgery is required.

    Args:
        model_name: One of the names in TRANSFORMER_MODELS.
        num_classes: Number of output classes.

    Returns:
        An nn.Module ready for fine-tuning.

    Raises:
        ValueError: If model_name is not recognised.
    """
    name = model_name.lower()
    if name not in _TIMM_NAMES:
        raise ValueError(
            f"Unknown transformer model: '{model_name}'. "
            f"Valid names: {sorted(TRANSFORMER_MODELS)}"
        )
    return timm.create_model(_TIMM_NAMES[name], pretrained=True, num_classes=num_classes)
