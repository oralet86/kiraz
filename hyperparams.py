"""
Hyperparameters management module for YOLO training.

This module provides a centralized interface for managing hyperparameters
for both classification and detection tasks, with support for HPO-optimized
parameters for specific models.
"""

from typing import Any, Dict
import os
import warnings

# Global training constants
EPOCHS_DETECT = 100
EPOCHS_CLS = 100
IMGSZ_DETECT = 640
IMGSZ_CLS = 320
BATCH_DETECT = 24
BATCH_CLS = 120
CACHE_DETECT = False
CACHE_CLS = False
PATIENCE_DETECT = 10
PATIENCE_CLS = 10
DETERMINISTIC = False
COS_LR = True
WORKERS = min(max(1, int((os.cpu_count() or 1) * 0.5)), 8)
AMP = True


def get_default_cls_hyperparams() -> Dict[str, Any]:
    """Get default hyperparameters for classification tasks."""
    return {
        "lr0": 1e-5,
        "lrf": 0.001,
        "optimizer": "AdamW",
        "dropout": 0.5,
        # Disable YOLO augmentations when using pre-augmented data
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        # Disable hardcoded Albumentations transforms (Blur, MedianBlur, ToGray, CLAHE)
        "augmentations": [],
    }


def get_default_detect_hyperparams() -> Dict[str, Any]:
    """Get default hyperparameters for detection tasks."""
    return {
        "lr0": 4e-5,
        "lrf": 0.001,
        "box": 15.0,
        "iou": 0.5,
        "cls": 1.5,
        "dfl": 2.0,
        "dropout": 0.5,
        "optimizer": "AdamW",
        # Disable YOLO augmentations when using pre-augmented data
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "close_mosaic": 0,
        # Disable hardcoded Albumentations transforms (Blur, MedianBlur, ToGray, CLAHE)
        "augmentations": [],
    }


# HPO-optimized hyperparameters database
HPO_DATABASE = {}


def get_training_config(task: str) -> Dict[str, Any]:
    """
    Get training configuration constants for a specific task.

    Args:
        task: Either "detect" or "cls"

    Returns:
        Dictionary containing training configuration parameters
    """
    task = task.lower()
    if task == "detect":
        return {
            "epochs": EPOCHS_DETECT,
            "imgsz": IMGSZ_DETECT,
            "batch": BATCH_DETECT,
            "cache": CACHE_DETECT,
            "workers": WORKERS,
            "amp": AMP,
            "patience": PATIENCE_DETECT,
            "deterministic": DETERMINISTIC,
            "cos_lr": COS_LR,
        }
    elif task == "cls":
        return {
            "epochs": EPOCHS_CLS,
            "imgsz": IMGSZ_CLS,
            "batch": BATCH_CLS,
            "cache": CACHE_CLS,
            "workers": WORKERS,
            "amp": AMP,
            "patience": PATIENCE_CLS,
            "deterministic": DETERMINISTIC,
            "cos_lr": COS_LR,
        }
    else:
        raise ValueError(f"Unknown task: {task}. Must be 'detect' or 'cls'")


def get_hyperparams(model_name: str, hpo: bool = False) -> Dict[str, Any]:
    """
    Get hyperparameters for a specific model.

    Args:
        model_name: Name of the model (e.g., 'yolo26l-cls', 'yolov10m', 'yolo11s')
        hpo: Whether to use HPO-optimized parameters if available (default: False)

    Returns:
        Dictionary of hyperparameters for the specified model.

    Examples:
        >>> params = get_hyperparams('yolo26l-cls', hpo=True)
        >>> params = get_hyperparams('yolo11s')  # defaults to hpo=False
        >>> params = get_hyperparams('yolov10m', hpo=True)
    """
    model_name_lower = model_name.lower()

    # Determine task type based on model name
    task = "cls" if "cls" in model_name_lower else "detect"

    # Get base training configuration
    training_config = get_training_config(task)

    # Check if HPO is requested and model is in database
    if hpo:
        if model_name_lower in HPO_DATABASE:
            hpo_params = HPO_DATABASE[model_name_lower].copy()
            # Merge training config with HPO params (HPO params take precedence)
            training_config.update(hpo_params)
            return training_config
        else:
            warnings.warn(
                f"HPO requested for '{model_name}' but no HPO data found. Falling back to defaults.",
                UserWarning,
                stacklevel=2,
            )

    # Return training config with default hyperparameters
    if task == "cls":
        training_config.update(get_default_cls_hyperparams())
    else:
        training_config.update(get_default_detect_hyperparams())

    return training_config


# Non-YOLO (PyTorch-native) training constants

EPOCHS_TORCH = 100
IMGSZ_CLS_TORCH = 320
IMGSZ_DETECT_TORCH = 640
BATCH_CLS_TORCH = 40
BATCH_DETECT_TORCH = 8
PATIENCE_TORCH = 10
LR_CLS_TORCH = 1e-4
LR_DETECT_TORCH = 2e-5
WEIGHT_DECAY_TORCH = 1e-4
AMP_TORCH = True

# Per-model hyperparameter overrides for PyTorch-native classification models.
# convnext-tiny works well with defaults; larger ConvNeXt models overfit, so we
# apply stronger regularisation (higher weight_decay, lower lr, label smoothing).
_TORCH_MODEL_OVERRIDES: Dict[str, Dict[str, Any]] = {
    # ConvNeXt V1
    "convnext-tiny": {},
    "convnext-small": {
        "lr": 5e-5,
        "weight_decay": 5e-4,
        "label_smoothing": 0.1,
    },
    "convnext-base": {
        "lr": 5e-5,
        "weight_decay": 5e-4,
        "label_smoothing": 0.1,
    },
    # ConvNeXt V2 — tiny models match tiny V1 capacity, so defaults are fine
    "convnextv2-atto": {},
    "convnextv2-femto": {},
    "convnextv2-pico": {},
    "convnextv2-nano": {},
    "convnextv2-tiny": {},
    "convnextv2-small": {
        "lr": 5e-5,
        "weight_decay": 5e-4,
        "label_smoothing": 0.1,
    },
    "convnextv2-base": {
        "lr": 5e-5,
        "weight_decay": 5e-4,
        "label_smoothing": 0.1,
    },
    # ViT — vit-small outperforming vit-base in prior runs confirms overfitting
    # even at the small scale, so all ViT variants get explicit regularisation.
    "vit-small": {
        "lr": 5e-5,
        "weight_decay": 3e-4,
        "label_smoothing": 0.1,
        "imgsz": 224,
    },
    "vit-base": {
        "lr": 2e-5,
        "weight_decay": 5e-4,
        "label_smoothing": 0.1,
        "imgsz": 224,
    },
    # DeiT — same tiered policy as ViT equivalents
    "deit-small": {
        "lr": 5e-5,
        "weight_decay": 3e-4,
        "label_smoothing": 0.1,
        "imgsz": 224,
    },
    "deit-base": {
        "lr": 2e-5,
        "weight_decay": 5e-4,
        "label_smoothing": 0.1,
        "imgsz": 224,
    },
    # MobileViT V1 — very compact (~1–6 M params) but still attention-based;
    # mild regularisation + native 256×256 input size
    "mobilevit-xxs": {
        "lr": 7e-5,
        "weight_decay": 2e-4,
        "label_smoothing": 0.05,
        "imgsz": 256,
    },
    "mobilevit-xs": {
        "lr": 7e-5,
        "weight_decay": 2e-4,
        "label_smoothing": 0.05,
        "imgsz": 256,
    },
    "mobilevit-s": {
        "lr": 5e-5,
        "weight_decay": 3e-4,
        "label_smoothing": 0.1,
        "imgsz": 256,
    },
    # MobileViT V2 — separable attention, similar capacity tiers to V1
    "mobilevitv2-050": {
        "lr": 7e-5,
        "weight_decay": 2e-4,
        "label_smoothing": 0.05,
        "imgsz": 256,
    },
    "mobilevitv2-075": {
        "lr": 7e-5,
        "weight_decay": 2e-4,
        "label_smoothing": 0.05,
        "imgsz": 256,
    },
    "mobilevitv2-100": {
        "lr": 5e-5,
        "weight_decay": 3e-4,
        "label_smoothing": 0.1,
        "imgsz": 256,
    },
    # Swin-Tiny — ~28 M params; treat like ViT-small in terms of regularisation
    "swin-tiny": {
        "lr": 3e-5,
        "weight_decay": 5e-4,
        "label_smoothing": 0.1,
        "imgsz": 224,
    },
}


def get_torch_hyperparams(mode: str, model_name: str = "") -> Dict[str, Any]:
    """
    Get hyperparameters for non-YOLO PyTorch-native models.

    Args:
        mode: Either "cls" or "detect".
        model_name: Optional model name used to apply per-model overrides.

    Returns:
        Dictionary of hyperparameters for training.
    """
    mode = mode.lower()
    if mode == "cls":
        base: Dict[str, Any] = {
            "lr": LR_CLS_TORCH,
            "weight_decay": WEIGHT_DECAY_TORCH,
            "label_smoothing": 0.0,
            "epochs": EPOCHS_TORCH,
            "batch": BATCH_CLS_TORCH,
            "imgsz": IMGSZ_CLS_TORCH,
            "patience": PATIENCE_TORCH,
            "amp": AMP_TORCH,
            "workers": WORKERS,
        }
    elif mode == "detect":
        base = {
            "lr": LR_DETECT_TORCH,
            "weight_decay": WEIGHT_DECAY_TORCH,
            "epochs": EPOCHS_TORCH,
            "batch": BATCH_DETECT_TORCH,
            "imgsz": IMGSZ_DETECT_TORCH,
            "patience": PATIENCE_TORCH,
            "amp": AMP_TORCH,
            "workers": WORKERS,
        }
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'cls' or 'detect'")

    overrides = _TORCH_MODEL_OVERRIDES.get(model_name.lower(), {})
    base.update(overrides)
    return base


# =============================================================================
# Pipeline (inference / sorting) hyperparameters
# =============================================================================

DETECTION_CONF_THRESH: float = 0.4  # minimum detector confidence to keep a box
TRACK_BUFFER: int = 30  # frames to keep a LOST track alive
MIN_SAMPLES_FOR_DECISION: int = 3  # minimum classified samples before deciding
MAX_SAMPLES_PER_TRACK: int = 10  # cap samples per track to reduce redundancy
SAMPLE_EVERY_N_FRAMES: int = 2  # sub-sample frames to reduce correlation
CROP_BUFFER_PX: int = 15  # pixel padding around bbox when cropping
IMPERFECT_THRESHOLD: float = 0.5  # weighted avg above this → IMPERFECT
CENTER_WEIGHT_SIGMA: float = 0.3  # Gaussian σ for center-offset weighting
ARUCO_REAL_SIZE_CM: float = 3.0  # known physical size of the ArUco marker
ARUCO_CALIB_FRAMES: int = 30  # frames to collect during ArUco calibration


def get_pipeline_hyperparams() -> Dict[str, Any]:
    """Get hyperparameters for the real-time sorting pipeline."""
    return {
        "detection_conf_thresh": DETECTION_CONF_THRESH,
        "track_buffer": TRACK_BUFFER,
        "min_samples_for_decision": MIN_SAMPLES_FOR_DECISION,
        "max_samples_per_track": MAX_SAMPLES_PER_TRACK,
        "sample_every_n_frames": SAMPLE_EVERY_N_FRAMES,
        "crop_buffer_px": CROP_BUFFER_PX,
        "imperfect_threshold": IMPERFECT_THRESHOLD,
        "center_weight_sigma": CENTER_WEIGHT_SIGMA,
        "aruco_real_size_cm": ARUCO_REAL_SIZE_CM,
        "aruco_calib_frames": ARUCO_CALIB_FRAMES,
    }


def list_available_models() -> Dict[str, Dict[str, Any]]:
    """
    List all available models with their HPO support status.

    Returns:
        Dictionary mapping model names to their HPO support info.
    """
    return {
        model: {
            "has_hpo": True,
            "task": "classification" if "cls" in model else "detection",
        }
        for model in HPO_DATABASE.keys()
    }


def validate_hyperparams(params: Dict[str, Any]) -> bool:
    """
    Validate hyperparameter dictionary for common required fields.

    Args:
        params: Hyperparameter dictionary to validate

    Returns:
        True if valid, False otherwise
    """
    required_fields = ["lr0", "lrf"]
    return all(field in params for field in required_fields)
