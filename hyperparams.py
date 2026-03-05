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
BATCH_DETECT = 40
BATCH_CLS = 256
CACHE_DETECT = True
CACHE_CLS = False
PATIENCE_DETECT = 15
PATIENCE_CLS = 10
DETERMINISTIC = False
COS_LR = True
WORKERS = max(1, int((os.cpu_count() or 1) * 0.5))
AMP = True


def get_default_cls_hyperparams() -> Dict[str, Any]:
    """Get default hyperparameters for classification tasks."""
    return {
        "lr0": 1e-4,
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
    }


def get_default_detect_hyperparams() -> Dict[str, Any]:
    """Get default hyperparameters for detection tasks."""
    return {
        "lr0": 2e-4,
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
