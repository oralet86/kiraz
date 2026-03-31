#!/usr/bin/env python3
"""Download and export models to ONNX format.

Supports:
- YOLO models (v8, v9, v10, v11, v12, v26) for detection and classification
- DETR (HuggingFace transformers)
- Faster R-CNN (torchvision)
- Classification models (timm/torchvision)

Usage:
    python convert_onnx.py --mode cls --model yolov8n-cls
    python convert_onnx.py --mode detect --model yolov8n
    python convert_onnx.py --mode cls --model all
    python convert_onnx.py --mode detect --model all
    python convert_onnx.py --mode all --model all
    python convert_onnx.py --mode all --model yolov8n
    python convert_onnx.py --pt path/to/best.pt                               # YOLO detect (auto-inferred)
    python convert_onnx.py --pt path/to/best.pt --mode cls --model convnext-tiny  # timm cls model
"""

from __future__ import annotations

import argparse
import gc
import time
import warnings
from pathlib import Path
from typing import List, Set
import torch
from ultralytics import YOLO
from models import (
    CLS_MODELS,
    FRCNN_MODELS,
    build_cls_model,
    build_frcnn_model,
)
from log import logger

# Suppress PyTorch ONNX export deprecation warning for cleaner output
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="You are using the legacy TorchScript-based ONNX export",
)


# Constants
ONNX_DIR = Path("onnx_models")
DETECT_MODELS: Set[str] = {
    # YOLO detection models
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "yolov9t",
    "yolov9s",
    "yolov9m",
    "yolov10n",
    "yolov10s",
    "yolov10m",
    "yolo11n",
    "yolo11s",
    "yolo11m",
    "yolo12n",
    "yolo12s",
    "yolo12m",
    "yolo26n",
    "yolo26s",
    "yolo26m",
    # Non-YOLO detection models
    "detr-r50",
    "faster-rcnn-r50",
}

CLS_MODELS_EXTENDED: Set[str] = {
    # YOLO classification models
    "yolov8n-cls",
    "yolov8s-cls",
    "yolov8m-cls",
    "yolov8l-cls",
    "yolo11n-cls",
    "yolo11s-cls",
    "yolo11m-cls",
    "yolo11l-cls",
    "yolo26n-cls",
    "yolo26s-cls",
    "yolo26m-cls",
    "yolo26l-cls",
    # Classification models from models module
    "convnext-tiny",
    "convnext-small",
    "convnext-base",
    "convnextv2-atto",
    "convnextv2-femto",
    "convnextv2-pico",
    "convnextv2-nano",
    "convnextv2-tiny",
    "convnextv2-base",
    "deit-small",
    "deit-base",
    "efficientnet-b0",
    "efficientnet-b1",
    "efficientnet-b2",
    "efficientnet-b3",
    "resnet50",
    "mobilenet-v2",
    "mobilenet-v3-large",
    "mobilenet-v3-small",
    "vit-small",
    "vit-base",
    "mobilevit-xxs",
    "mobilevit-xs",
    "mobilevit-s",
    "mobilevitv2-050",
    "mobilevitv2-075",
    "mobilevitv2-100",
    "swin-tiny",
}

# Image sizes for different model families
YOLO_DETECT_IMGZ = 640
YOLO_CLS_IMGZ = 224
TIMM_IMGZ = 224
DETR_IMGZ = 800
FRCNN_IMGZ = 800


def export_yolo_detection(model_name: str) -> Path:
    """Export YOLO detection model to ONNX."""
    logger.info(f"Exporting YOLO detection model: {model_name}")

    # Load YOLO model
    model = YOLO(f"{model_name}.pt")

    # Export to ONNX
    onnx_path = ONNX_DIR / f"{model_name}.onnx"
    model.export(
        format="onnx",
        imgsz=YOLO_DETECT_IMGZ,
        opset=12,
        simplify=True,
        half=False,
    )

    # Move exported file to our directory
    exported_path = Path(f"{model_name}.onnx")
    if exported_path.exists():
        exported_path.rename(onnx_path)

    logger.info(f"Exported {model_name} to {onnx_path}")
    return onnx_path


def export_yolo_classification(model_name: str) -> Path:
    """Export YOLO classification model to ONNX."""
    logger.info(f"Exporting YOLO classification model: {model_name}")

    # Load YOLO classification model
    model = YOLO(f"{model_name}.pt")

    # Export to ONNX
    onnx_path = ONNX_DIR / f"{model_name}.onnx"
    model.export(
        format="onnx",
        imgsz=YOLO_CLS_IMGZ,
        opset=12,
        simplify=True,
        half=False,
    )

    # Move exported file to our directory
    exported_path = Path(f"{model_name}.onnx")
    if exported_path.exists():
        exported_path.rename(onnx_path)

    logger.info(f"Exported {model_name} to {onnx_path}")
    return onnx_path


def export_detr(model_name: str) -> Path:
    """Export DETR model to ONNX."""
    logger.info(f"Exporting DETR model: {model_name}")

    try:
        from transformers import DetrForObjectDetection
    except ImportError as e:
        logger.error(f"Failed to import transformers: {e}")
        raise

    # Load model
    if model_name == "detr-r50":
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    else:
        raise ValueError(f"Unknown DETR model: {model_name}")

    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, DETR_IMGZ, DETR_IMGZ)

    # Export to ONNX
    onnx_path = ONNX_DIR / f"{model_name}.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,  # Use opset 18 for PyTorch 2.10 compatibility
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits", "pred_boxes"],
        dynamo=False,  # Disable dynamo to avoid dynamic_axes warnings
    )

    logger.info(f"Exported {model_name} to {onnx_path}")
    return onnx_path


def export_faster_rcnn(model_name: str) -> Path:
    """Export Faster R-CNN model to ONNX."""
    logger.info(f"Exporting Faster R-CNN model: {model_name}")

    # Build model with 3 classes (bg, cherry, stem)
    model = build_frcnn_model(model_name, num_classes=3)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, FRCNN_IMGZ, FRCNN_IMGZ)

    # Export to ONNX
    onnx_path = ONNX_DIR / f"{model_name}.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,  # Use opset 18 for PyTorch 2.10 compatibility
        do_constant_folding=True,
        input_names=["input"],
        output_names=["boxes", "labels", "scores"],
        dynamo=False,  # Disable dynamo to avoid dynamic_axes warnings
    )

    logger.info(f"Exported {model_name} to {onnx_path}")
    return onnx_path


def export_classification_model(model_name: str) -> Path:
    """Export classification model (timm/torchvision) to ONNX."""
    logger.info(f"Exporting classification model: {model_name}")

    # Build model with 2 classes (cherry, cherry-imperfect)
    model = build_cls_model(model_name, num_classes=2)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, TIMM_IMGZ, TIMM_IMGZ)

    # Export to ONNX
    onnx_path = ONNX_DIR / f"{model_name}.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,  # Use opset 18 for PyTorch 2.10 compatibility
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamo=False,  # Disable dynamo to avoid dynamic_axes warnings
    )

    logger.info(f"Exported {model_name} to {onnx_path}")
    return onnx_path


def export_yolo_detect_from_pt(pt_path: Path) -> Path:
    """Export a YOLO detection model from a .pt file path to ONNX."""
    logger.info(f"Exporting YOLO detection model from file: {pt_path}")

    model = YOLO(str(pt_path))
    onnx_path = ONNX_DIR / f"{pt_path.stem}.onnx"
    model.export(
        format="onnx",
        imgsz=YOLO_DETECT_IMGZ,
        opset=12,
        simplify=True,
        half=False,
    )

    # YOLO exports next to the .pt file; move it to our directory
    exported_path = pt_path.with_suffix(".onnx")
    if not exported_path.exists():
        exported_path = Path(pt_path.stem + ".onnx")
    if exported_path.exists() and exported_path != onnx_path:
        exported_path.rename(onnx_path)

    logger.info(f"Exported {pt_path.name} to {onnx_path}")
    return onnx_path


def export_cls_from_pt(pt_path: Path, model_name: str, num_classes: int = 2) -> Path:
    """Export a timm/torchvision classification model from a .pt file path to ONNX."""
    logger.info(f"Exporting classification model '{model_name}' from file: {pt_path}")

    model = build_cls_model(model_name, num_classes=num_classes)
    state_dict = torch.load(pt_path, map_location="cpu")
    # Unwrap common checkpoint wrappers
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, TIMM_IMGZ, TIMM_IMGZ)
    onnx_path = ONNX_DIR / f"{pt_path.stem}.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamo=False,
    )

    logger.info(f"Exported {pt_path.name} to {onnx_path}")
    return onnx_path


def export_model(model_name: str, mode: str) -> Path:
    """Export a single model to ONNX format."""
    model_name = model_name.lower()

    if mode == "detect":
        if model_name.startswith(
            ("yolov8", "yolov9", "yolov10", "yolo11", "yolo12", "yolo26")
        ):
            return export_yolo_detection(model_name)
        elif model_name == "detr-r50":
            return export_detr(model_name)
        elif model_name in FRCNN_MODELS:
            return export_faster_rcnn(model_name)
        else:
            raise ValueError(f"Unknown detection model: {model_name}")

    elif mode == "cls":
        if (
            model_name.startswith(
                ("yolov8", "yolov9", "yolo10", "yolo11", "yolo12", "yolo26")
            )
            and "-cls" in model_name
        ):
            return export_yolo_classification(model_name)
        elif model_name in CLS_MODELS:
            return export_classification_model(model_name)
        else:
            raise ValueError(f"Unknown classification model: {model_name}")

    else:
        raise ValueError(f"Unknown mode: {mode}")


def export_all_models(mode: str) -> List[Path]:
    """Export all models for a given mode."""
    models = DETECT_MODELS if mode == "detect" else CLS_MODELS_EXTENDED
    exported_paths = []

    logger.info(f"Exporting {len(models)} {mode} models to ONNX...")

    for model_name in sorted(models):
        try:
            start_time = time.time()
            path = export_model(model_name, mode)
            exported_paths.append(path)

            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            elapsed = time.time() - start_time
            logger.info(f"✓ {model_name} exported in {elapsed:.1f}s")

        except Exception as e:
            logger.error(f"✗ Failed to export {model_name}: {e}")
            continue

    return exported_paths


def export_all_models_all() -> List[Path]:
    """Export all models across both detection and classification."""
    all_models = list(DETECT_MODELS) + list(CLS_MODELS_EXTENDED)
    exported_paths = []

    logger.info(f"Exporting {len(all_models)} total models to ONNX...")

    for model_name in sorted(all_models):
        try:
            start_time = time.time()

            # Determine mode for this model
            if model_name in DETECT_MODELS:
                mode = "detect"
            else:
                mode = "cls"

            path = export_model(model_name, mode)
            exported_paths.append(path)

            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            elapsed = time.time() - start_time
            logger.info(f"✓ {model_name} ({mode}) exported in {elapsed:.1f}s")

        except Exception as e:
            logger.error(f"✗ Failed to export {model_name}: {e}")
            continue

    return exported_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Export models to ONNX format")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["cls", "detect", "all"],
        help="Model mode: cls for classification, detect for detection, all for all models",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to export, or 'all' for all models in the mode",
    )
    parser.add_argument(
        "--pt",
        type=str,
        help="Path to a .pt file to export directly. Mode is inferred from filename ('cls' → classification, otherwise detection) unless --mode is also given.",
    )

    args = parser.parse_args()

    if args.pt:
        pt_path = Path(args.pt)
        if not pt_path.exists():
            logger.error(f".pt file not found: {pt_path}")
            return
        if pt_path.suffix != ".pt":
            logger.error(f"Expected a .pt file, got: {pt_path}")
            return
        if args.mode and args.mode not in ("cls", "detect"):
            logger.error("--mode must be 'cls' or 'detect' when used with --pt")
            return
        inferred_mode = args.mode or ("cls" if "cls" in pt_path.stem else "detect")
        if inferred_mode == "cls" and not args.model:
            parser.error("--model (architecture name, e.g. convnext-tiny) is required when exporting a cls .pt file")
        ONNX_DIR.mkdir(exist_ok=True)
        logger.info(f"ONNX models will be saved to: {ONNX_DIR}")
        try:
            if inferred_mode == "cls":
                path = export_cls_from_pt(pt_path, args.model)
            else:
                path = export_yolo_detect_from_pt(pt_path)
            logger.info(f"Exported {pt_path.name} to {path}")
        except Exception as e:
            logger.error(f"Failed to export {pt_path}: {e}")
        logger.info("ONNX export completed!")
        return

    if not args.mode or not args.model:
        parser.error("--mode and --model are required unless --pt is given")

    # Create output directory
    ONNX_DIR.mkdir(exist_ok=True)
    logger.info(f"ONNX models will be saved to: {ONNX_DIR}")

    # Export models
    if args.mode == "all":
        if args.model.lower() == "all":
            exported_paths = export_all_models_all()
            logger.info(f"Exported {len(exported_paths)} total models to ONNX")
        else:
            # Export specific model in "all" mode (auto-detect mode)
            try:
                if args.model in DETECT_MODELS:
                    path = export_model(args.model, "detect")
                elif args.model in CLS_MODELS_EXTENDED:
                    path = export_model(args.model, "cls")
                else:
                    raise ValueError(f"Unknown model: {args.model}")
                logger.info(f"Exported {args.model} to {path}")
            except Exception as e:
                logger.error(f"Failed to export {args.model}: {e}")
                return
    elif args.model.lower() == "all":
        exported_paths = export_all_models(args.mode)
        logger.info(f"Exported {len(exported_paths)} {args.mode} models to ONNX")
    else:
        try:
            path = export_model(args.model, args.mode)
            logger.info(f"Exported {args.model} to {path}")
        except Exception as e:
            logger.error(f"Failed to export {args.model}: {e}")
            return

    logger.info("ONNX export completed!")


if __name__ == "__main__":
    main()
