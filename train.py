from __future__ import annotations

import argparse
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Tuple
import torch
import gc
import time
import numpy as np
import random
import os
import pandas as pd
from hyperparams import get_hyperparams, HPO_DATABASE
from log import logger, add_log_file
from paths import (
    RESULTS_DIR,
    RESULTS_CSV,
    DATASET_CLS_AUGMENTED_DIR,
    DATASET_DETECT_AUGMENTED_DIR,
    MODELS_DIR,
)
# from paths import DATASET_COMBINED_DIR as DATASET_AUGMENTED_DETECT_DIR
# from paths import DATASET_CLS_DIR as DATASET_AUGMENTED_CLS_DIR

parser = argparse.ArgumentParser(
    description="Train a YOLO model (classification or detection)"
)
parser.add_argument(
    "--mode",
    type=str,
    choices=["cls", "detect"],
    required=True,
    help="Training mode: 'cls' for classification, 'detect' for object detection",
)
parser.add_argument("--model", type=str, help="Base model to start training from")
parser.add_argument(
    "--hpo",
    action="store_true",
    help="Use HPO-optimized hyperparameters for supported models",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducible training (default: 42)",
)
parser.add_argument(
    "--epoch",
    type=int,
    help="Override the number of training epochs (for testing purposes)",
)

args = parser.parse_args()

if not args.model:
    raise ValueError("--model is required")
MODEL_NAME = args.model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Torch CUDA settings
torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"  # type: ignore


def setup_logging(log_file: Path) -> None:
    """Setup additional logging for training session."""
    add_log_file(log_file)
    logger.info(f"Training log file: {log_file}")


def set_seed(seed: int) -> None:
    """Set random seeds for all libraries to ensure reproducible results."""
    logger.info(f"Setting random seed to {seed} for reproducible training")

    # Python random seed
    random.seed(seed)

    # NumPy seed
    np.random.seed(seed)

    # PyTorch seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for additional libraries
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dataset_layout(dataset_dir: Path, mode: str) -> None:
    """Validate dataset structure based on mode."""
    if mode == "cls":
        # Classification uses folder-based structure: split/class/*.jpg
        required = [
            dataset_dir / "train",
            dataset_dir / "val",
            dataset_dir / "test",
        ]
        missing_msg = (
            "Classification dataset structure is missing required split folders"
        )
    else:  # detect
        # Detection uses images/labels structure
        required = [
            dataset_dir / "train" / "images",
            dataset_dir / "train" / "labels",
            dataset_dir / "val" / "images",
            dataset_dir / "val" / "labels",
            dataset_dir / "test" / "images",
            dataset_dir / "test" / "labels",
        ]
        missing_msg = "Detection dataset structure is missing required paths"

    missing = [p for p in required if not p.exists()]
    if missing:
        msg_lines = [missing_msg] + [f" - {p}" for p in missing]
        msg = "\n".join(msg_lines)
        raise FileNotFoundError(msg)


def ensure_data_yaml(dataset_dir: Path, yaml_path: Path) -> None:
    """Create data.yaml for detection mode if it doesn't exist."""
    # For augmented dataset, create yaml in the augmented dataset directory
    augmented_yaml_path = dataset_dir / "data.yaml"

    if augmented_yaml_path.exists():
        logger.info(f"Found existing data.yaml: {augmented_yaml_path}")
        return

    yaml_text = (
        f"path: {dataset_dir.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "\n"
        "nc: 2\n"
        "names:\n"
        "  0: cherry\n"
        "  1: stem\n"
    )

    augmented_yaml_path.write_text(yaml_text, encoding="utf-8")
    logger.info(f"Created data.yaml at: {augmented_yaml_path}")


def _extract_metrics(results_obj: Any, mode: str) -> Dict[str, Any]:
    """Extract metrics based on training mode."""
    metrics: Dict[str, Any] = {}

    if mode == "cls":
        # For classification, metrics are in .top1
        v = getattr(results_obj, "top1", None)
        if v is not None:
            metrics["top1"] = float(v)
    else:  # detect
        # For detection, metrics are in .box
        box = getattr(results_obj, "box", None)
        if box is not None:
            for k in ("map", "map50", "map75", "mp", "mr", "f1"):
                v = getattr(box, k, None)
                if v is None:
                    continue

                # Handle scalar
                if isinstance(v, (int, float, np.generic)):
                    metrics[f"box_{k}"] = float(v)

                # Handle numpy array / list / tuple (per-class values)
                elif isinstance(v, (np.ndarray, list, tuple)):
                    if len(v) > 0:
                        metrics[f"box_{k}"] = float(np.mean(v))
                        metrics[f"box_{k}_per_class"] = [float(x) for x in v]

        rd = getattr(results_obj, "results_dict", None)
        if isinstance(rd, dict):
            for k, v in rd.items():
                if isinstance(v, (int, float, np.generic)):
                    metrics[k] = float(v)
                elif isinstance(v, (np.ndarray, list, tuple)) and len(v) > 0:
                    metrics[k] = float(np.mean(v))

    # Speed metrics are common to both modes
    speed = getattr(results_obj, "speed", None)
    if isinstance(speed, dict):
        metrics["speed_preprocess_ms"] = float(speed.get("preprocess", 0))
        metrics["speed_inference_ms"] = float(speed.get("inference", 0))
        metrics["speed_postprocess_ms"] = float(speed.get("postprocess", 0))

    return metrics


def log_results_to_file(
    model_name: str, metrics: Dict[str, Any], train_time: float, mode: str
) -> None:
    """Append results to unified results.csv file using pandas."""
    # Use val split inference speed as representative inference time
    inference_time = metrics.get("val_speed_inference_ms", 0.0)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row_data: Dict[str, Any] = {}

    # Basic info
    row_data.update(
        {
            "model_name": model_name,
            "mode": mode,
            "seed": args.seed,
            "timestamp": timestamp,
            "train_time": train_time,
            "inference_time": inference_time,
        }
    )

    # Primary metrics (different for each mode)
    if mode == "cls":
        row_data.update(
            {
                "train_primary": metrics.get("train_top1", np.nan),
                "val_primary": metrics.get("val_top1", np.nan),
                "test_primary": metrics.get("test_top1", np.nan),
            }
        )
    else:  # detect
        row_data.update(
            {
                "train_primary": metrics.get("train_box_map50", np.nan),
                "val_primary": metrics.get("val_box_map50", np.nan),
                "test_primary": metrics.get("test_box_map50", np.nan),
            }
        )

    # Classification metrics (only for classification mode)
    cls_metrics = ["top1"]
    for split in ["train", "val", "test"]:
        for metric in cls_metrics:
            key = f"{split}_{metric}"
            row_data[key] = metrics.get(key, np.nan) if mode == "cls" else np.nan

    # Detection metrics (only for detection mode)
    det_metrics = [
        "box_map",
        "box_map50",
        "box_map75",
        "box_f1",
        "box_mp",
        "box_mr",
    ]
    for split in ["train", "val", "test"]:
        for metric in det_metrics:
            key = f"{split}_{metric}"
            row_data[key] = metrics.get(key, np.nan) if mode == "detect" else np.nan

    # Common metrics (precision, recall, f1)
    common_metrics = ["precision", "recall", "f1"]
    for split in ["train", "val", "test"]:
        for metric in common_metrics:
            key = f"{split}_{metric}"
            row_data[key] = metrics.get(key, np.nan)

    # Speed metrics
    speed_metrics = [
        "speed_preprocess_ms",
        "speed_inference_ms",
        "speed_postprocess_ms",
    ]
    for split in ["train", "val", "test"]:
        for metric in speed_metrics:
            key = f"{split}_{metric}"
            row_data[key] = metrics.get(key, np.nan)

    # Load existing CSV or create new DataFrame
    if RESULTS_CSV.exists():
        try:
            df = pd.read_csv(RESULTS_CSV)
        except Exception as e:
            logger.error(f"Failed to read existing results.csv: {e}")
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    # Append new row
    new_row = pd.DataFrame([row_data])
    df = pd.concat([df, new_row], ignore_index=True)

    # Save to CSV
    try:
        df.to_csv(RESULTS_CSV, index=False)
        logger.info(f"Results saved to {RESULTS_CSV}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")


def train_model(ts: str, mode: str) -> Tuple[Path, float]:
    """Train model based on mode."""
    # Use augmented dataset based on mode
    dataset_dir = (
        DATASET_CLS_AUGMENTED_DIR if mode == "cls" else DATASET_DETECT_AUGMENTED_DIR
    )
    data_yaml = dataset_dir / "data.yaml" if mode == "detect" else None

    logger.info(f"Validating dataset layout under {dataset_dir}")
    ensure_dataset_layout(dataset_dir, mode)

    if mode == "detect":
        ensure_data_yaml(dataset_dir, data_yaml)

    # Construct full model path from MODELS_DIR
    MODEL_PATH = MODELS_DIR / MODEL_NAME
    if not MODEL_PATH.exists():
        logger.info(f"Model not found in {MODELS_DIR}, will download: {MODEL_NAME}")
        MODEL_PATH = MODEL_NAME  # Let YOLO handle download
    else:
        logger.info(f"Using model from: {MODEL_PATH}")

    logger.info(f"Loading base model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH), task="classify" if mode == "cls" else "detect")

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Using CUDA device: {device_name}")
    else:
        logger.info("Using CPU")

    # Get hyperparameters based on flag
    hyperparams = get_hyperparams(MODEL_NAME, hpo=args.hpo)

    # Override epochs if --epoch flag is provided
    if args.epoch is not None:
        hyperparams["epochs"] = args.epoch
        logger.info(f"Overriding epochs to {args.epoch} for testing")

    # Determine parameter type for logging
    model_name_lower = MODEL_NAME.lower()
    if args.hpo and model_name_lower in HPO_DATABASE:
        param_type = f"HPO-optimized ({model_name_lower})"
    else:
        param_type = "default"
    logger.info(f"Using {param_type} hyperparameters")

    logger.info("Starting training...")
    start_time = time.time()

    # Training data parameter differs by mode
    train_data = str(dataset_dir.resolve()) if mode == "cls" else str(data_yaml)

    try:
        results = model.train(
            data=train_data,
            device=DEVICE,
            plots=True,
            project=str(RESULTS_DIR.resolve()),
            name=f"{ts}-train",
            save=True,
            exist_ok=True,
            seed=args.seed,
            **hyperparams,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise RuntimeError(f"Training failed: {e}") from e

    if results is None:
        raise RuntimeError("Training did not return a results object")

    best_path = Path(results.save_dir) / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(
            f"Training finished but best.pt not found at {best_path}"
        )

    train_time = time.time() - start_time
    logger.info(f"Training complete. Best weights: {best_path}")

    best_model = YOLO(best_path)
    best_model.export(format="onnx", opset=20)

    return best_path, train_time


def evaluate_on_splits(best_weights: Path, ts: str, mode: str) -> Dict[str, Any]:
    """Evaluate model on train, val, and test splits using batch evaluation."""
    model = YOLO(best_weights)
    all_metrics = {}

    # Use augmented dataset based on mode
    dataset_dir = (
        DATASET_CLS_AUGMENTED_DIR if mode == "cls" else DATASET_DETECT_AUGMENTED_DIR
    )
    data_yaml = dataset_dir / "data.yaml" if mode == "detect" else None

    # Evaluation data parameter differs by mode
    eval_data = str(dataset_dir.resolve()) if mode == "cls" else str(data_yaml)

    hyperparams = get_hyperparams(MODEL_NAME, hpo=args.hpo)

    # Evaluate on all splits
    for split in ["train", "val", "test"]:
        logger.info(f"Evaluating on {split} split...")

        # Clear GPU memory before evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        split_results = model.val(
            data=eval_data,
            split=split,
            device=DEVICE,
            project=str(RESULTS_DIR),
            name=f"{ts}-{split}",
            exist_ok=True,
            batch=(hyperparams.get("batch", 16)) * 2,
        )

        # Extract metrics and add split prefix
        split_metrics = _extract_metrics(split_results, mode)
        for key, value in split_metrics.items():
            all_metrics[f"{split}_{key}"] = value

        # For detection, map YOLO's metrics to common names
        if mode == "detect":
            all_metrics[f"{split}_precision"] = split_metrics.get("box_mp", 0.0)
            all_metrics[f"{split}_recall"] = split_metrics.get("box_mr", 0.0)
            all_metrics[f"{split}_f1"] = split_metrics.get("box_f1", 0.0)

    return all_metrics


def main() -> None:
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = RESULTS_DIR / f"{ts}-run.log"
    setup_logging(log_file)

    # Set random seed for reproducible training
    set_seed(args.seed)

    logger.info(
        "Starting %s training with model: %s (seed: %d)",
        args.mode,
        MODEL_NAME,
        args.seed,
    )

    # Train model
    best_weights, train_time = train_model(ts, args.mode)

    # Evaluate on all splits
    all_metrics = evaluate_on_splits(best_weights, ts, args.mode)

    # Log results
    log_results_to_file(MODEL_NAME, all_metrics, train_time, args.mode)

    logger.info("\nTRAINING COMPLETED")
    logger.info(f"Model: {MODEL_NAME} ({args.mode}) - Seed: {args.seed}")
    logger.info(f"Train Time: {train_time:.2f}s")

    logger.info("Training completed in %.2f seconds", train_time)


if __name__ == "__main__":
    main()
