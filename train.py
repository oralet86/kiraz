from __future__ import annotations

import argparse
import gc
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from hyperparams import HPO_DATABASE, get_hyperparams
from log import add_log_file, logger
from metrics import cls_precision_recall_f1, detect_metrics
from paths import (
    DATASET_CLS_AUGMENTED_DIR,
    DATASET_DETECT_AUGMENTED_DIR,
    MODELS_DIR,
    RESULTS_CSV,
    RESULTS_DIR,
)

DATASET_DETECT = DATASET_DETECT_AUGMENTED_DIR
DATASET_CLS = DATASET_CLS_AUGMENTED_DIR

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
parser.add_argument(
    "--batch-mult",
    type=float,
    default=1.0,
    help="Batch size multiplier (default: 1.0)",
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


def ensure_data_yaml(dataset_dir: Path, yaml_path: Path | None) -> None:
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


def _extract_cls_metrics(results_obj: Any) -> Dict[str, Any]:
    """Extract classification metrics from a val() result object."""
    metrics: Dict[str, Any] = {}

    v = getattr(results_obj, "top1", None)
    if v is not None:
        metrics["top1"] = float(v)

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
    inference_time = metrics.get("val_speed_inference_ms", 0.0)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row_data: Dict[str, Any] = {
        "model_name": model_name,
        "mode": mode,
        "seed": args.seed,
        "timestamp": timestamp,
        "train_time": train_time,
        "inference_time": inference_time,
    }

    # Classification metrics
    cls_metrics = ["top1", "precision", "recall", "f1"]
    for split in ["train", "val", "test"]:
        for metric in cls_metrics:
            key = f"{split}_{metric}"
            row_data[key] = metrics.get(key, np.nan) if mode == "cls" else np.nan

    # Detection metrics
    det_metrics = ["map50", "map50_95", "mean_iou", "box_mp", "box_mr", "box_f1"]
    for split in ["train", "val", "test"]:
        for metric in det_metrics:
            key = f"{split}_{metric}"
            row_data[key] = metrics.get(key, np.nan) if mode == "detect" else np.nan

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


def train_model(ts: str, mode: str) -> Tuple[Path, YOLO, float]:
    """Train model based on mode."""
    # Use augmented dataset based on mode
    dataset_dir = DATASET_CLS if mode == "cls" else DATASET_DETECT
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

    # Apply batch size multiplier and ensure even integer
    if "batch" in hyperparams:
        original_batch = hyperparams["batch"]
        scaled_batch = int(original_batch * args.batch_mult)
        # Ensure even integer, roll down if odd
        if scaled_batch % 2 != 0:
            scaled_batch -= 1
        # Ensure at least 2
        scaled_batch = max(2, scaled_batch)
        hyperparams["batch"] = scaled_batch
        logger.info(
            f"Batch size: {original_batch} * {args.batch_mult} → {scaled_batch}"
        )

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

    # Export model before cleanup
    best_model = YOLO(best_path)
    best_model.export(format="onnx", opset=20)

    # Clean up training model and results, keep best_model for evaluation
    del model
    del results

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return best_path, best_model, train_time


def evaluate_on_splits(model: YOLO, ts: str, mode: str) -> Dict[str, Any]:
    """Evaluate model on val and test splits."""
    all_metrics: Dict[str, Any] = {}

    dataset_dir = DATASET_CLS if mode == "cls" else DATASET_DETECT
    hyperparams = get_hyperparams(MODEL_NAME, hpo=args.hpo)
    original_batch = hyperparams.get("batch", 8)
    scaled_batch = int(original_batch * args.batch_mult)
    if scaled_batch % 2 != 0:
        scaled_batch -= 1
    eval_batch = max(2, scaled_batch)

    for split in ["val", "test"]:
        logger.info(f"Evaluating on {split} split...")

        if mode == "cls":
            split_results = model.val(
                data=str(dataset_dir.resolve()),
                split=split,
                device=DEVICE,
                project=str(RESULTS_DIR),
                name=f"{ts}-{split}",
                exist_ok=True,
                batch=eval_batch,
            )
            split_metrics = _extract_cls_metrics(split_results)
            prf = cls_precision_recall_f1(split_results)
            split_metrics.update(prf)
        else:
            data_yaml = str((dataset_dir / "data.yaml").resolve())
            split_metrics = detect_metrics(
                model=model,
                data=data_yaml,
                split=split,
                device=DEVICE,
                batch=eval_batch,
            )

        for key, value in split_metrics.items():
            all_metrics[f"{split}_{key}"] = value

    del model
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return all_metrics


def cleanup_cuda() -> None:
    """Clean up CUDA memory with optional aggressive cleanup."""
    if torch.cuda.is_available():
        # Force garbage collection across all generations
        for _ in range(2):
            gc.collect()

        # Clear CUDA cache
        torch.cuda.empty_cache()

        # Additional aggressive cleanup
        torch.cuda.synchronize()  # Ensure all operations complete
        torch.cuda.empty_cache()  # Clear again after sync

        # Reset peak memory stats to free up tracking overhead
        torch.cuda.reset_peak_memory_stats()

        # Force cleanup of any remaining tensors
        for _ in range(2):
            torch.cuda.empty_cache()
            gc.collect()


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

    cleanup_cuda()

    # Train model
    best_weights, best_model, train_time = train_model(ts, args.mode)

    cleanup_cuda()

    # Evaluate on val and test splits
    all_metrics = evaluate_on_splits(best_model, ts, args.mode)

    # Log results
    log_results_to_file(MODEL_NAME, all_metrics, train_time, args.mode)

    logger.info("\nTRAINING COMPLETED")
    logger.info(f"Model: {MODEL_NAME} ({args.mode}) - Seed: {args.seed}")
    logger.info(f"Train Time: {train_time:.2f}s")

    logger.info("Training completed in %.2f seconds", train_time)

    # Final cleanup before exit
    cleanup_cuda()
    logger.info("Final cleanup completed")


if __name__ == "__main__":
    main()
