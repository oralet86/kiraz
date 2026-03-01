from __future__ import annotations

import argparse
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple
import torch
import time
import numpy as np
import random
import os
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from hyperparams import get_hyperparams, HPO_DATABASE
from log import logger, add_log_file
from paths import (
    RESULTS_DIR,
    DATASET_CLS_DIR,
    DATASET_COMBINED_DIR,
    DATA_YAML,
    RESULTS_CSV,
    MODELS_DIR,
)

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

args = parser.parse_args()

# Validate required arguments based on mode
if args.mode == "cls":
    if not args.model:
        raise ValueError("--model is required for classification mode")
    MODEL_NAME = args.model
else:  # detect
    if not args.model:
        raise ValueError("--model is required for detection mode")
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

    logger.info("Random seeds set for reproducible training")


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
    if yaml_path.exists():
        logger.info("Found existing data.yaml: %s", yaml_path)
        return

    yaml_text = (
        f"path: {dataset_dir.resolve().as_posix()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"test: test/images\n"
        f"\n"
        f"names:\n"
        f"  0: stem\n"
    )

    yaml_path.write_text(yaml_text, encoding="utf-8")
    logger.info("Created data.yaml at: %s", yaml_path)


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


def log_metrics(metrics: Dict[str, Any], mode: str, prefix: str = "Metrics") -> None:
    """Log metrics to console based on mode."""
    if not metrics:
        logger.warning("%s: none found", prefix)
        return

    logger.info(prefix)

    if mode == "cls":
        preferred_order = [
            "top1",
            "roc_auc",
            "precision",
            "recall",
            "f1",
            "speed_preprocess_ms",
            "speed_inference_ms",
            "speed_postprocess_ms",
        ]
    else:  # detect
        preferred_order = [
            "box_map",
            "box_map50",
            "box_map75",
            "box_f1",
            "box_mp",
            "box_mr",
            "speed_preprocess_ms",
            "speed_inference_ms",
            "speed_postprocess_ms",
        ]

    # Log key metrics first
    for key in preferred_order:
        if key in metrics:
            val = metrics[key]
            if isinstance(val, float):
                logger.info("  %-24s %.4f", key + ":", val)
            else:
                logger.info("  %-24s %s", key + ":", val)

    # Log remaining metrics
    remaining = sorted(k for k in metrics if k not in preferred_order)
    for key in remaining:
        val = metrics[key]
        if isinstance(val, float):
            logger.info("  %-24s %.4f", key + ":", val)
        else:
            logger.info("  %-24s %s", key + ":", val)


def calculate_mean_iou_detect(
    model: YOLO, dataset_dir: Path, split: str
) -> Dict[str, float]:
    """Calculate mean IoU for detection by matching predictions to ground truth using box_iou."""
    img_dir = dataset_dir / split / "images"
    lbl_dir = dataset_dir / split / "labels"

    if not img_dir.exists():
        return {}

    iou_values: List[float] = []

    for img_path in img_dir.glob("*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue

        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        try:
            results = model(str(img_path), verbose=False)
            preds = results[0].boxes
            if preds is None or len(preds) == 0:
                continue

            pred_xyxy = preds.xyxy  # (N, 4) tensor

            # Load ground truth in YOLO format and convert to xyxy
            img = results[0].orig_img
            h, w = img.shape[:2]
            gt_boxes: List[List[float]] = []
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    _, cx, cy, bw, bh = map(float, parts[:5])
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    x2 = (cx + bw / 2) * w
                    y2 = (cy + bh / 2) * h
                    gt_boxes.append([x1, y1, x2, y2])

            if not gt_boxes:
                continue

            gt_xyxy = torch.tensor(
                gt_boxes, dtype=torch.float32, device=pred_xyxy.device
            )

            # box_iou returns (N_pred, N_gt) matrix; take max IoU per GT box
            iou_matrix = box_iou(pred_xyxy, gt_xyxy)  # (N_pred, N_gt)
            matched_iou = iou_matrix.max(dim=0).values  # best pred IoU per GT
            iou_values.extend(matched_iou.cpu().tolist())

        except Exception as e:
            logger.warning("IoU calculation failed for %s: %s", img_path.name, e)

    if not iou_values:
        return {}

    return {"mean_iou": float(np.mean(iou_values))}


def calculate_additional_metrics_cls(
    model: YOLO, val_files: List[str], dataset_dir: Path
) -> Dict[str, float]:
    """Calculate precision, recall, F1, and ROC-AUC for classification."""
    try:
        # Get predictions for validation files
        y_true = []
        y_pred = []
        y_scores = []  # For ROC-AUC (probability scores)

        class_dirs = sorted(
            [d for d in (dataset_dir / "train").iterdir() if d.is_dir()]
        )
        class_names = [d.name for d in class_dirs]

        for file_path in val_files:
            # True label
            true_class = Path(file_path).parent.name
            true_label = class_names.index(true_class)
            y_true.append(true_label)

            # Predicted label and probabilities
            results = model(file_path)
            pred_class = results[0].probs.top1
            y_pred.append(pred_class)

            # Get probability scores for ROC-AUC (probability of positive class)
            # For binary classification, we'll use the probability of class 1
            probs = results[0].probs.data
            if len(probs) == 2:  # Binary classification
                y_scores.append(float(probs[1]))  # Probability of class 1
            else:
                y_scores.append(
                    float(probs[pred_class])
                )  # Fallback to predicted class prob

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        # Calculate ROC-AUC for binary classification
        roc_auc = 0.0
        if len(set(y_true)) == 2:  # Only for binary classification
            try:
                roc_auc = roc_auc_score(y_true, y_scores)
            except Exception as e:
                logger.warning("Could not calculate ROC-AUC: %s", e)

        return {"precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc}
    except Exception as e:
        logger.warning("Could not calculate additional metrics: %s", e)
        return {}


def log_results_to_file(
    model_name: str, metrics: Dict[str, Any], train_time: float, mode: str
) -> None:
    """Append results to unified results.csv file."""
    # Use val split inference speed as representative inference time
    inference_time = metrics.get("val_speed_inference_ms", 0.0)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    results_file = RESULTS_CSV

    # Unified header for both modes; primary metric column named generically
    if not results_file.exists():
        header = "model_name,mode,seed,train_primary,train_prec,train_recall,train_f1,val_primary,val_prec,val_recall,val_f1,test_primary,test_prec,test_recall,test_f1,train_time,inference_time,timestamp\n"
        try:
            with open(results_file, "w", encoding="utf-8") as f:
                f.write(header)
        except (IOError, OSError) as e:
            logger.error("Failed to create results file: %s", e)
            return

    if mode == "cls":
        train_primary = metrics.get("train_top1", 0.0)
        val_primary = metrics.get("val_top1", 0.0)
        test_primary = metrics.get("test_top1", 0.0)
    else:  # detect
        train_primary = metrics.get("train_box_map50", 0.0)
        val_primary = metrics.get("val_box_map50", 0.0)
        test_primary = metrics.get("test_box_map50", 0.0)

    train_prec = metrics.get("train_precision", 0.0)
    train_recall = metrics.get("train_recall", 0.0)
    train_f1 = metrics.get("train_f1", 0.0)
    val_prec = metrics.get("val_precision", 0.0)
    val_recall = metrics.get("val_recall", 0.0)
    val_f1 = metrics.get("val_f1", 0.0)
    test_prec = metrics.get("test_precision", 0.0)
    test_recall = metrics.get("test_recall", 0.0)
    test_f1 = metrics.get("test_f1", 0.0)

    line = f'"{model_name}",{mode},{args.seed},{train_primary:.6f},{train_prec:.6f},{train_recall:.6f},{train_f1:.6f},{val_primary:.6f},{val_prec:.6f},{val_recall:.6f},{val_f1:.6f},{test_primary:.6f},{test_prec:.6f},{test_recall:.6f},{test_f1:.6f},{train_time:.2f},{inference_time:.2f},"{timestamp}"\n'

    try:
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(line)
    except (IOError, OSError) as e:
        logger.error("Failed to write results to file: %s", e)


def train_model(ts: str, mode: str) -> Tuple[Path, Dict[str, Any], float]:
    """Train model based on mode."""
    # Use appropriate dataset based on mode
    dataset_dir = DATASET_CLS_DIR if mode == "cls" else DATASET_COMBINED_DIR
    data_yaml = DATA_YAML if mode == "detect" else None

    logger.info("Validating dataset layout under %s", dataset_dir)
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

    logger.info("Loading base model: %s", MODEL_PATH)
    model = YOLO(str(MODEL_PATH), task="classify" if mode == "cls" else "detect")

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info("Using CUDA device: %s", device_name)
    else:
        logger.info("Using CPU")

    # Get hyperparameters based on flag
    hyperparams = get_hyperparams(MODEL_NAME, hpo=args.hpo)

    # Determine parameter type for logging
    model_name_lower = MODEL_NAME.lower()
    if args.hpo and model_name_lower in HPO_DATABASE:
        param_type = f"HPO-optimized ({model_name_lower})"
    else:
        param_type = "default"
    logger.info("Using %s hyperparameters", param_type)

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
        logger.error("Training failed: %s", e)
        raise RuntimeError(f"Training failed: {e}") from e

    if results is None:
        raise RuntimeError("Training did not return a results object")

    # Get training metrics
    train_metrics = _extract_metrics(results, mode)

    best_path = Path(results.save_dir) / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(
            f"Training finished but best.pt not found at {best_path}"
        )

    train_time = time.time() - start_time
    logger.info("Training complete. Best weights: %s", best_path)

    best_model = YOLO(best_path)
    best_model.export(format="onnx")

    return best_path, train_metrics, train_time


def evaluate_on_splits(best_weights: Path, ts: str, mode: str) -> Dict[str, Any]:
    """Evaluate model on train, val, and test splits."""
    model = YOLO(best_weights)
    all_metrics = {}

    # Use appropriate dataset based on mode
    dataset_dir = DATASET_CLS_DIR if mode == "cls" else DATASET_COMBINED_DIR
    data_yaml = DATA_YAML if mode == "detect" else None

    # Evaluation data parameter differs by mode
    eval_data = str(dataset_dir.resolve()) if mode == "cls" else str(data_yaml)

    # Evaluate on all splits
    for split in ["train", "val", "test"]:
        logger.info("Evaluating on %s split...", split)

        try:
            split_results = model.val(
                data=eval_data,
                split=split,
                device=DEVICE,
                project=str(RESULTS_DIR),
                name=f"{ts}-{split}",
                exist_ok=True,
            )
        except Exception as e:
            logger.error("Failed to evaluate on %s split: %s", split, e)
            continue

        # Extract metrics and add split prefix
        split_metrics = _extract_metrics(split_results, mode)
        for key, value in split_metrics.items():
            all_metrics[f"{split}_{key}"] = value

        # Calculate additional metrics based on mode
        if mode == "cls":
            # For classification, calculate precision, recall, F1
            split_files = []
            split_dir = dataset_dir / split
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    for img_file in class_dir.glob("*"):
                        if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                            split_files.append(str(img_file))

            additional_metrics = calculate_additional_metrics_cls(
                model, split_files, dataset_dir
            )
            for key, value in additional_metrics.items():
                all_metrics[f"{split}_{key}"] = value
        else:  # detect
            # For detection, derive precision, recall, F1 from YOLO's val() metrics (box_mp, box_mr)
            prec = split_metrics.get("box_mp", 0.0)
            rec = split_metrics.get("box_mr", 0.0)
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            all_metrics[f"{split}_precision"] = prec
            all_metrics[f"{split}_recall"] = rec
            all_metrics[f"{split}_f1"] = f1

            iou_metrics = calculate_mean_iou_detect(model, dataset_dir, split)
            for key, value in iou_metrics.items():
                all_metrics[f"{split}_{key}"] = value

    # Log all metrics
    log_metrics(all_metrics, mode)
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
    best_weights, train_metrics, train_time = train_model(ts, args.mode)

    # Evaluate on all splits
    all_metrics = evaluate_on_splits(best_weights, ts, args.mode)

    # Combine train metrics with evaluation metrics
    all_metrics.update(train_metrics)

    # Log results
    log_results_to_file(MODEL_NAME, all_metrics, train_time, args.mode)

    logger.info("Training completed in %.2f seconds", train_time)


if __name__ == "__main__":
    main()
