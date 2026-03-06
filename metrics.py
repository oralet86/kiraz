"""
Metrics computation module for Kiraz project.

Provides additional metrics beyond what YOLO's built-in val() exposes:
  - Classification: precision, recall, F1 (macro-averaged) derived from confusion matrix
  - Detection: mAP50-95 (re-exposed for clarity), mean IoU over matched prediction/GT pairs
"""

from __future__ import annotations

import argparse
from typing import Any, Dict

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou

from log import logger
from paths import DATASET_CLS_AUGMENTED_DIR, DATASET_DETECT_AUGMENTED_DIR


# IoU threshold for considering a detection a true positive when computing mean IoU
_IOU_MATCH_THRESHOLD = 0.5


def cls_precision_recall_f1(val_result: Any) -> Dict[str, float]:
    """Compute macro-averaged precision, recall and F1 for a classification val() result.

    Derives metrics from the confusion matrix stored on the validator.  The
    confusion matrix has shape (nc, nc) where entry [pred, gt] counts the
    number of samples predicted as class ``pred`` whose true label is ``gt``.

    Args:
        val_result: Object returned by ``model.val()`` for a classification task.

    Returns:
        Dictionary with keys ``precision``, ``recall``, ``f1``.
    """
    cm = getattr(val_result, "confusion_matrix", None)
    if cm is None:
        raise ValueError("val_result has no confusion_matrix attribute")

    matrix: np.ndarray = cm.matrix  # shape (nc, nc), float
    nc = matrix.shape[0]

    precisions: list[float] = []
    recalls: list[float] = []

    for c in range(nc):
        tp = matrix[c, c]
        fp = matrix[c, :].sum() - tp  # predicted as c but not c
        fn = matrix[:, c].sum() - tp  # true c but predicted as something else

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(float(precision))
        recalls.append(float(recall))

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    denom = macro_precision + macro_recall
    macro_f1 = float(2 * macro_precision * macro_recall / denom) if denom > 0 else 0.0

    return {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
    }


def detect_metrics(
    model: YOLO,
    data: str,
    split: str,
    device: str,
    batch: int,
) -> Dict[str, float]:
    """Run a single validation pass that captures mAP50-95 and mean IoU simultaneously.

    For mean IoU, matched prediction/GT box pairs at IoU >= ``_IOU_MATCH_THRESHOLD``
    are identified per image and their raw IoU values are averaged across the split.

    Args:
        model: Trained YOLO detection model.
        data: Path to data.yaml.
        split: One of "train", "val", "test".
        device: Torch device string.
        batch: Batch size.

    Returns:
        Dictionary with keys ``map50``, ``map50_95``, ``mean_iou``.
    """
    iou_values: list[float] = []

    from ultralytics.models.yolo.detect.val import DetectionValidator

    class _IouCapturingValidator(DetectionValidator):
        """Subclass that captures raw per-pair IoU during validation."""

        def _process_batch(
            self,
            preds: Dict[str, torch.Tensor],
            batch_prepared: Dict[str, Any],
        ) -> Dict[str, np.ndarray]:
            if batch_prepared["cls"].shape[0] > 0 and preds["cls"].shape[0] > 0:
                iou_matrix = box_iou(batch_prepared["bboxes"], preds["bboxes"])
                if iou_matrix.numel() > 0:
                    best_iou_per_gt = iou_matrix.max(dim=1).values
                    matched = best_iou_per_gt[best_iou_per_gt >= _IOU_MATCH_THRESHOLD]
                    iou_values.extend(matched.cpu().tolist())
            return super()._process_batch(preds, batch_prepared)

    validator = _IouCapturingValidator(
        args=dict(
            data=data,
            split=split,
            device=device,
            batch=batch,
            verbose=False,
            plots=False,
            save=False,
            save_json=False,
            save_txt=False,
        )
    )
    result = validator(model=model.model)

    map50 = float(result.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(result.get("metrics/mAP50-95(B)", 0.0))
    box_mp = float(result.get("metrics/precision(B)", 0.0))
    box_mr = float(result.get("metrics/recall(B)", 0.0))
    denom = box_mp + box_mr
    box_f1 = float(2 * box_mp * box_mr / denom) if denom > 0 else 0.0
    mean_iou = float(np.mean(iou_values)) if iou_values else 0.0

    return {
        "map50": map50,
        "map50_95": map50_95,
        "box_mp": box_mp,
        "box_mr": box_mr,
        "box_f1": box_f1,
        "mean_iou": mean_iou,
    }


# ---------------------------------------------------------------------------
# Standalone test / smoke-test entry point
# ---------------------------------------------------------------------------


def _test_classification(model_name: str, device: str) -> None:
    """Smoke-test classification metrics on the augmented classification dataset."""
    logger.info("=== Classification metrics smoke test ===")
    logger.info(f"Model: {model_name}  |  Dataset: {DATASET_CLS_AUGMENTED_DIR}")

    model = YOLO(model_name, task="classify")

    data = str(DATASET_CLS_AUGMENTED_DIR.resolve())

    for split in ("train", "val", "test"):
        result = model.val(
            data=data, split=split, device=device, verbose=False, plots=False
        )
        metrics = cls_precision_recall_f1(result)
        top1 = float(getattr(result, "top1", float("nan")))
        logger.info(
            f"  [{split:5s}]  top1={top1:.4f}  "
            f"precision={metrics['precision']:.4f}  "
            f"recall={metrics['recall']:.4f}  "
            f"f1={metrics['f1']:.4f}"
        )


def _test_detection(model_name: str, device: str) -> None:
    """Smoke-test detection metrics on the augmented detection dataset."""
    logger.info("=== Detection metrics smoke test ===")
    logger.info(f"Model: {model_name}  |  Dataset: {DATASET_DETECT_AUGMENTED_DIR}")

    model = YOLO(model_name, task="detect")

    data_yaml = str((DATASET_DETECT_AUGMENTED_DIR / "data.yaml").resolve())

    for split in ("train", "val", "test"):
        metrics = detect_metrics(
            model=model,
            data=data_yaml,
            split=split,
            device=device,
            batch=12,
        )
        logger.info(
            f"  [{split:5s}]  map50={metrics['map50']:.4f}"
            f"  map50-95={metrics['map50_95']:.4f}"
            f"  mean_iou={metrics['mean_iou']:.4f}"
            f"  mp={metrics['box_mp']:.4f}"
            f"  mr={metrics['box_mr']:.4f}"
            f"  f1={metrics['box_f1']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test metrics.py functions")
    parser.add_argument(
        "--mode",
        choices=["cls", "detect", "both"],
        default="both",
        help="Which pipeline to test",
    )
    parser.add_argument(
        "--cls-model",
        default="yolo11n-cls.pt",
        help="Classification model name/path",
    )
    parser.add_argument(
        "--det-model",
        default="yolo11n.pt",
        help="Detection model name/path",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    if args.mode in ("cls", "both"):
        _test_classification(args.cls_model, device)

    if args.mode in ("detect", "both"):
        _test_detection(args.det_model, device)


if __name__ == "__main__":
    main()
