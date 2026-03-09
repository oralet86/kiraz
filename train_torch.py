"""PyTorch-native training for non-YOLO detection and classification models.

Supported detection:    faster-rcnn-r50, faster-rcnn-r101, detr-r50, detr-r101
Supported classification: resnet50, resnet101, efficientnet-b{0..3},
  convnext-tiny, convnext-small, convnext-base,
  convnextv2-atto, convnextv2-femto, convnextv2-pico,
  convnextv2-nano, convnextv2-tiny, convnextv2-base,
  vit-small, vit-base, deit-small, deit-base,
  mobilevit-xxs, mobilevit-xs, mobilevit-s,
  mobilevitv2-050, mobilevitv2-075, mobilevitv2-100,
  swin-tiny,
  mobilenet-v2, mobilenet-v3-small, mobilenet-v3-large
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision.ops import box_iou
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision

from models import (
    CLS_MODELS,
    DETECT_MODELS,
    FRCNN_MODELS,
    build_cls_model,
    build_detr_model,
    build_frcnn_model,
)
from hyperparams import get_torch_hyperparams
from log import logger, add_log_file
from paths import (
    DATASET_CLS_AUGMENTED_DIR,
    DATASET_DETECT_AUGMENTED_DIR,
    RESULTS_DIR,
    RESULTS_CSV,
)

# Constants

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
NUM_CLS_CLASSES = 2
NUM_DETECT_CLASSES = 3  # background=0, cherry=1, stem=2
NUM_DETR_LABELS = 2  # cherry=0, stem=1
DETECT_SCORE_THRESH = 0.5
IOU_MATCH_THRESH = 0.5

# Argument parsing

parser = argparse.ArgumentParser(
    description="Train non-YOLO detection and classification models"
)
parser.add_argument("--mode", type=str, choices=["cls", "detect"], required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epoch", type=int, help="Override number of training epochs")
parser.add_argument("--batch-mult", type=float, default=1.0)

args = parser.parse_args()
MODEL_NAME = args.model.lower()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"  # type: ignore

# Datasets


class YoloDetectionDataset(Dataset):
    """YOLO-format detection split for Faster R-CNN.

    Labels are 1-indexed (0=bg, 1=cherry, 2=stem).
    Boxes are absolute pixel [x1, y1, x2, y2].
    """

    def __init__(self, split_dir: Path, imgsz: int, train: bool) -> None:
        self.imgsz = imgsz
        self.train = train
        self.label_dir = split_dir / "labels"
        self.image_paths: List[Path] = sorted(
            p
            for p in (split_dir / "images").iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_boxes(
        self, img_path: Path, w: int, h: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        label_path = self.label_dir / img_path.with_suffix(".txt").name
        boxes: List[List[float]] = []
        labels: List[int] = []
        if label_path.exists():
            for line in label_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                boxes.append(
                    [
                        (cx - bw / 2) * w,
                        (cy - bh / 2) * h,
                        (cx + bw / 2) * w,
                        (cy + bh / 2) * h,
                    ]
                )
                labels.append(cls + 1)
        return (
            torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            torch.as_tensor(labels, dtype=torch.int64),
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        boxes, labels = self._load_boxes(img_path, orig_w, orig_h)

        image = image.resize((self.imgsz, self.imgsz), Image.BILINEAR)
        sx, sy = self.imgsz / orig_w, self.imgsz / orig_h
        if boxes.numel() > 0:
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy

        if self.train and random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if boxes.numel() > 0:
                tmp = boxes.clone()
                boxes[:, 0] = self.imgsz - tmp[:, 2]
                boxes[:, 2] = self.imgsz - tmp[:, 0]

        img_t = T.functional.to_tensor(image)
        img_t = T.functional.normalize(img_t, IMAGENET_MEAN, IMAGENET_STD)
        return img_t, {"boxes": boxes, "labels": labels}


class DetrDetectionDataset(Dataset):
    """YOLO-format detection split for DETR.

    Returns PIL images and labels with 0-indexed class_labels and
    normalized [cx, cy, w, h] boxes.
    """

    def __init__(self, split_dir: Path) -> None:
        self.label_dir = split_dir / "labels"
        self.image_paths: List[Path] = sorted(
            p
            for p in (split_dir / "images").iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Image.Image, Dict[str, torch.Tensor], Tuple[int, int]]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        label_path = self.label_dir / img_path.with_suffix(".txt").name

        cls_list: List[int] = []
        box_list: List[List[float]] = []
        if label_path.exists():
            for line in label_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_list.append(int(parts[0]))
                box_list.append([float(x) for x in parts[1:5]])

        label = {
            "class_labels": torch.as_tensor(cls_list, dtype=torch.int64),
            "boxes": torch.as_tensor(box_list, dtype=torch.float32).reshape(-1, 4),
        }
        return image, label, (orig_h, orig_w)


def _frcnn_collate(
    batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


def _detr_collate(
    batch: List[Tuple[Image.Image, Dict[str, torch.Tensor], Tuple[int, int]]],
) -> Tuple[List[Image.Image], List[Dict[str, torch.Tensor]], List[Tuple[int, int]]]:
    images, labels, sizes = zip(*batch)
    return list(images), list(labels), list(sizes)


# Training helpers


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logging(log_file: Path) -> None:
    """Attach per-run log file to the shared logger."""
    add_log_file(log_file)
    logger.info(f"Training log file: {log_file}")


def apply_batch_mult(base_batch: int, mult: float) -> int:
    """Scale base batch size; ensure even integer >= 2."""
    scaled = int(base_batch * mult)
    if scaled % 2 != 0:
        scaled -= 1
    return max(2, scaled)


def cleanup_cuda() -> None:
    """Release CUDA memory aggressively."""
    if not torch.cuda.is_available():
        return
    for _ in range(2):
        gc.collect()
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# Training loops


def train_cls_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    criterion: nn.Module,
    device: str,
    epoch: int,
    epochs: int,
) -> float:
    """One classification training epoch. Returns mean loss."""
    model.train()
    total = 0.0
    count = 0
    pbar = tqdm(loader, desc=f"E{epoch}/{epochs} train", leave=False, unit="batch")
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(
            enabled=scaler.is_enabled(),
            device_type="cuda" if DEVICE == "cuda" else "cpu",
        ):
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
        count += 1
        pbar.set_postfix(loss=f"{total / count:.4f}")
    return total / max(len(loader), 1)


def train_frcnn_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: str,
    epoch: int,
    epochs: int,
) -> float:
    """One Faster R-CNN training epoch. Returns mean loss."""
    model.train()
    total = 0.0
    count = 0
    pbar = tqdm(loader, desc=f"E{epoch}/{epochs} train", leave=False, unit="batch")
    for images, targets in pbar:
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        with autocast(
            enabled=scaler.is_enabled(),
            device_type="cuda" if DEVICE == "cuda" else "cpu",
        ):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
        count += 1
        pbar.set_postfix(loss=f"{total / count:.4f}")
    return total / max(len(loader), 1)


def train_detr_epoch(
    model: nn.Module,
    processor: "DetrImageProcessor",
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: str,
    epoch: int,
    epochs: int,
) -> float:
    """One DETR training epoch. Returns mean loss."""
    model.train()
    total = 0.0
    count = 0
    pbar = tqdm(loader, desc=f"E{epoch}/{epochs} train", leave=False, unit="batch")
    for pil_images, labels, _ in pbar:
        enc = processor(images=pil_images, return_tensors="pt")
        pixel_values = enc["pixel_values"].to(device, non_blocking=True)
        pixel_mask = enc["pixel_mask"].to(device, non_blocking=True)
        batch_labels = [{k: v.to(device) for k, v in lbl.items()} for lbl in labels]
        optimizer.zero_grad(set_to_none=True)
        with autocast(
            enabled=scaler.is_enabled(),
            device_type="cuda" if DEVICE == "cuda" else "cpu",
        ):
            out = model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                labels=batch_labels,
            )
            loss = out.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
        count += 1
        pbar.set_postfix(loss=f"{total / count:.4f}")
    return total / max(len(loader), 1)


# Evaluation functions


def _match_preds_to_gt(
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
) -> Tuple[int, int, int, List[float]]:
    """Greedy IoU matching at IOU_MATCH_THRESH. Returns (tp, fp, fn, matched_ious)."""
    n_pred, n_gt = len(pred_boxes), len(gt_boxes)
    if n_pred == 0:
        return 0, 0, n_gt, []
    if n_gt == 0:
        return 0, n_pred, 0, []
    iou_mat = box_iou(pred_boxes, gt_boxes)  # (n_pred, n_gt)
    matched_gt: set[int] = set()
    tp, fp = 0, 0
    iou_values: List[float] = []
    for pi in range(n_pred):
        best_iou, best_gi = -1.0, -1
        for gi in range(n_gt):
            if gi in matched_gt or int(pred_labels[pi]) != int(gt_labels[gi]):
                continue
            val = float(iou_mat[pi, gi])
            if val > best_iou:
                best_iou, best_gi = val, gi
        if best_iou >= IOU_MATCH_THRESH:
            tp += 1
            iou_values.append(best_iou)
            matched_gt.add(best_gi)
        else:
            fp += 1
    fn = n_gt - len(matched_gt)
    return tp, fp, fn, iou_values


def _compute_detect_metrics(
    preds: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
) -> Dict[str, float]:
    """mAP50/95, box_mp, box_mr, box_f1, mean_iou from collected preds/targets.

    All boxes are absolute pixel [x1, y1, x2, y2]. Labels are 1-indexed.
    """
    metric = MeanAveragePrecision(iou_type="bbox")
    metric.update(preds, targets)
    result = metric.compute()
    map50 = float(result["map_50"])
    map50_95 = float(result["map"])

    total_tp, total_fp, total_fn = 0, 0, 0
    all_ious: List[float] = []
    for pred, gt in zip(preds, targets):
        high_conf = pred["scores"] >= DETECT_SCORE_THRESH
        tp, fp, fn, ious = _match_preds_to_gt(
            pred["boxes"][high_conf].cpu(),
            pred["labels"][high_conf].cpu(),
            gt["boxes"].cpu(),
            gt["labels"].cpu(),
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_ious.extend(ious)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    denom = precision + recall
    f1 = 2 * precision * recall / denom if denom > 0 else 0.0
    return {
        "map50": map50,
        "map50_95": map50_95,
        "box_mp": precision,
        "box_mr": recall,
        "box_f1": f1,
        "mean_iou": float(np.mean(all_ious)) if all_ious else 0.0,
    }


def eval_cls_split(
    model: nn.Module,
    dataset_dir: Path,
    split: str,
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
) -> Dict[str, float]:
    """Evaluate classification model on one split. Returns acc/P/R/F1 + speeds."""
    transform = T.Compose(
        [
            T.Resize(imgsz + 32),
            T.CenterCrop(imgsz),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    dataset = ImageFolder(str(dataset_dir / split), transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device == "cuda"),
    )

    total_pre = total_inf = total_post = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            t0 = time.perf_counter()
            images = images.to(device, non_blocking=True)
            t1 = time.perf_counter()
            with autocast(
                enabled=(device == "cuda"),
                device_type="cuda" if device == "cuda" else "cpu",
            ):
                logits = model(images)
            if device == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            preds = logits.argmax(dim=1).cpu().tolist()
            t3 = time.perf_counter()
            total_pre += (t1 - t0) * 1000
            total_inf += (t2 - t1) * 1000
            total_post += (t3 - t2) * 1000
            all_preds.extend(preds)
            all_targets.extend(labels.tolist())

    n = max(len(all_preds), 1)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    return {
        "top1": accuracy_score(all_targets, all_preds),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "speed_preprocess_ms": total_pre / n,
        "speed_inference_ms": total_inf / n,
        "speed_postprocess_ms": total_post / n,
    }


def compute_cls_val_loss(
    model: nn.Module,
    dataset_dir: Path,
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
) -> float:
    """Compute cross-entropy loss on the validation split."""
    transform = T.Compose(
        [
            T.Resize(imgsz + 32),
            T.CenterCrop(imgsz),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    dataset = ImageFolder(str(dataset_dir / "val"), transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device == "cuda"),
    )
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            loss = criterion(model(images).float(), labels)
            total += loss.item()
    return total / max(len(loader), 1)


def eval_frcnn_split(
    model: nn.Module,
    dataset_dir: Path,
    split: str,
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
) -> Dict[str, float]:
    """Evaluate Faster R-CNN on one split."""
    dataset = YoloDetectionDataset(dataset_dir / split, imgsz=imgsz, train=False)
    loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        collate_fn=_frcnn_collate,
        pin_memory=(device == "cuda"),
    )

    all_preds: List[Dict[str, torch.Tensor]] = []
    all_targets: List[Dict[str, torch.Tensor]] = []
    total_pre = total_inf = total_post = 0.0
    n_images = 0

    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            t0 = time.perf_counter()
            images_dev = [img.to(device, non_blocking=True) for img in images]
            t1 = time.perf_counter()
            predictions = model(images_dev)
            if device == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            all_preds.extend({k: v.cpu() for k, v in p.items()} for p in predictions)
            all_targets.extend({k: v.cpu() for k, v in t.items()} for t in targets)
            t3 = time.perf_counter()
            total_pre += (t1 - t0) * 1000
            total_inf += (t2 - t1) * 1000
            total_post += (t3 - t2) * 1000
            n_images += len(images)

    metrics = _compute_detect_metrics(all_preds, all_targets)
    n = max(n_images, 1)
    metrics["speed_preprocess_ms"] = total_pre / n
    metrics["speed_inference_ms"] = total_inf / n
    metrics["speed_postprocess_ms"] = total_post / n
    return metrics


def compute_frcnn_val_loss(
    model: nn.Module,
    dataset_dir: Path,
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
) -> float:
    """Compute Faster R-CNN loss on the validation split (train mode, no_grad)."""
    dataset = YoloDetectionDataset(dataset_dir / "val", imgsz=imgsz, train=False)
    loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        collate_fn=_frcnn_collate,
        pin_memory=(device == "cuda"),
    )
    model.train()
    total = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            total += sum(loss_dict.values()).item()
    model.eval()
    return total / max(len(loader), 1)


def eval_detr_split(
    model: nn.Module,
    processor: "DetrImageProcessor",
    dataset_dir: Path,
    split: str,
    batch: int,
    workers: int,
    device: str,
) -> Dict[str, float]:
    """Evaluate DETR on one split."""
    dataset = DetrDetectionDataset(dataset_dir / split)
    loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        collate_fn=_detr_collate,
    )

    all_preds: List[Dict[str, torch.Tensor]] = []
    all_targets: List[Dict[str, torch.Tensor]] = []
    total_pre = total_inf = total_post = 0.0
    n_images = 0

    model.eval()
    with torch.no_grad():
        for pil_images, labels, orig_sizes in loader:
            t0 = time.perf_counter()
            enc = processor(images=pil_images, return_tensors="pt")
            pixel_values = enc["pixel_values"].to(device, non_blocking=True)
            pixel_mask = enc["pixel_mask"].to(device, non_blocking=True)
            t1 = time.perf_counter()
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            if device == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            results = processor.post_process_object_detection(
                outputs, threshold=0.0, target_sizes=list(orig_sizes)
            )
            for r, lbl, (orig_h, orig_w) in zip(results, labels, orig_sizes):
                all_preds.append(
                    {
                        "boxes": r["boxes"].cpu(),
                        "labels": (r["labels"] + 1).cpu(),
                        "scores": r["scores"].cpu(),
                    }
                )
                gt_boxes = lbl["boxes"].clone()
                if gt_boxes.numel() > 0:
                    cx = gt_boxes[:, 0]
                    cy = gt_boxes[:, 1]
                    bw = gt_boxes[:, 2]
                    bh = gt_boxes[:, 3]
                    gt_boxes = torch.stack(
                        [
                            (cx - bw / 2) * orig_w,
                            (cy - bh / 2) * orig_h,
                            (cx + bw / 2) * orig_w,
                            (cy + bh / 2) * orig_h,
                        ],
                        dim=1,
                    )
                all_targets.append(
                    {
                        "boxes": gt_boxes.cpu(),
                        "labels": (lbl["class_labels"] + 1).cpu(),
                    }
                )
            t3 = time.perf_counter()
            total_pre += (t1 - t0) * 1000
            total_inf += (t2 - t1) * 1000
            total_post += (t3 - t2) * 1000
            n_images += len(pil_images)

    metrics = _compute_detect_metrics(all_preds, all_targets)
    n = max(n_images, 1)
    metrics["speed_preprocess_ms"] = total_pre / n
    metrics["speed_inference_ms"] = total_inf / n
    metrics["speed_postprocess_ms"] = total_post / n
    return metrics


def compute_detr_val_loss(
    model: nn.Module,
    processor: "DetrImageProcessor",
    dataset_dir: Path,
    batch: int,
    workers: int,
    device: str,
) -> float:
    """Compute DETR loss on the validation split (train mode, no_grad)."""
    dataset = DetrDetectionDataset(dataset_dir / "val")
    loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        collate_fn=_detr_collate,
    )
    model.train()
    total = 0.0
    with torch.no_grad():
        for pil_images, labels, _ in loader:
            enc = processor(images=pil_images, return_tensors="pt")
            pixel_values = enc["pixel_values"].to(device, non_blocking=True)
            pixel_mask = enc["pixel_mask"].to(device, non_blocking=True)
            batch_labels = [{k: v.to(device) for k, v in lbl.items()} for lbl in labels]
            out = model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                labels=batch_labels,
            )
            total += out.loss.item()
    model.eval()
    return total / max(len(loader), 1)


# Results logging


def log_results_to_file(
    model_name: str, metrics: Dict[str, Any], train_time: float, mode: str
) -> None:
    """Append a row to the shared results.csv."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row: Dict[str, Any] = {
        "model_name": model_name,
        "mode": mode,
        "seed": args.seed,
        "timestamp": timestamp,
        "train_time": train_time,
        "inference_time": metrics.get("val_speed_inference_ms", 0.0),
    }
    cls_keys = ["top1", "precision", "recall", "f1"]
    for split in ["train", "val", "test"]:
        for key in cls_keys:
            row[f"{split}_{key}"] = (
                metrics.get(f"{split}_{key}", np.nan) if mode == "cls" else np.nan
            )
    det_keys = ["map50", "map50_95", "mean_iou", "box_mp", "box_mr", "box_f1"]
    for split in ["train", "val", "test"]:
        for key in det_keys:
            row[f"{split}_{key}"] = (
                metrics.get(f"{split}_{key}", np.nan) if mode == "detect" else np.nan
            )
    speed_keys = ["speed_preprocess_ms", "speed_inference_ms", "speed_postprocess_ms"]
    for split in ["train", "val", "test"]:
        for key in speed_keys:
            row[f"{split}_{key}"] = metrics.get(f"{split}_{key}", np.nan)

    if RESULTS_CSV.exists():
        try:
            df = pd.read_csv(RESULTS_CSV)
        except Exception as exc:
            logger.error(f"Failed to read results.csv: {exc}")
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    try:
        df.to_csv(RESULTS_CSV, index=False)
        logger.info(f"Results saved to {RESULTS_CSV}")
    except Exception as exc:
        logger.error(f"Failed to save results.csv: {exc}")


# Top-level training runners


def run_cls(ts: str) -> Tuple[float, Dict[str, Any]]:
    """Full classification training + evaluation pipeline."""
    hp = get_torch_hyperparams("cls", MODEL_NAME)
    if args.epoch is not None:
        hp["epochs"] = args.epoch
        logger.info(f"Overriding epochs to {args.epoch}")
    batch = apply_batch_mult(hp["batch"], args.batch_mult)
    logger.info(f"Batch size: {hp['batch']} * {args.batch_mult} -> {batch}")

    imgsz: int = hp["imgsz"]
    epochs: int = hp["epochs"]
    patience: int = hp["patience"]
    workers: int = hp["workers"]

    train_transform = T.Compose(
        [
            T.Resize(imgsz + 32),
            T.RandomCrop(imgsz),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    train_dataset = ImageFolder(
        str(DATASET_CLS_AUGMENTED_DIR / "train"), transform=train_transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=(DEVICE == "cuda"),
    )

    logger.info(f"Building classification model: {MODEL_NAME}")
    model = build_cls_model(MODEL_NAME, num_classes=NUM_CLS_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=hp.get("label_smoothing", 0.0))
    optimizer = optim.AdamW(
        model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(enabled=hp["amp"] and DEVICE == "cuda")

    weights_dir = RESULTS_DIR / f"{ts}-train" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = weights_dir / "best.pt"

    best_val_acc = -1.0
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs", unit="epoch"):
        train_loss = train_cls_epoch(
            model, train_loader, optimizer, scaler, criterion, DEVICE, epoch, epochs
        )
        scheduler.step()
        val_loss = compute_cls_val_loss(
            model, DATASET_CLS_AUGMENTED_DIR, imgsz, batch, workers, DEVICE
        )
        val_metrics = eval_cls_split(
            model, DATASET_CLS_AUGMENTED_DIR, "val", imgsz, batch, workers, DEVICE
        )
        val_acc = val_metrics["top1"]
        logger.info(
            f"Epoch {epoch}/{epochs}  train_loss={train_loss:.4f}"
            f"  val_loss={val_loss:.4f}"
            f"  val_acc={val_acc:.4f}  val_f1={val_metrics['f1']:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_ckpt)
            logger.info(f"  Saved best model (val_acc={val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.2f}s")
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))

    all_metrics: Dict[str, Any] = {}
    for split in ["val", "test"]:
        split_metrics = eval_cls_split(
            model, DATASET_CLS_AUGMENTED_DIR, split, imgsz, batch, workers, DEVICE
        )
        logger.info(
            f"[{split}] top1={split_metrics['top1']:.4f}"
            f"  prec={split_metrics['precision']:.4f}"
            f"  rec={split_metrics['recall']:.4f}"
            f"  f1={split_metrics['f1']:.4f}"
        )
        for key, val in split_metrics.items():
            all_metrics[f"{split}_{key}"] = val

    del model
    cleanup_cuda()
    return train_time, all_metrics


def run_frcnn(ts: str) -> Tuple[float, Dict[str, Any]]:
    """Full Faster R-CNN training + evaluation pipeline."""
    hp = get_torch_hyperparams("detect")
    if args.epoch is not None:
        hp["epochs"] = args.epoch
        logger.info(f"Overriding epochs to {args.epoch}")
    batch = apply_batch_mult(hp["batch"], args.batch_mult)
    logger.info(f"Batch size: {hp['batch']} * {args.batch_mult} -> {batch}")

    imgsz: int = hp["imgsz"]
    epochs: int = hp["epochs"]
    patience: int = hp["patience"]
    workers: int = hp["workers"]

    train_dataset = YoloDetectionDataset(
        DATASET_DETECT_AUGMENTED_DIR / "train", imgsz=imgsz, train=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        collate_fn=_frcnn_collate,
        pin_memory=(DEVICE == "cuda"),
    )

    logger.info(f"Building Faster R-CNN model: {MODEL_NAME}")
    model = build_frcnn_model(MODEL_NAME, num_classes=NUM_DETECT_CLASSES)
    model = model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=hp["lr"], weight_decay=hp["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(enabled=hp["amp"] and DEVICE == "cuda")

    weights_dir = RESULTS_DIR / f"{ts}-train" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = weights_dir / "best.pt"

    best_val_map50 = -1.0
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs", unit="epoch"):
        train_loss = train_frcnn_epoch(
            model, train_loader, optimizer, scaler, DEVICE, epoch, epochs
        )
        scheduler.step()
        val_loss = compute_frcnn_val_loss(
            model, DATASET_DETECT_AUGMENTED_DIR, imgsz, batch, workers, DEVICE
        )
        val_metrics = eval_frcnn_split(
            model, DATASET_DETECT_AUGMENTED_DIR, "val", imgsz, batch, workers, DEVICE
        )
        val_map50 = val_metrics["map50"]
        logger.info(
            f"Epoch {epoch}/{epochs}  train_loss={train_loss:.4f}"
            f"  val_loss={val_loss:.4f}  val_mAP50={val_map50:.4f}"
        )
        if val_map50 > best_val_map50:
            best_val_map50 = val_map50
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_ckpt)
            logger.info(f"  Saved best model (val_mAP50={val_map50:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.2f}s")
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))

    all_metrics: Dict[str, Any] = {}
    for split in ["val", "test"]:
        split_metrics = eval_frcnn_split(
            model, DATASET_DETECT_AUGMENTED_DIR, split, imgsz, batch, workers, DEVICE
        )
        logger.info(
            f"[{split}] mAP50={split_metrics['map50']:.4f}"
            f"  mAP50-95={split_metrics['map50_95']:.4f}"
            f"  box_f1={split_metrics['box_f1']:.4f}"
        )
        for key, val in split_metrics.items():
            all_metrics[f"{split}_{key}"] = val

    del model
    cleanup_cuda()
    return train_time, all_metrics


def run_detr(ts: str) -> Tuple[float, Dict[str, Any]]:
    """Full DETR training + evaluation pipeline."""
    hp = get_torch_hyperparams("detect")
    if args.epoch is not None:
        hp["epochs"] = args.epoch
        logger.info(f"Overriding epochs to {args.epoch}")
    batch = apply_batch_mult(hp["batch"], args.batch_mult)
    logger.info(f"Batch size: {hp['batch']} * {args.batch_mult} -> {batch}")

    epochs: int = hp["epochs"]
    patience: int = hp["patience"]
    workers: int = hp["workers"]

    logger.info(f"Building DETR model: {MODEL_NAME}")
    model, processor = build_detr_model(MODEL_NAME, num_labels=NUM_DETR_LABELS)
    model = model.to(DEVICE)

    train_dataset = DetrDetectionDataset(DATASET_DETECT_AUGMENTED_DIR / "train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        collate_fn=_detr_collate,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=hp["lr"], weight_decay=hp["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(enabled=hp["amp"] and DEVICE == "cuda")

    weights_dir = RESULTS_DIR / f"{ts}-train" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = weights_dir / "best.pt"

    best_val_map50 = -1.0
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs", unit="epoch"):
        train_loss = train_detr_epoch(
            model, processor, train_loader, optimizer, scaler, DEVICE, epoch, epochs
        )
        scheduler.step()
        val_loss = compute_detr_val_loss(
            model, processor, DATASET_DETECT_AUGMENTED_DIR, batch, workers, DEVICE
        )
        val_metrics = eval_detr_split(
            model,
            processor,
            DATASET_DETECT_AUGMENTED_DIR,
            "val",
            batch,
            workers,
            DEVICE,
        )
        val_map50 = val_metrics["map50"]
        logger.info(
            f"Epoch {epoch}/{epochs}  train_loss={train_loss:.4f}"
            f"  val_loss={val_loss:.4f}  val_mAP50={val_map50:.4f}"
        )
        if val_map50 > best_val_map50:
            best_val_map50 = val_map50
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_ckpt)
            logger.info(f"  Saved best model (val_mAP50={val_map50:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.2f}s")
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))

    all_metrics: Dict[str, Any] = {}
    for split in ["val", "test"]:
        split_metrics = eval_detr_split(
            model,
            processor,
            DATASET_DETECT_AUGMENTED_DIR,
            split,
            batch,
            workers,
            DEVICE,
        )
        logger.info(
            f"[{split}] mAP50={split_metrics['map50']:.4f}"
            f"  mAP50-95={split_metrics['map50_95']:.4f}"
            f"  box_f1={split_metrics['box_f1']:.4f}"
        )
        for key, val in split_metrics.items():
            all_metrics[f"{split}_{key}"] = val

    del model
    cleanup_cuda()
    return train_time, all_metrics


# Main


def main() -> None:
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = RESULTS_DIR / f"{ts}-run.log"
    setup_logging(log_file)
    set_seed(args.seed)

    if torch.cuda.is_available():
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Using CPU")

    logger.info(
        "Starting %s training: model=%s  seed=%d",
        args.mode,
        MODEL_NAME,
        args.seed,
    )

    valid_models = CLS_MODELS if args.mode == "cls" else DETECT_MODELS
    if MODEL_NAME not in valid_models:
        raise ValueError(
            f"Model '{MODEL_NAME}' not recognised for mode '{args.mode}'.\n"
            f"  Classification: {sorted(CLS_MODELS)}\n"
            f"  Detection:      {sorted(DETECT_MODELS)}"
        )

    cleanup_cuda()

    if args.mode == "cls":
        train_time, all_metrics = run_cls(ts)
    elif MODEL_NAME in FRCNN_MODELS:
        train_time, all_metrics = run_frcnn(ts)
    else:
        train_time, all_metrics = run_detr(ts)

    log_results_to_file(MODEL_NAME, all_metrics, train_time, args.mode)
    logger.info("TRAINING COMPLETED")
    logger.info(f"Model: {MODEL_NAME} ({args.mode}) - Seed: {args.seed}")
    logger.info("Training completed in %.2f seconds", train_time)
    cleanup_cuda()


if __name__ == "__main__":
    main()
