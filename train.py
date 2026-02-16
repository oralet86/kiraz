from __future__ import annotations

from ultralytics import YOLO
from pathlib import Path
import logging
from datetime import datetime
from typing import Any, Dict
import torch
import numpy as np

LOG = True

RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

DATASET_DIR = Path("data_resized_split")
DATA_YAML = DATASET_DIR / "data.yaml"

BASE_MODEL_PATH = Path("models") / "yolo11s.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 200
PATIENCE = 20
IMGSZ = 768
BATCH = -1

# Torch CUDA settings
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"  # type: ignore


def setup_logging(log_file: Path) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO if LOG else logging.DEBUG)

    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO if LOG else logging.DEBUG)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO if LOG else logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)


def ensure_dataset_layout(dataset_dir: Path) -> None:
    required = [
        dataset_dir / "train" / "images",
        dataset_dir / "train" / "labels",
        dataset_dir / "val" / "images",
        dataset_dir / "val" / "labels",
        dataset_dir / "test" / "images",
        dataset_dir / "test" / "labels",
    ]

    missing = [p for p in required if not p.exists()]
    if missing:
        msg = "Dataset structure is missing required paths:\n" + "\n".join(
            f" - {p}" for p in missing
        )
        raise FileNotFoundError(msg)


def ensure_data_yaml(dataset_dir: Path, yaml_path: Path) -> None:
    if yaml_path.exists():
        logging.info("Found existing data.yaml: %s", yaml_path)
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
    logging.info("Created data.yaml at: %s", yaml_path)


def _extract_metrics(results_obj: Any) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}

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

    speed = getattr(results_obj, "speed", None)
    if isinstance(speed, dict):
        metrics["speed_preprocess_ms"] = float(speed.get("preprocess", 0))
        metrics["speed_inference_ms"] = float(speed.get("inference", 0))
        metrics["speed_postprocess_ms"] = float(speed.get("postprocess", 0))

    return metrics


def log_metrics(metrics: Dict[str, Any], prefix: str = "Test Metrics") -> None:
    if not metrics:
        logging.warning("%s: none found", prefix)
        return

    logging.info(prefix)

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
                logging.info("  %-24s %.4f", key + ":", val)
            else:
                logging.info("  %-24s %s", key + ":", val)

    # Log remaining metrics
    remaining = sorted(k for k in metrics if k not in preferred_order)
    for key in remaining:
        val = metrics[key]
        if isinstance(val, float):
            logging.info("  %-24s %.4f", key + ":", val)
        else:
            logging.info("  %-24s %s", key + ":", val)


def train_model(ts: str) -> Path:
    logging.info("Validating dataset layout under %s", DATASET_DIR)
    ensure_dataset_layout(DATASET_DIR)
    ensure_data_yaml(DATASET_DIR, DATA_YAML)

    logging.info("Loading base model: %s", BASE_MODEL_PATH)
    model = YOLO(BASE_MODEL_PATH)

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logging.info("Using CUDA device: %s", device_name)
    else:
        logging.info("Using CPU")

    logging.info("Starting training...")
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        cache=True,
        workers=8,
        amp=True,
        plots=True,
        patience=PATIENCE,
        project=str(RESULTS_DIR.resolve()),
        name=f"{ts}-train",
        save=True,
        exist_ok=True,
        deterministic=False,
        cos_lr=True,
        lr0=4e-4,
        lrf=0.01,
        optimizer="AdamW",
        mosaic=0.5,
        scale=0.5,
        translate=0.5,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.25,
        fliplr=0.5,
        perspective=5e-4,
        mixup=0.1,
    )

    if results is None:
        raise RuntimeError("Training did not return results object")

    best_path = Path(results.save_dir) / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(
            f"Training finished but best.pt not found at {best_path}"
        )

    best_model = YOLO(best_path)
    best_model.export(format="onnx")

    logging.info("Training complete. Best weights: %s", best_path)
    return best_path


def evaluate_on_test(best_weights: Path, ts: str) -> Dict[str, Any]:
    logging.info("Evaluating on test split...")
    model = YOLO(best_weights)

    test_results = model.val(
        data=str(DATA_YAML),
        split="test",
        device=DEVICE,
        imgsz=IMGSZ,
        project=str(RESULTS_DIR),
        name=f"{ts}-test",
        exist_ok=True,
    )

    metrics = _extract_metrics(test_results)
    log_metrics(metrics)
    return metrics


def main() -> None:
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    log_file = RESULTS_DIR / f"{ts}-run.log"
    setup_logging(log_file)
    logging.info("Log file: %s", log_file)

    best_weights = train_model(ts)
    _ = evaluate_on_test(best_weights, ts)


if __name__ == "__main__":
    main()
