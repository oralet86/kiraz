from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple
import gc
import time

import optuna
import torch
import numpy as np
from ultralytics import YOLO

from hyperparams import (
    get_training_config,
    get_default_cls_hyperparams,
    get_default_detect_hyperparams,
)
from log import logger, add_log_file
from paths import (
    RESULTS_HPO_CLS_DIR,
    RESULTS_HPO_DETECT_DIR,
    DATASET_CLS_AUGMENTED_DIR,
    DATASET_DETECT_AUGMENTED_DIR,
    MODELS_DIR,
)

parser = argparse.ArgumentParser(
    description="Optuna-based hyperparameter optimization for YOLO models"
)
parser.add_argument(
    "--mode",
    type=str,
    choices=["cls", "detect"],
    required=True,
    help="Training mode: cls or detect",
)
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Base YOLO model (e.g. yolo11l-cls.pt, yolo11m.pt)",
)
parser.add_argument(
    "--iterations",
    type=int,
    default=20,
    help="Number of Optuna trials (default: 20)",
)
parser.add_argument(
    "--trial-epochs",
    type=int,
    default=100,
    help="Training epochs per trial (default: 30)",
)
parser.add_argument(
    "--batch",
    type=int,
    default=None,
    help="Batch size override (uses hyperparams.py default otherwise)",
)

args = parser.parse_args()

MODE: str = args.mode
MODEL_NAME: str = args.model
N_TRIALS: int = args.iterations
TRIAL_EPOCHS: int = args.trial_epochs

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"  # type: ignore

HPO_OUT_DIR: Path = RESULTS_HPO_CLS_DIR if MODE == "cls" else RESULTS_HPO_DETECT_DIR
DATASET_DIR: Path = (
    DATASET_CLS_AUGMENTED_DIR if MODE == "cls" else DATASET_DETECT_AUGMENTED_DIR
)


def ensure_dataset_layout() -> None:
    if MODE == "cls":
        required = [DATASET_DIR / split for split in ["train", "val", "test"]]
        missing_msg = "Classification dataset missing required split folders"
    else:
        required = [
            DATASET_DIR / split / sub
            for split in ["train", "val", "test"]
            for sub in ["images", "labels"]
        ]
        missing_msg = "Detection dataset missing required paths"

    missing = [p for p in required if not p.exists()]
    if missing:
        details = "\n".join(f" - {p}" for p in missing)
        raise FileNotFoundError(f"{missing_msg}:\n{details}")


def ensure_data_yaml() -> None:
    yaml_path = DATASET_DIR / "data.yaml"
    if yaml_path.exists():
        logger.info(f"Found existing data.yaml: {yaml_path}")
        return

    yaml_text = (
        f"path: {DATASET_DIR.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "\n"
        "nc: 2\n"
        "names:\n"
        "  0: cherry\n"
        "  1: stem\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    logger.info(f"Created data.yaml at: {yaml_path}")


def _suggest_cls_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "lr0": trial.suggest_float("lr0", 5e-5, 5e-4, log=True),
        "lrf": trial.suggest_float("lrf", 5e-4, 5e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.3, 0.65),
        "momentum": trial.suggest_float("momentum", 0.88, 0.97),
        "weight_decay": trial.suggest_float("weight_decay", 2e-4, 1e-3, log=True),
    }


def _suggest_detect_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "lr0": trial.suggest_float("lr0", 1e-4, 5e-4, log=True),
        "lrf": trial.suggest_float("lrf", 5e-4, 5e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.3, 0.65),
        "momentum": trial.suggest_float("momentum", 0.88, 0.97),
        "weight_decay": trial.suggest_float("weight_decay", 2e-4, 1e-3, log=True),
        "box": trial.suggest_float("box", 12.0, 18.0),
        "cls": trial.suggest_float("cls", 1.0, 2.5),
        "dfl": trial.suggest_float("dfl", 1.5, 2.5),
    }


def _build_trial_params(suggested: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = (
        get_default_cls_hyperparams()
        if MODE == "cls"
        else get_default_detect_hyperparams()
    )
    params.update(get_training_config(MODE))
    params["epochs"] = TRIAL_EPOCHS
    params["patience"] = max(5, TRIAL_EPOCHS // 5)
    if args.batch is not None:
        params["batch"] = args.batch
    params.update(suggested)
    return params


def objective(trial: optuna.Trial) -> float:
    suggested = (
        _suggest_cls_params(trial) if MODE == "cls" else _suggest_detect_params(trial)
    )
    params = _build_trial_params(suggested)

    model_path = MODELS_DIR / MODEL_NAME
    resolved_model = str(model_path) if model_path.exists() else MODEL_NAME
    task = "classify" if MODE == "cls" else "detect"
    model = YOLO(resolved_model, task=task)

    data = (
        str(DATASET_DIR.resolve()) if MODE == "cls" else str(DATASET_DIR / "data.yaml")
    )

    results = model.train(
        data=data,
        device=DEVICE,
        project=str(HPO_OUT_DIR / "trials"),
        name=f"trial-{trial.number}",
        exist_ok=True,
        verbose=False,
        plots=False,
        **params,
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if results is None:
        raise optuna.TrialPruned()

    if MODE == "cls":
        return float(getattr(results, "top1", 0.0))
    else:
        box = getattr(results, "box", None)
        return float(getattr(box, "map50", 0.0)) if box is not None else 0.0


def _extract_metrics(results_obj: Any) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}

    if MODE == "cls":
        for k in ("top1", "top5"):
            v = getattr(results_obj, k, None)
            if v is not None:
                metrics[k] = float(v)
    else:
        box = getattr(results_obj, "box", None)
        if box is not None:
            for k in ("map", "map50", "map75", "mp", "mr", "f1"):
                v = getattr(box, k, None)
                if v is None:
                    continue
                if isinstance(v, (int, float, np.generic)):
                    metrics[f"box_{k}"] = float(v)
                elif isinstance(v, (np.ndarray, list, tuple)) and len(v) > 0:
                    metrics[f"box_{k}"] = float(np.mean(v))

    speed = getattr(results_obj, "speed", None)
    if isinstance(speed, dict):
        metrics["speed_preprocess_ms"] = float(speed.get("preprocess", 0))
        metrics["speed_inference_ms"] = float(speed.get("inference", 0))
        metrics["speed_postprocess_ms"] = float(speed.get("postprocess", 0))

    return metrics


def train_final_model(best_params: Dict[str, Any], ts: str) -> Tuple[Path, float]:
    logger.info("Training final model with best hyperparameters...")

    params: Dict[str, Any] = (
        get_default_cls_hyperparams()
        if MODE == "cls"
        else get_default_detect_hyperparams()
    )
    params.update(get_training_config(MODE))
    if args.batch is not None:
        params["batch"] = args.batch
    params.update(best_params)

    model_path = MODELS_DIR / MODEL_NAME
    resolved_model = str(model_path) if model_path.exists() else MODEL_NAME
    task = "classify" if MODE == "cls" else "detect"
    model = YOLO(resolved_model, task=task)

    data = (
        str(DATASET_DIR.resolve()) if MODE == "cls" else str(DATASET_DIR / "data.yaml")
    )

    start_time = time.time()
    results = model.train(
        data=data,
        device=DEVICE,
        project=str(HPO_OUT_DIR.resolve()),
        name=f"{ts}-best",
        exist_ok=True,
        plots=True,
        save=True,
        **params,
    )

    if results is None:
        raise RuntimeError("Final model training did not return a results object")

    best_path = Path(results.save_dir) / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(
            f"Training finished but best.pt not found at {best_path}"
        )

    YOLO(best_path).export(format="onnx", opset=20)

    train_time = time.time() - start_time
    logger.info(f"Final model training complete. Best weights: {best_path}")
    return best_path, train_time


def evaluate_on_splits(best_weights: Path, ts: str) -> Dict[str, Any]:
    model = YOLO(best_weights)
    all_metrics: Dict[str, Any] = {}

    data = (
        str(DATASET_DIR.resolve()) if MODE == "cls" else str(DATASET_DIR / "data.yaml")
    )

    for split in ["train", "val", "test"]:
        logger.info(f"Evaluating on {split} split...")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        split_results = model.val(
            data=data,
            split=split,
            device=DEVICE,
            project=str(HPO_OUT_DIR),
            name=f"{ts}-{split}",
            exist_ok=True,
        )

        split_metrics = _extract_metrics(split_results)
        for key, value in split_metrics.items():
            all_metrics[f"{split}_{key}"] = value

    logger.info("Evaluation metrics:")
    for key in sorted(all_metrics):
        val = all_metrics[key]
        if isinstance(val, float):
            logger.info("  %-30s %.4f", f"{key}:", val)
        else:
            logger.info("  %-30s %s", f"{key}:", val)

    return all_metrics


def save_trial_results(study: optuna.Study, ts: str) -> None:
    HPO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    trials_csv = HPO_OUT_DIR / f"{ts}-trials.csv"

    df = study.trials_dataframe()
    df.to_csv(trials_csv, index=False)
    logger.info(f"All trial results saved to {trials_csv}")

    best_csv = HPO_OUT_DIR / f"{ts}-best-params.csv"
    row: Dict[str, Any] = {
        "model": MODEL_NAME,
        "mode": MODE,
        "best_value": study.best_value,
        "timestamp": ts,
    }
    row.update(study.best_params)

    with open(best_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    logger.info(f"Best params saved to {best_csv}")


def main() -> None:
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    HPO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = HPO_OUT_DIR / f"{ts}-hpo.log"
    add_log_file(log_file)

    logger.info(
        "Starting Optuna HPO | mode=%s | model=%s | trials=%d | trial_epochs=%d",
        MODE,
        MODEL_NAME,
        N_TRIALS,
        TRIAL_EPOCHS,
    )

    if torch.cuda.is_available():
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Using CPU")

    ensure_dataset_layout()
    if MODE == "detect":
        ensure_data_yaml()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        study_name=f"{MODEL_NAME}-{MODE}-{ts}",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    try:
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    except KeyboardInterrupt:
        logger.warning("HPO interrupted by user.")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        logger.error("No trials completed successfully. Cannot proceed.")
        return

    best_params = study.best_params
    best_value = study.best_value

    logger.info("Best trial value: %.6f", best_value)
    logger.info("Best hyperparameters:")
    for k, v in best_params.items():
        logger.info("  %s: %s", k, v)

    save_trial_results(study, ts)

    best_weights, train_time = train_final_model(best_params, ts)

    all_metrics = evaluate_on_splits(best_weights, ts)

    logger.info("HPO and final training completed successfully.")
    logger.info(f"Results saved in: {HPO_OUT_DIR}")

    primary_key = "val_top1" if MODE == "cls" else "val_box_map50"
    primary_val = all_metrics.get(primary_key)
    if primary_val is not None:
        logger.info(f"Final {primary_key}: {primary_val:.4f}")


if __name__ == "__main__":
    main()
