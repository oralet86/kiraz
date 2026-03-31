"""
Centralized Path Configuration for Kiraz Project

This module centralizes all path constants used across the project
to make it easier to modify directory structures and paths.

Usage:
    from paths import *
    # Access any path constant directly
"""

from pathlib import Path

# CORE DIRECTORIES

# Results directories
RESULTS_DIR = Path("results")
RESULTS_HPO_CLS_DIR = Path("results_hpo_cls")
RESULTS_HPO_DETECT_DIR = Path("results_hpo_detect")

# New pipeline dataset directories
DATASET_ORIGINAL_DIR = Path("datasets/original")
DATASET_NEGATIVES_DIR = DATASET_ORIGINAL_DIR / "negatives"
DATASET_DETECT_REMAPPED_DIR = Path("datasets/data_detect_remapped")
DATASET_CLS_REMAPPED_DIR = Path("datasets/data_cls_remapped")
DATASET_CLS_CLIPPED_DIR = Path("datasets/data_cls_clipped")
DATASET_DETECT_STRATIFIED_DIR = Path("datasets/data_detect_stratified")
DATASET_CLS_STRATIFIED_DIR = Path("datasets/data_cls_stratified")
DATASET_DETECT_AUGMENTED_DIR = Path("datasets/data_detect_augmented")
DATASET_CLS_AUGMENTED_DIR = Path("datasets/data_cls_augmented")
DATASET_DETECT_CHREDUCED_DIR = Path("datasets/data_detect_chreduced")
DATASET_CLS_CHREDUCED_DIR = Path("datasets/data_cls_chreduced")
DATASET_DETECT_CHROMATIC_DIR = Path("datasets/data_detect_chromatic")
DATASET_CLS_CHROMATIC_DIR = Path("datasets/data_cls_chromatic")

# Logs directory
LOGS_DIR = Path("logs")

# Python package for model builders and storage for pretrained weights (.pt)
MODELS_DIR = Path("models")

# FILE PATHS

# Results CSV files
RESULTS_CSV = RESULTS_DIR / Path("results.csv")
RESULTS_HPO_CLS_CSV = Path("results_hpo_cls.csv")
RESULTS_HPO_DETECT_CSV = Path("results_hpo_detect.csv")

# Log files
UNIVERSAL_LOG_FILE = LOGS_DIR / "logs.log"

# TEMPORARY/UTILITY PATHS

# Temporary directory for ArUco markers
TEMP_MARKERS_DIR = Path("temp_markers")

# DATASET STRUCTURE PATHS

# Standard dataset splits
DATASET_SPLITS = ["train", "val", "test"]

# Dataset subdirectories (relative to dataset roots)
IMAGES_SUBDIR = "images"
LABELS_SUBDIR = "labels"

# PATH GENERATION FUNCTIONS


def get_dataset_paths(dataset_root: Path, splits: list | None = None) -> dict:
    """
    Generate standardized dataset structure paths.

    Args:
        dataset_root: Root directory of the dataset
        splits: List of splits (default: ["train", "val", "test"])

    Returns:
        Dictionary with paths for images and labels directories
    """
    if splits is None:
        splits = DATASET_SPLITS

    paths = {}
    for split in splits:
        paths[f"{split}_images"] = dataset_root / split / IMAGES_SUBDIR
        paths[f"{split}_labels"] = dataset_root / split / LABELS_SUBDIR

    return paths


def get_results_paths(mode: str, timestamp: str) -> dict:
    """
    Generate paths for training results based on mode and timestamp.

    Args:
        mode: Training mode ("cls" or "detect")
        timestamp: Training timestamp string

    Returns:
        Dictionary with result paths
    """
    base_dir = (
        RESULTS_DIR
        if mode in ["cls", "detect"]
        else RESULTS_HPO_CLS_DIR
        if mode == "hpo_cls"
        else RESULTS_HPO_DETECT_DIR
    )

    return {
        "base": base_dir,
        "train": base_dir / f"{timestamp}-train",
        "log": base_dir / f"{timestamp}-run.log",
        "weights": base_dir / f"{timestamp}-train" / "weights" / "best.pt",
    }


def ensure_directories() -> None:
    """
    Ensure all core directories exist.
    """
    directories = [
        RESULTS_DIR,
        RESULTS_HPO_CLS_DIR,
        RESULTS_HPO_DETECT_DIR,
        LOGS_DIR,
        MODELS_DIR,
        DATASET_DETECT_REMAPPED_DIR,
        DATASET_CLS_REMAPPED_DIR,
        DATASET_CLS_CLIPPED_DIR,
        DATASET_DETECT_STRATIFIED_DIR,
        DATASET_CLS_STRATIFIED_DIR,
        DATASET_DETECT_AUGMENTED_DIR,
        DATASET_CLS_AUGMENTED_DIR,
        DATASET_DETECT_CHREDUCED_DIR,
        DATASET_CLS_CHREDUCED_DIR,
        DATASET_DETECT_CHROMATIC_DIR,
        DATASET_CLS_CHROMATIC_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Print all defined paths for verification
    print("Kiraz Project Paths Configuration")

    print("\nCore Directories:")
    print(f"  RESULTS_DIR: {RESULTS_DIR}")
    print(f"  RESULTS_HPO_CLS_DIR: {RESULTS_HPO_CLS_DIR}")
    print(f"  RESULTS_HPO_DETECT_DIR: {RESULTS_HPO_DETECT_DIR}")
    print(f"  LOGS_DIR: {LOGS_DIR}")
    print(f"  MODELS_DIR: {MODELS_DIR}")

    print(f"  RESULTS_CSV: {RESULTS_CSV}")
    print(f"  UNIVERSAL_LOG_FILE: {UNIVERSAL_LOG_FILE}")

    print("\nAll paths configured successfully!")
