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
RESULTS_CLS_DIR = Path("results_cls")
RESULTS_DETECT_DIR = Path("results_detect")
RESULTS_HPO_CLS_DIR = Path("results_hpo_cls")
RESULTS_HPO_DETECT_DIR = Path("results_hpo_detect")

# Dataset directories
DATASET_CLS_DIR = Path("datasets/data_class")
DATASET_COMBINED_DIR = Path("datasets/data_combined")
DATASET_SPLIT_DIR = Path("datasets/data_split")
DATASET_CLIPPED_SPLIT_DIR = Path("datasets/data_clipped_split")
DATASET_CLS_ALT_DIR = Path("datasets/data_cls")

# Logs directory
LOGS_DIR = Path("logs")

# Models directory
MODELS_DIR = Path("models")

# FILE PATHS

# Data configuration files
DATA_YAML = DATASET_COMBINED_DIR / "data.yaml"

# Results CSV files
RESULTS_CSV = Path("results.csv")
RESULTS_CLS_CSV = Path("results_cls.csv")
RESULTS_DETECT_CSV = Path("results_detect.csv")
RESULTS_HPO_CLS_CSV = Path("results_hpo_cls.csv")
RESULTS_HPO_DETECT_CSV = Path("results_hpo_detect.csv")
HPO_RESULTS_CLS_CSV = Path("hpo_results_cls.csv")
HPO_RESULTS_DETECT_CSV = Path("hpo_results_detect.csv")

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


def get_dataset_paths(dataset_root: Path, splits: list = None) -> dict:
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
        RESULTS_CLS_DIR, 
        RESULTS_DETECT_DIR,
        RESULTS_HPO_CLS_DIR,
        RESULTS_HPO_DETECT_DIR,
        LOGS_DIR,
        MODELS_DIR,
        DATASET_CLS_DIR,
        DATASET_COMBINED_DIR,
        DATASET_SPLIT_DIR,
        DATASET_CLIPPED_SPLIT_DIR,
        DATASET_CLS_ALT_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Print all defined paths for verification
    print("Kiraz Project Paths Configuration")

    print("\nCore Directories:")
    print(f"  RESULTS_DIR: {RESULTS_DIR}")
    print(f"  RESULTS_CLS_DIR: {RESULTS_CLS_DIR}")
    print(f"  RESULTS_DETECT_DIR: {RESULTS_DETECT_DIR}")
    print(f"  RESULTS_HPO_CLS_DIR: {RESULTS_HPO_CLS_DIR}")
    print(f"  RESULTS_HPO_DETECT_DIR: {RESULTS_HPO_DETECT_DIR}")
    print(f"  LOGS_DIR: {LOGS_DIR}")
    print(f"  MODELS_DIR: {MODELS_DIR}")

    print("\nDataset Directories:")
    print(f"  DATASET_CLS_DIR: {DATASET_CLS_DIR}")
    print(f"  DATASET_COMBINED_DIR: {DATASET_COMBINED_DIR}")
    print(f"  DATASET_SPLIT_DIR: {DATASET_SPLIT_DIR}")
    print(f"  DATASET_CLIPPED_SPLIT_DIR: {DATASET_CLIPPED_SPLIT_DIR}")
    print(f"  DATASET_CLS_ALT_DIR: {DATASET_CLS_ALT_DIR}")

    print("\nFile Paths:")
    print(f"  DATA_YAML: {DATA_YAML}")
    print(f"  RESULTS_CSV: {RESULTS_CSV}")
    print(f"  UNIVERSAL_LOG_FILE: {UNIVERSAL_LOG_FILE}")

    print("\nAll paths configured successfully!")
