"""
Dataset Utilities for Cherry Detection and Classification Projects

This module provides comprehensive utilities for:
1. Dataset resolution analysis and statistics
2. YOLO label remapping and manipulation
3. Detection to classification dataset conversion
4. Image cropping and clipping with label preservation
5. Singular object extraction from multi-object images
6. Dataset statistics and class distribution analysis
7. Dataset merging and stratified splitting
8. YAML configuration support
"""

from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import shutil
import yaml
import random
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Set
from log import logger
from data_augmentation import augment_detection_dataset, augment_classification_dataset
from paths import (
    DATASET_ORIGINAL_DIR,
    DATASET_DETECT_REMAPPED_DIR,
    DATASET_CLS_REMAPPED_DIR,
    DATASET_CLS_CLIPPED_DIR,
    DATASET_DETECT_STRATIFIED_DIR,
    DATASET_CLS_STRATIFIED_DIR,
    DATASET_DETECT_AUGMENTED_DIR,
    DATASET_CLS_AUGMENTED_DIR,
    DATASET_SPLITS,
)

# GLOBAL CONFIGURATION

# Supported image extensions
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]

# Default cropping parameters
DEFAULT_BUFFER_RATIO = 0.20  # 20% buffer around objects

# Dataset splitting parameters
DEFAULT_SPLIT_RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}
DEFAULT_RANDOM_SEED = 42

# YOLO label mappings
DEFAULT_LABEL_MAPPING = {0: 0, 1: 0, 2: 1}  # Combine classes 0 and 1, keep 2 separate
CLASSIFICATION_MAPPING = {
    0: "cherry",
    1: "cherry-imperfect",
}  # For classification conversion

# New pipeline-specific mappings
DETECTION_REMAPPED_MAPPING = {0: 0, 1: 0, 2: 1}  # cherry/cherry-imperfect->0, stem->1
CLS_REMAPPED_MAPPING = {0: 0, 1: 1}  # cherry->0, cherry-imperfect->1 (remove stem)


# RESOLUTION ANALYSIS UTILITIES


def collect_image_paths(
    dataset_dir: Path, splits: Optional[List[str]] = None
) -> List[Path]:
    """
    Collect all image paths from dataset splits.

    Args:
        dataset_dir: Root directory of the dataset
        splits: List of splits to process (default: ["train", "val", "test"])

    Returns:
        List of Path objects for all found images
    """
    if splits is None:
        splits = DATASET_SPLITS

    image_paths = []
    for split in splits:
        images_dir = dataset_dir / split / "images"
        if not images_dir.exists():
            logger.warning(f"Images directory not found: {images_dir}")
            continue

        for path in images_dir.rglob("*"):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(path)

    logger.info(f"Collected {len(image_paths)} images from {len(splits)} splits")
    return image_paths


def analyze_image_resolutions(
    image_paths: List[Path],
) -> Tuple[List[int], List[int], List[int], List[int], Counter]:
    """
    Analyze resolutions of collected images.

    Args:
        image_paths: List of image file paths

    Returns:
        Tuple containing:
        - List of widths
        - List of heights
        - List of short sides
        - List of long sides
        - Counter of exact resolutions (width, height): count
    """
    widths = []
    heights = []
    short_sides = []
    long_sides = []
    exact_resolutions = Counter()

    processed = 0
    for path in image_paths:
        try:
            img = cv2.imread(str(path))
            if img is None:
                logger.warning(f"Failed to read image: {path}")
                continue

            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
            short_sides.append(min(w, h))
            long_sides.append(max(w, h))
            exact_resolutions[(w, h)] += 1
            processed += 1

        except Exception as e:
            logger.error(f"Error processing {path}: {e}")

    logger.info(f"Analyzed {processed} images successfully")
    return widths, heights, short_sides, long_sides, exact_resolutions


def print_resolution_statistics(name: str, values: List[int]) -> None:
    """
    Print statistical summary of resolution values.

    Args:
        name: Name of the metric (e.g., "Width", "Height")
        values: List of numeric values
    """
    if not values:
        logger.warning(f"No values provided for {name}")
        return

    values_array = np.array(values)
    logger.info(f"{name}:")
    logger.info(f"  min:    {values_array.min():.1f}")
    logger.info(f"  max:    {values_array.max():.1f}")
    logger.info(f"  mean:   {values_array.mean():.1f}")
    logger.info(f"  median: {np.median(values_array):.1f}")
    logger.info(f"  std:    {values_array.std():.1f}")


def print_resolution_buckets(short_sides: List[int]) -> None:
    """
    Print distribution of images by short-side resolution buckets.

    Args:
        short_sides: List of short-side pixel values
    """
    buckets = {
        "<512": 0,
        "512\u2013639": 0,
        "640\u2013767": 0,
        "768\u20131023": 0,
        "1024+": 0,
    }

    for side in short_sides:
        if side < 512:
            buckets["<512"] += 1
        elif side < 640:
            buckets["512\u2013639"] += 1
        elif side < 768:
            buckets["640\u2013767"] += 1
        elif side < 1024:
            buckets["768\u20131023"] += 1
        else:
            buckets["1024+"] += 1

    logger.info("Short-side distribution:")
    total = len(short_sides)
    for bucket, count in buckets.items():
        percentage = count / total * 100 if total > 0 else 0
        logger.info(f"  {bucket:10s}: {count:4d} ({percentage:4.1f}%)")


def analyze_dataset_resolution(
    dataset_dir: Path, splits: Optional[List[str]] = None
) -> None:
    """
    Complete resolution analysis for a dataset.

    Args:
        dataset_dir: Root directory of the dataset
        splits: List of splits to analyze (default: ["train", "val", "test"])
    """
    logger.info(f"Starting resolution analysis for dataset: {dataset_dir}")

    image_paths = collect_image_paths(dataset_dir, splits)

    if not image_paths:
        logger.error("No images found in dataset")
        return

    logger.info(f"Total images to analyze: {len(image_paths)}")

    widths, heights, short_sides, long_sides, exact_resolutions = (
        analyze_image_resolutions(image_paths)
    )

    # Print statistics
    print_resolution_statistics("Width", widths)
    print_resolution_statistics("Height", heights)
    print_resolution_statistics("Short side", short_sides)
    print_resolution_statistics("Long side", long_sides)

    print_resolution_buckets(short_sides)

    # Top resolutions
    logger.info("Top 10 most common exact resolutions:")
    for (w, h), count in exact_resolutions.most_common(10):
        logger.info(f"  {w}x{h}: {count}")


# DATASET STATISTICS AND ANALYSIS UTILITIES


def load_class_names(dataset_path: Path) -> Optional[List[str]]:
    """
    Load class names from YOLO data.yaml file.

    Args:
        dataset_path: Root directory containing data.yaml

    Returns:
        List of class names or None if file not found
    """
    yaml_path = dataset_path / "data.yaml"
    if not yaml_path.exists():
        logger.warning(f"data.yaml not found at {yaml_path}")
        return None

    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        if "names" in data:
            if isinstance(data["names"], list):
                return data["names"]
            elif isinstance(data["names"], dict):
                return [data["names"][k] for k in sorted(data["names"].keys())]
    except Exception as e:
        logger.error(f"Error reading {yaml_path}: {e}")

    return None


def read_label_counts(label_path: Path) -> Counter:
    """
    Count objects per class in a YOLO label file.

    Args:
        label_path: Path to YOLO label file

    Returns:
        Counter with class_id -> count mapping
    """
    counts = Counter()
    if not label_path.exists():
        return counts

    try:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    counts[int(parts[0])] += 1
    except Exception as e:
        logger.error(f"Error reading {label_path}: {e}")

    return counts


def get_image_files(images_dir: Path) -> List[Path]:
    """
    Get all image files from directory using global extensions.

    Args:
        images_dir: Directory containing images

    Returns:
        List of image file paths
    """
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(images_dir.glob(f"*{ext}"))
    return images


def dataset_statistics(dataset_dir: Path, splits: Optional[List[str]] = None) -> None:
    """
    Print comprehensive dataset statistics including class distributions.

    Args:
        dataset_dir: Root directory of the dataset
        splits: List of splits to analyze (default: ["train", "val", "test"])
    """
    if splits is None:
        splits = DATASET_SPLITS

    class_names = load_class_names(dataset_dir)
    overall_class_counts = defaultdict(int)
    overall_images = 0
    overall_objects = 0

    logger.info("=" * 50)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 50)

    for split in splits:
        split_path = dataset_dir / split
        if not split_path.exists():
            logger.warning(f"Split directory not found: {split_path}")
            continue

        images_dir = split_path / "images"
        labels_dir = split_path / "labels"

        images = get_image_files(images_dir)
        label_files = list(labels_dir.glob("*.txt"))

        class_counts = defaultdict(int)
        total_objects = 0

        for label_file in label_files:
            counts = read_label_counts(label_file)
            for class_id, count in counts.items():
                class_counts[class_id] += count
                overall_class_counts[class_id] += count
                total_objects += count
                overall_objects += count

        overall_images += len(images)

        logger.info(f"[{split.upper()}]")
        logger.info(f"  Images: {len(images)}")
        logger.info(f"  Objects: {total_objects}")

        for class_id in sorted(class_counts):
            name = (
                class_names[class_id]
                if class_names and class_id < len(class_names)
                else f"class_{class_id}"
            )
            logger.info(f"    {name} ({class_id}): {class_counts[class_id]}")

    logger.info("=" * 30)
    logger.info("OVERALL TOTALS")
    logger.info("=" * 30)
    logger.info(f"Total images: {overall_images}")
    logger.info(f"Total objects: {overall_objects}")

    for class_id in sorted(overall_class_counts):
        name = (
            class_names[class_id]
            if class_names and class_id < len(class_names)
            else f"class_{class_id}"
        )
        logger.info(f"  {name} ({class_id}): {overall_class_counts[class_id]}")


# DATASET MERGING AND SPLITTING UTILITIES


def merge_dataset_splits(
    source_path: Path, output_path: Path, splits: Optional[List[str]] = None
) -> None:
    """
    Merge train/val/test splits into a single flat dataset.

    Args:
        source_path: Dataset with split directories
        output_path: Output directory for merged dataset
        splits: List of splits to merge (default: ["train", "val", "test"])
    """
    if splits is None:
        splits = DATASET_SPLITS

    logger.info(f"Merging dataset splits from {source_path} to {output_path}")

    # Create output directories
    (output_path / "images").mkdir(parents=True, exist_ok=True)
    (output_path / "labels").mkdir(parents=True, exist_ok=True)

    total_images = 0

    for split in splits:
        split_path = source_path / split
        if not split_path.exists():
            logger.warning(f"Split directory not found: {split_path}")
            continue

        images = get_image_files(split_path / "images")
        split_count = 0

        for img_path in images:
            filename = img_path.name
            stem, suffix = img_path.stem, img_path.suffix
            dest_img = output_path / "images" / filename

            # Handle filename conflicts
            counter = 1
            while dest_img.exists():
                dest_img = output_path / "images" / f"{stem}_{split}_{counter}{suffix}"
                counter += 1

            # Copy image
            shutil.copy(img_path, dest_img)

            # Copy corresponding label if exists
            label_src = split_path / "labels" / f"{stem}.txt"
            if label_src.exists():
                label_dest = output_path / "labels" / (dest_img.stem + ".txt")
                shutil.copy(label_src, label_dest)

            split_count += 1

        total_images += split_count
        logger.info(f"  {split}: {split_count} images merged")

    logger.info(f"Merge complete. Total images merged: {total_images}")


def stratified_dataset_split(
    data_dir: Path,
    output_dir: Path,
    split_ratios: Optional[Dict[str, float]] = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> None:
    """
    Perform object-aware stratified dataset splitting.

    Maintains class distribution across splits based on object counts,
    not just image counts for better balance in detection datasets.

    Args:
        data_dir: Source dataset with images/labels directories
        output_dir: Output directory for split dataset
        split_ratios: Split ratios (default: {"train": 0.6, "val": 0.2, "test": 0.2})
        random_seed: Random seed for reproducible splits
    """
    if split_ratios is None:
        split_ratios = DEFAULT_SPLIT_RATIOS

    random.seed(random_seed)
    logger.info(f"Starting stratified split with ratios: {split_ratios}")

    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    # Collect image-label pairs
    image_files = []
    for img in get_image_files(images_dir):
        label = labels_dir / f"{img.stem}.txt"
        if label.exists():
            image_files.append((img, label))

    if not image_files:
        raise RuntimeError("No image-label pairs found.")

    logger.info(f"Found {len(image_files)} image-label pairs")

    # Analyze class distribution
    image_class_counts = {}
    total_class_counts = Counter()
    total_objects = 0

    for img, lbl in image_files:
        counts = read_label_counts(lbl)
        image_class_counts[img] = counts
        for class_id, count in counts.items():
            total_class_counts[class_id] += count
            total_objects += count

    logger.info("Total objects per class:")
    for class_id, count in total_class_counts.items():
        logger.info(f"  Class {class_id}: {count}")

    # Calculate target objects per split
    target_objects = {
        split: total_objects * ratio for split, ratio in split_ratios.items()
    }

    # Initialize tracking variables
    current_objects = {split: 0 for split in split_ratios}
    current_class_counts = {split: Counter() for split in split_ratios}
    assignments = {}

    # Shuffle for random assignment
    random.shuffle(image_files)

    # Assign images to splits using object-aware scoring
    for img, lbl in image_files:
        img_counts = image_class_counts[img]
        img_obj_count = sum(img_counts.values())

        best_split = None
        best_score = float("inf")

        for split in split_ratios:
            # Skip if split is over target (with 5% tolerance)
            if current_objects[split] >= target_objects[split] * 1.05:
                continue

            # Calculate object distribution score
            projected_obj = current_objects[split] + img_obj_count
            obj_score = abs(projected_obj - target_objects[split])

            # Calculate class distribution score
            class_score = 0
            for class_id, count in img_counts.items():
                projected_cls = current_class_counts[split][class_id] + count
                target_cls = total_class_counts[class_id] * split_ratios[split]
                class_score += abs(projected_cls - target_cls)

            total_score = obj_score + class_score

            if total_score < best_score:
                best_score = total_score
                best_split = split

        # Fallback: assign to split with fewest objects
        if best_split is None:
            best_split = min(current_objects, key=current_objects.get)

        assignments[img] = best_split
        current_objects[best_split] += img_obj_count
        for class_id, count in img_counts.items():
            current_class_counts[best_split][class_id] += count

    # Create output directories
    for split in split_ratios:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Copy files to assigned splits
    for img, lbl in image_files:
        split = assignments[img]
        shutil.copy(img, output_dir / split / "images" / img.name)
        shutil.copy(lbl, output_dir / split / "labels" / lbl.name)

    # Print final distribution
    logger.info("\nFinal object distribution:")
    for split in split_ratios:
        logger.info(f"\n{split.upper()}")
        for class_id in sorted(total_class_counts):
            actual = current_class_counts[split][class_id]
            total = total_class_counts[class_id]
            percentage = actual / total if total > 0 else 0
            logger.info(f"  Class {class_id}: {actual} ({percentage:.2%})")

    logger.info(f"Stratified split complete. Output: {output_dir}")


# YOLO LABEL MANIPULATION UTILITIES


def load_yolo_polygons(
    label_path: Path, img_w: int, img_h: int
) -> List[Tuple[int, List[List[float]]]]:
    """
    Load YOLO format polygons from label file.

    Args:
        label_path: Path to YOLO label file
        img_w: Image width in pixels
        img_h: Image height in pixels

    Returns:
        List of tuples (class_id, [[x1, y1], [x2, y2], ...]) in pixel coordinates
    """
    polygons = []

    if not label_path.exists():
        logger.warning(f"Label file not found: {label_path}")
        return polygons

    try:
        with open(label_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parts = list(map(float, line.split()))
                if len(parts) < 5:  # Minimum: class + 2 points (4 coords)
                    logger.warning(f"Invalid line {line_num} in {label_path}: {line}")
                    continue

                cls = int(parts[0])
                coords = parts[1:]

                # Convert normalized coordinates to pixel coordinates
                pts = []
                for i in range(0, len(coords), 2):
                    if i + 1 < len(coords):
                        x = coords[i] * img_w
                        y = coords[i + 1] * img_h
                        pts.append([x, y])

                if len(pts) >= 3:  # Minimum 3 points for a polygon
                    polygons.append((cls, pts))
                else:
                    logger.warning(
                        f"Insufficient points for polygon in {label_path} line {line_num}"
                    )

    except Exception as e:
        logger.error(f"Error reading {label_path}: {e}")

    return polygons


def save_yolo_polygons(
    label_path: Path,
    polygons: List[Tuple[int, List[List[float]]]],
    crop_x1: int,
    crop_y1: int,
    crop_w: int,
    crop_h: int,
) -> None:
    """
    Save YOLO format polygons with coordinates adjusted for crop.

    Args:
        label_path: Output path for label file
        polygons: List of (class_id, [[x1, y1], ...]) tuples in pixel coordinates
        crop_x1: Crop origin x coordinate
        crop_y1: Crop origin y coordinate
        crop_w: Crop width
        crop_h: Crop height
    """
    try:
        with open(label_path, "w") as f:
            for cls, pts in polygons:
                new_pts = []

                for x, y in pts:
                    # Shift to crop coordinates
                    nx = x - crop_x1
                    ny = y - crop_y1

                    # Clip to crop boundaries
                    nx = max(0, min(crop_w, nx))
                    ny = max(0, min(crop_h, ny))

                    # Normalize to crop dimensions
                    norm_x = nx / crop_w
                    norm_y = ny / crop_h
                    new_pts.append(f"{norm_x:.6f} {norm_y:.6f}")

                if new_pts:  # Only write if we have valid points
                    line = f"{cls} {' '.join(new_pts)}"
                    f.write(line + "\n")

    except Exception as e:
        logger.error(f"Error saving {label_path}: {e}")


def remap_yolo_labels(
    dataset_path: Path,
    label_mapping: Optional[Dict[int, int]] = None,
    splits: Optional[List[str]] = None,
) -> None:
    """
    Remap YOLO class labels according to mapping.

    Args:
        dataset_path: Root path of the dataset
        label_mapping: Dictionary mapping old_class_id -> new_class_id
                      (default: {0: 0, 1: 0, 2: 1})
        splits: List of splits to process (default: ["train", "val", "test"])
    """
    if label_mapping is None:
        label_mapping = DEFAULT_LABEL_MAPPING

    if splits is None:
        splits = DATASET_SPLITS

    logger.info(f"Remapping labels with mapping: {label_mapping}")

    total_files = 0
    total_labels = 0

    for split in splits:
        label_dir = dataset_path / split / "labels"
        if not label_dir.exists():
            logger.warning(f"Label directory not found: {label_dir}")
            continue

        logger.info(f"Processing {split} labels...")
        split_files = 0
        split_labels = 0

        for label_file in label_dir.glob("*.txt"):
            try:
                with open(label_file, "r") as f:
                    lines = f.readlines()

                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue

                    old_idx = int(parts[0])
                    if old_idx in label_mapping:
                        parts[0] = str(label_mapping[old_idx])
                        new_lines.append(" ".join(parts) + "\n")
                        split_labels += 1

                with open(label_file, "w") as f:
                    f.writelines(new_lines)

                split_files += 1

            except Exception as e:
                logger.error(f"Error processing {label_file}: {e}")

        logger.info(f"  {split}: {split_files} files, {split_labels} labels processed")
        total_files += split_files
        total_labels += split_labels

    logger.info(f"Label remapping complete: {total_files} files, {total_labels} labels")


def remap_flat_labels(
    dataset_path: Path,
    label_mapping: Optional[Dict[int, int]] = None,
) -> None:
    """
    Remap YOLO class labels in a flat dataset (images/ + labels/ at root).

    Args:
        dataset_path: Root path containing images/ and labels/ subdirectories
        label_mapping: Dictionary mapping old_class_id -> new_class_id
                      (default: {0: 0, 1: 0, 2: 1})
    """
    if label_mapping is None:
        label_mapping = DEFAULT_LABEL_MAPPING

    label_dir = dataset_path / "labels"
    if not label_dir.exists():
        raise ValueError(f"Labels directory not found: {label_dir}")

    logger.info(
        f"Remapping flat labels in {dataset_path} with mapping: {label_mapping}"
    )

    total_files = 0
    total_labels = 0

    for label_file in label_dir.glob("*.txt"):
        try:
            with open(label_file, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                old_idx = int(parts[0])
                if old_idx in label_mapping:
                    parts[0] = str(label_mapping[old_idx])
                    new_lines.append(" ".join(parts) + "\n")
                    total_labels += 1

            with open(label_file, "w") as f:
                f.writelines(new_lines)

            total_files += 1

        except Exception as e:
            logger.error(f"Error processing {label_file}: {e}")

    logger.info(
        f"Flat label remapping complete: {total_files} files, {total_labels} labels"
    )


def filter_flat_labels_by_class(
    source_path: Path,
    target_path: Path,
    allowed_classes: Set[int],
    class_mapping: Optional[Dict[int, int]] = None,
) -> None:
    """
    Filter a flat YOLO dataset to only keep specified classes and optionally remap them.

    Args:
        source_path: Source dataset path with images/ and labels/ at root
        target_path: Target dataset path
        allowed_classes: Set of class IDs to keep
        class_mapping: Optional mapping from old_class_id to new_class_id
    """
    if class_mapping is None:
        class_mapping = {cls_id: i for i, cls_id in enumerate(sorted(allowed_classes))}

    src_img_dir = source_path / "images"
    src_lbl_dir = source_path / "labels"
    tgt_img_dir = target_path / "images"
    tgt_lbl_dir = target_path / "labels"

    tgt_img_dir.mkdir(parents=True, exist_ok=True)
    tgt_lbl_dir.mkdir(parents=True, exist_ok=True)

    if not src_img_dir.exists():
        raise ValueError(f"Images directory not found: {src_img_dir}")

    logger.info(
        f"Filtering flat dataset classes {allowed_classes} from {source_path} to {target_path}"
    )
    logger.info(f"Class mapping: {class_mapping}")

    total_files = 0
    total_labels = 0

    for img_path in src_img_dir.glob("*"):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_path = src_lbl_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        filtered_lines = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                old_class_id = int(parts[0])
                if old_class_id in allowed_classes:
                    new_class_id = class_mapping[old_class_id]
                    parts[0] = str(new_class_id)
                    filtered_lines.append(" ".join(parts) + "\n")
                    total_labels += 1

        if filtered_lines:
            shutil.copy2(img_path, tgt_img_dir / img_path.name)
            with open(tgt_lbl_dir / f"{img_path.stem}.txt", "w") as f:
                f.writelines(filtered_lines)
            total_files += 1

    logger.info(f"Flat filtering complete: {total_files} files, {total_labels} labels")


def crop_flat_dataset(
    input_dir: Path,
    output_dir: Path,
    buffer_ratio: float = DEFAULT_BUFFER_RATIO,
) -> None:
    """
    Crop a flat dataset (images/ + labels/ at root) around all objects per image.

    Args:
        input_dir: Input flat dataset with images/ and labels/ at root
        output_dir: Output flat dataset directory
        buffer_ratio: Buffer size around objects (default: 0.20)
    """
    img_dir = input_dir / "images"
    lbl_dir = input_dir / "labels"
    out_img_dir = output_dir / "images"
    out_lbl_dir = output_dir / "labels"

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists():
        raise ValueError(f"Images directory not found: {img_dir}")

    logger.info(f"Cropping flat dataset from {input_dir} to {output_dir}")
    logger.info(f"Buffer ratio: {buffer_ratio}")

    total_images = 0

    for img_path in img_dir.glob("*"):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h_orig, w_orig = img.shape[:2]
            polygons = load_yolo_polygons(
                lbl_dir / f"{img_path.stem}.txt", w_orig, h_orig
            )

            if not polygons:
                continue

            all_points = []
            for _, pts in polygons:
                all_points.extend(pts)

            x1, y1, x2, y2 = calculate_crop_bounds(
                all_points, buffer_ratio, w_orig, h_orig
            )
            crop = img[y1:y2, x1:x2]
            c_h, c_w = crop.shape[:2]

            cv2.imwrite(str(out_img_dir / img_path.name), crop)
            save_yolo_polygons(
                out_lbl_dir / img_path.with_suffix(".txt").name,
                polygons,
                x1,
                y1,
                c_w,
                c_h,
            )

            total_images += 1

        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")

    logger.info(f"Flat crop complete: {total_images} images processed")


# DATASET CONVERSION UTILITIES


def convert_detection_to_classification(
    src_root: Path,
    dest_root: Path,
    class_mapping: Optional[Dict[int, str]] = None,
    splits: Optional[List[str]] = None,
) -> None:
    """
    Convert YOLO detection dataset to classification dataset.

    Each image with a single object is copied to the appropriate class folder.
    Images with multiple objects or no objects are skipped.

    Args:
        src_root: Source detection dataset root
        dest_root: Destination classification dataset root
        class_mapping: Dictionary mapping class_id -> class_name
                      (default: {0: "cherry", 1: "cherry-imperfect"})
        splits: List of splits to process (default: ["train", "val", "test"])
    """
    if class_mapping is None:
        class_mapping = CLASSIFICATION_MAPPING

    if splits is None:
        splits = DATASET_SPLITS

    logger.info("Converting detection dataset to classification")
    logger.info(f"Source: {src_root}")
    logger.info(f"Destination: {dest_root}")
    logger.info(f"Class mapping: {class_mapping}")

    total_images = 0

    for split in splits:
        src_img_dir = src_root / split / "images"
        src_lbl_dir = src_root / split / "labels"

        if not src_img_dir.exists():
            logger.warning(f"Images directory not found: {src_img_dir}")
            continue

        split_count = 0

        for img_path in src_img_dir.glob("*"):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_path = src_lbl_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue

            try:
                with open(label_path, "r") as f:
                    lines = f.readlines()

                # Only process single-object images
                if len(lines) != 1:
                    continue

                line = lines[0].strip()
                if not line:
                    continue

                class_id = int(line.split()[0])

                # Only copy if class is in our mapping
                if class_id in class_mapping:
                    class_name = class_mapping[class_id]
                    target_dir = dest_root / split / class_name
                    target_dir.mkdir(parents=True, exist_ok=True)

                    shutil.copy2(img_path, target_dir / img_path.name)
                    split_count += 1

            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")

        logger.info(f"  {split}: {split_count} images converted")
        total_images += split_count

    logger.info(f"Classification dataset created at: {dest_root}")
    logger.info(f"Total images converted: {total_images}")


# IMAGE CROPPING UTILITIES


def calculate_crop_bounds(
    points: List[List[float]],
    buffer_ratio: float = DEFAULT_BUFFER_RATIO,
    img_w: int = 0,
    img_h: int = 0,
) -> Tuple[int, int, int, int]:
    """
    Calculate crop bounds around a set of points with buffer.

    Args:
        points: List of [x, y] coordinates
        buffer_ratio: Buffer size as ratio of object size
        img_w: Image width (for clipping)
        img_h: Image height (for clipping)

    Returns:
        Tuple of (x1, y1, x2, y2) crop bounds
    """
    if not points:
        return 0, 0, 100, 100

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    # Calculate object dimensions
    obj_w = max(xs) - min(xs)
    obj_h = max(ys) - min(ys)

    # Calculate buffer based on object size
    buffer = int(buffer_ratio * max(obj_w, obj_h, 1))  # Ensure non-zero

    # Calculate bounds with buffer
    x1 = max(0, int(min(xs) - buffer))
    y1 = max(0, int(min(ys) - buffer))
    x2 = min(img_w if img_w > 0 else int(max(xs) + buffer), int(max(xs) + buffer))
    y2 = min(img_h if img_h > 0 else int(max(ys) + buffer), int(max(ys) + buffer))

    return x1, y1, x2, y2


def crop_single_objects(
    input_dir: Path,
    output_dir: Path,
    buffer_ratio: float = DEFAULT_BUFFER_RATIO,
    splits: Optional[List[str]] = None,
) -> None:
    """
    Crop images to individual objects, creating one image per object.

    Args:
        input_dir: Input dataset root with detection labels
        output_dir: Output directory for cropped images
        buffer_ratio: Buffer size around objects (default: 0.20)
        splits: List of splits to process (default: ["train", "val", "test"])
    """
    if splits is None:
        splits = DATASET_SPLITS

    logger.info(f"Cropping single objects from {input_dir} to {output_dir}")
    logger.info(f"Buffer ratio: {buffer_ratio}")

    total_objects = 0

    for split in splits:
        img_dir = input_dir / split / "images"
        lbl_dir = input_dir / split / "labels"

        out_img_dir = output_dir / split / "images"
        out_lbl_dir = output_dir / split / "labels"

        # Ensure output directories exist
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        if not img_dir.exists():
            logger.warning(f"Images directory not found: {img_dir}")
            continue

        split_objects = 0

        for img_path in img_dir.glob("*"):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                h_orig, w_orig = img.shape[:2]

                # Load polygons for this image
                polygons = load_yolo_polygons(
                    lbl_dir / f"{img_path.stem}.txt", w_orig, h_orig
                )

                # Process each object separately
                for idx, (cls, pts) in enumerate(polygons):
                    # Calculate crop bounds for this object
                    x1, y1, x2, y2 = calculate_crop_bounds(
                        pts, buffer_ratio, w_orig, h_orig
                    )

                    # Crop the image
                    crop = img[y1:y2, x1:x2]
                    c_h, c_w = crop.shape[:2]

                    # Save with index suffix for multiple objects
                    base_name = f"{img_path.stem}_{idx}"
                    cv2.imwrite(
                        str(out_img_dir / f"{base_name}{img_path.suffix}"), crop
                    )

                    # Save adjusted label
                    save_yolo_polygons(
                        out_lbl_dir / f"{base_name}.txt", [(cls, pts)], x1, y1, c_w, c_h
                    )

                    split_objects += 1

            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")

        logger.info(f"  {split}: {split_objects} objects cropped")
        total_objects += split_objects

    logger.info(f"Single object cropping complete: {total_objects} total objects")


def crop_multi_objects(
    input_dir: Path,
    output_dir: Path,
    buffer_ratio: float = DEFAULT_BUFFER_RATIO,
    splits: Optional[List[str]] = None,
) -> None:
    """
    Crop images to contain all objects with buffer (one crop per image).

    Args:
        input_dir: Input dataset root with detection labels
        output_dir: Output directory for cropped images
        buffer_ratio: Buffer size around objects (default: 0.20)
        splits: List of splits to process (default: ["train", "val", "test"])
    """
    if splits is None:
        splits = DATASET_SPLITS

    logger.info(f"Cropping multi-object images from {input_dir} to {output_dir}")
    logger.info(f"Buffer ratio: {buffer_ratio}")

    total_images = 0

    for split in splits:
        img_dir = input_dir / split / "images"
        lbl_dir = input_dir / split / "labels"

        out_img_dir = output_dir / split / "images"
        out_lbl_dir = output_dir / split / "labels"

        # Ensure output directories exist
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        if not img_dir.exists():
            logger.warning(f"Images directory not found: {img_dir}")
            continue

        split_images = 0

        for img_path in img_dir.glob("*"):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                h_orig, w_orig = img.shape[:2]

                # Load all polygons
                polygons = load_yolo_polygons(
                    lbl_dir / f"{img_path.stem}.txt", w_orig, h_orig
                )

                if not polygons:
                    continue

                # Collect all points from all objects
                all_points = []
                for _, pts in polygons:
                    all_points.extend(pts)

                # Calculate crop bounds around all objects
                x1, y1, x2, y2 = calculate_crop_bounds(
                    all_points, buffer_ratio, w_orig, h_orig
                )

                # Crop the image
                crop = img[y1:y2, x1:x2]
                c_h, c_w = crop.shape[:2]

                # Save cropped image
                cv2.imwrite(str(out_img_dir / img_path.name), crop)

                # Save adjusted labels for all objects
                save_yolo_polygons(
                    out_lbl_dir / img_path.with_suffix(".txt").name,
                    polygons,
                    x1,
                    y1,
                    c_w,
                    c_h,
                )

                split_images += 1

            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")

        logger.info(f"  {split}: {split_images} images cropped")
        total_images += split_images

    logger.info(f"Multi-object cropping complete: {total_images} total images")


# PIPELINE FUNCTIONS


def create_data_yaml(dataset_path: Path) -> None:
    """
    Create data.yaml file for YOLO dataset.

    Args:
        dataset_path: Path to dataset directory
    """
    # Determine class names based on dataset path
    if "detect" in dataset_path.name:
        class_names = ["cherry", "stem"]
    elif "cls" in dataset_path.name:
        class_names = ["cherry", "cherry-imperfect"]
    else:
        # Default fallback
        class_names = ["cherry", "cherry-imperfect", "stem"]

    yaml_content = {
        "train": "train",
        "val": "val",
        "test": "test",
        "nc": len(class_names),
        "names": class_names,
        "roboflow": {
            "workspace": "weatherapp",
            "project": "cherry-6vrfb",
            "version": 11,
            "license": "CC BY 4.0",
            "url": "https://universe.roboflow.com/weatherapp/cherry-6vrfb/dataset/11",
        },
    }

    yaml_path = dataset_path / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    logger.info(f"Created data.yaml at {yaml_path}")


def filter_labels_by_class(
    source_path: Path,
    target_path: Path,
    allowed_classes: Set[int],
    class_mapping: Optional[Dict[int, int]] = None,
) -> None:
    """
    Filter YOLO dataset to only keep specified classes and optionally remap them.

    Args:
        source_path: Source dataset path
        target_path: Target dataset path
        allowed_classes: Set of class IDs to keep
        class_mapping: Optional mapping from old_class_id to new_class_id
    """
    splits = DATASET_SPLITS

    if class_mapping is None:
        class_mapping = {cls_id: i for i, cls_id in enumerate(sorted(allowed_classes))}

    logger.info(
        f"Filtering classes {allowed_classes} from {source_path} to {target_path}"
    )
    logger.info(f"Class mapping: {class_mapping}")

    total_files = 0
    total_labels = 0

    for split in splits:
        src_img_dir = source_path / split / "images"
        src_lbl_dir = source_path / split / "labels"
        tgt_img_dir = target_path / split / "images"
        tgt_lbl_dir = target_path / split / "labels"

        # Create target directories
        tgt_img_dir.mkdir(parents=True, exist_ok=True)
        tgt_lbl_dir.mkdir(parents=True, exist_ok=True)

        if not src_img_dir.exists():
            logger.warning(f"Images directory not found: {src_img_dir}")
            continue

        split_files = 0
        split_labels = 0

        for img_path in src_img_dir.glob("*"):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_path = src_lbl_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue

            # Read and filter labels
            filtered_lines = []
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue

                    old_class_id = int(parts[0])
                    if old_class_id in allowed_classes:
                        new_class_id = class_mapping[old_class_id]
                        parts[0] = str(new_class_id)
                        filtered_lines.append(" ".join(parts) + "\n")
                        split_labels += 1

            # Only copy if we have valid labels
            if filtered_lines:
                shutil.copy2(img_path, tgt_img_dir / img_path.name)
                with open(tgt_lbl_dir / f"{img_path.stem}.txt", "w") as f:
                    f.writelines(filtered_lines)
                split_files += 1

        logger.info(f"  {split}: {split_files} files, {split_labels} labels processed")
        total_files += split_files
        total_labels += split_labels

    logger.info(f"Filtering complete: {total_files} files, {total_labels} labels")


def run_full_dataset_pipeline() -> None:
    """
    Run the complete dataset processing pipeline.

    The source dataset (original) is a flat dataset with images/ and labels/ at root.
    Class IDs in the original: 0=cherry, 1=cherry-imperfect, 2=stem.

    Pipeline steps:
    1. original (flat) -> data_detect_remapped (flat): remap cherry/cherry-imperfect->0, stem->1
    2. original (flat) -> data_cls_remapped (flat): keep cherry(0) and cherry-imperfect(1), drop stem
    3. data_cls_remapped (flat) -> data_cls_clipped (flat): crop around objects
    4. data_detect_remapped (flat) -> data_detect_stratified (split): stratified train/val/test
    5. data_cls_clipped (flat) -> data_cls_stratified (split): stratified train/val/test
    6. data_detect_stratified (split) -> data_detect_augmented (split): augment
    7. data_cls_stratified (split) -> data_cls_augmented (split): convert to cls format + augment
    """
    buffer_ratio = DEFAULT_BUFFER_RATIO
    random_seed = DEFAULT_RANDOM_SEED
    split_ratios = DEFAULT_SPLIT_RATIOS

    # Clean all output directories before starting
    for output_dir in [
        DATASET_DETECT_REMAPPED_DIR,
        DATASET_CLS_REMAPPED_DIR,
        DATASET_CLS_CLIPPED_DIR,
        DATASET_DETECT_STRATIFIED_DIR,
        DATASET_CLS_STRATIFIED_DIR,
        DATASET_DETECT_AUGMENTED_DIR,
        DATASET_CLS_AUGMENTED_DIR,
    ]:
        if output_dir.exists():
            shutil.rmtree(output_dir)

    logger.info("STARTING FULL DATASET PIPELINE")

    # Step 1: original -> data_detect_remapped
    # Remap: cherry->0, cherry-imperfect->0, stem->1 (binary: fruit vs stem)
    logger.info(
        "STEP 1: original -> data_detect_remapped (cherry/cherry-imperfect->0, stem->1)"
    )
    shutil.copytree(DATASET_ORIGINAL_DIR, DATASET_DETECT_REMAPPED_DIR)
    remap_flat_labels(DATASET_DETECT_REMAPPED_DIR, DETECTION_REMAPPED_MAPPING)
    create_data_yaml(DATASET_DETECT_REMAPPED_DIR)

    # Step 2: original -> data_cls_remapped
    # Keep cherry(0) and cherry-imperfect(1), drop stem(2)
    logger.info(
        "STEP 2: original -> data_cls_remapped (cherry->0, cherry-imperfect->1, drop stem)"
    )
    filter_flat_labels_by_class(
        DATASET_ORIGINAL_DIR,
        DATASET_CLS_REMAPPED_DIR,
        allowed_classes={0, 1},
        class_mapping=CLS_REMAPPED_MAPPING,
    )
    create_data_yaml(DATASET_CLS_REMAPPED_DIR)

    # Step 3: data_cls_remapped -> data_cls_clipped
    # Crop each image tightly around all its objects
    logger.info("STEP 3: data_cls_remapped -> data_cls_clipped (crop around objects)")
    crop_flat_dataset(DATASET_CLS_REMAPPED_DIR, DATASET_CLS_CLIPPED_DIR, buffer_ratio)
    shutil.copy2(
        DATASET_CLS_REMAPPED_DIR / "data.yaml",
        DATASET_CLS_CLIPPED_DIR / "data.yaml",
    )

    # Step 4: data_detect_remapped -> data_detect_stratified
    logger.info(
        "STEP 4: data_detect_remapped -> data_detect_stratified (stratified split)"
    )
    stratified_dataset_split(
        DATASET_DETECT_REMAPPED_DIR,
        DATASET_DETECT_STRATIFIED_DIR,
        split_ratios,
        random_seed,
    )
    shutil.copy2(
        DATASET_DETECT_REMAPPED_DIR / "data.yaml",
        DATASET_DETECT_STRATIFIED_DIR / "data.yaml",
    )

    # Step 5: data_cls_clipped -> data_cls_stratified
    logger.info("STEP 5: data_cls_clipped -> data_cls_stratified (stratified split)")
    stratified_dataset_split(
        DATASET_CLS_CLIPPED_DIR, DATASET_CLS_STRATIFIED_DIR, split_ratios, random_seed
    )
    shutil.copy2(
        DATASET_CLS_CLIPPED_DIR / "data.yaml",
        DATASET_CLS_STRATIFIED_DIR / "data.yaml",
    )

    # Step 6: data_detect_stratified -> data_detect_augmented
    logger.info("STEP 6: data_detect_stratified -> data_detect_augmented (augment)")
    augment_detection_dataset(
        DATASET_DETECT_STRATIFIED_DIR, DATASET_DETECT_AUGMENTED_DIR, augment_factor=4
    )
    shutil.copy2(
        DATASET_DETECT_STRATIFIED_DIR / "data.yaml",
        DATASET_DETECT_AUGMENTED_DIR / "data.yaml",
    )

    # Step 7: data_cls_stratified -> data_cls_augmented
    # augment_classification_dataset expects split/class/images folder structure,
    # so convert from YOLO detection format first using a temp directory.
    logger.info(
        "STEP 7: data_cls_stratified -> data_cls_augmented (convert to cls format + augment)"
    )
    temp_cls_dir = DATASET_CLS_STRATIFIED_DIR.parent / "temp_cls_format"
    if temp_cls_dir.exists():
        shutil.rmtree(temp_cls_dir)
    convert_detection_to_classification(DATASET_CLS_STRATIFIED_DIR, temp_cls_dir)
    augment_classification_dataset(
        temp_cls_dir, DATASET_CLS_AUGMENTED_DIR, augment_factor=4
    )
    shutil.rmtree(temp_cls_dir)

    logger.info("DATASET PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("Generated datasets:")
    logger.info(f"  - Detection remapped:        {DATASET_DETECT_REMAPPED_DIR}")
    logger.info(f"  - Classification remapped:   {DATASET_CLS_REMAPPED_DIR}")
    logger.info(f"  - Classification clipped:    {DATASET_CLS_CLIPPED_DIR}")
    logger.info(f"  - Detection stratified:      {DATASET_DETECT_STRATIFIED_DIR}")
    logger.info(f"  - Classification stratified: {DATASET_CLS_STRATIFIED_DIR}")
    logger.info(f"  - Detection augmented:       {DATASET_DETECT_AUGMENTED_DIR}")
    logger.info(f"  - Classification augmented:  {DATASET_CLS_AUGMENTED_DIR}")


if __name__ == "__main__":
    run_full_dataset_pipeline()
