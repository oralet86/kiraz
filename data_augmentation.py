"""
Data Augmentation Module for Kiraz Project

This module provides Albumentations-based data augmentation for both
classification and detection tasks. It handles batch processing of datasets
and saves augmented versions to disk for training.

Features:
- Classification augmentation with folder-based structure
- Detection augmentation with YOLO label handling
- Separate transforms for train/validation splits
- Batch processing with progress tracking
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

import albumentations as A
import albumentations.core.bbox_utils as bbox_utils
import cv2
import numpy as np
from log import logger

# Import project paths
from paths import (
    DATASET_CLS_AUGMENTED_DIR,
    DATASET_DETECT_AUGMENTED_DIR,
    DATASET_CLS_REMAPPED_DIR,
    DATASET_DETECT_REMAPPED_DIR,
    ensure_directories,
)


def get_classification_transforms(split: str = "train") -> A.Compose:
    """
    Get Albumentations transforms for classification tasks.

    Args:
        split: Dataset split ("train", "val", or "test")

    Returns:
        Albumentations compose object
    """
    if split == "train":
        transforms = A.Compose(
            [
                A.RandomResizedCrop(size=(320, 320), scale=(0.3, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    scale=(1.0, 1.0),
                    translate_percent=(-0.1, 0.1),
                    rotate=(-2.0, 2.0),
                    shear=(-0.03, 0.03),
                    p=0.8,
                ),
                A.Perspective(scale=(0.0001, 0.0003), p=0.5),
                A.HueSaturationValue(
                    hue_shift_limit=9,
                    sat_shift_limit=76,
                    val_shift_limit=89,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.35, contrast_limit=0.35, p=0.5
                ),
            ]
        )
    else:
        transforms = None

    return transforms


def get_detection_transforms(split: str = "train") -> A.Compose:
    """
    Get Albumentations transforms for detection tasks with bbox handling.
    Lighter augmentation to preserve model performance.

    Args:
        split: Dataset split ("train", "val", or "test")

    Returns:
        Albumentations compose object with bbox parameters
    """
    bbox_params = A.BboxParams(
        format="yolo", label_fields=["class_labels"], min_visibility=0.6
    )

    if split == "train":
        transforms = A.Compose(
            [
                A.HorizontalFlip(p=0.25),
                A.RandomBrightnessContrast(
                    brightness_limit=0.15, contrast_limit=0.15, p=0.25
                ),
                A.HueSaturationValue(
                    hue_shift_limit=4,
                    sat_shift_limit=20,
                    val_shift_limit=30,
                    p=0.25,
                ),
            ],
            bbox_params=bbox_params,
        )
    else:
        transforms = None

    return transforms


def validate_and_fix_bbox(bbox: List[float]) -> List[float]:
    """
    Validate and fix bounding box coordinates.

    Args:
        bbox: [x_center, y_center, width, height] in normalized format

    Returns:
        Validated bounding box or None if invalid
    """
    if len(bbox) < 4:
        return None

    try:
        x_center, y_center, width, height = map(float, bbox[:4])

        # Ensure positive width and height
        width = max(0.01, width)
        height = max(0.01, height)

        # Ensure bbox stays within bounds
        x_min = max(0.0, x_center - width / 2)
        y_min = max(0.0, y_center - height / 2)
        x_max = min(1.0, x_center + width / 2)
        y_max = min(1.0, y_center + height / 2)

        # Recalculate center and size from clipped bounds
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        return [x_center, y_center, width, height]

    except (ValueError, TypeError):
        return None


def parse_yolo_label(label_path: Path) -> Tuple[List[List[float]], List[int]]:
    """
    Parse YOLO label file.

    Args:
        label_path: Path to YOLO label file

    Returns:
        Tuple of (bboxes, class_labels)
        bboxes: List of [x_center, y_center, width, height] in normalized format
        class_labels: List of class IDs
    """
    bboxes = []
    class_labels = []

    if not label_path.exists():
        return bboxes, class_labels

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]

            if len(coords) == 4:
                # Standard YOLO format: x_center, y_center, width, height
                bbox = coords
            elif len(coords) >= 8 and len(coords) % 2 == 0:
                # Polygon/OBB format: x1 y1 x2 y2 x3 y3 x4 y4 ...
                # Convert to AABB YOLO format
                xs = coords[0::2]
                ys = coords[1::2]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                bbox = [
                    (x_min + x_max) / 2,
                    (y_min + y_max) / 2,
                    x_max - x_min,
                    y_max - y_min,
                ]
            else:
                continue

            fixed_bbox = validate_and_fix_bbox(bbox)
            if fixed_bbox is not None:
                bboxes.append(fixed_bbox)
                class_labels.append(class_id)

    return bboxes, class_labels


def save_yolo_label(
    label_path: Path, bboxes: List[List[float]], class_labels: List[int]
) -> None:
    """
    Save YOLO label file.

    Args:
        label_path: Path to save label file
        bboxes: List of [x_center, y_center, width, height] in normalized format
        class_labels: List of class IDs
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)

    with open(label_path, "w") as f:
        for bbox, class_id in zip(bboxes, class_labels):
            line = f"{class_id} {' '.join([f'{x:.6f}' for x in bbox])}\n"
            f.write(line)


def draw_bboxes_on_image(
    image: np.ndarray, bboxes: List[List[float]], class_labels: List[int]
) -> np.ndarray:
    """
    Draw bounding boxes on image using Albumentations coordinate utilities.

    Args:
        image: Input image in BGR format
        bboxes: List of [x_center, y_center, width, height] in normalized format (YOLO)
        class_labels: List of class IDs

    Returns:
        Image with bounding boxes drawn
    """
    if not bboxes:
        return image

    img_height, img_width = image.shape[:2]
    result_image = image.copy()

    # Convert YOLO format to Albumentations format for robust coordinate handling
    # YOLO: [x_center, y_center, width, height] normalized
    # Albumentations: [x_min, y_min, x_max, y_max] normalized
    bboxes_alb = []
    for bbox in bboxes:
        x_center, y_center, width, height = bbox

        # Convert YOLO to Pascal VOC format, then normalize
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2

        # Ensure coordinates are within [0, 1] bounds
        x_min = max(0.0, min(1.0, x_min))
        y_min = max(0.0, min(1.0, y_min))
        x_max = max(0.0, min(1.0, x_max))
        y_max = max(0.0, min(1.0, y_max))

        bboxes_alb.append([x_min, y_min, x_max, y_max])

    # Use Albumentations to denormalize coordinates to pixel values
    bboxes_alb_np = np.array(bboxes_alb)
    bboxes_pixels = bbox_utils.denormalize_bboxes(
        bboxes_alb_np, (img_height, img_width)
    )

    # Draw bounding boxes
    for bbox_pixels, class_id in zip(bboxes_pixels, class_labels):
        x_min, y_min, x_max, y_max = map(int, bbox_pixels)

        # Ensure coordinates are within image bounds
        x_min = max(0, min(x_min, img_width - 1))
        y_min = max(0, min(y_min, img_height - 1))
        x_max = max(0, min(x_max, img_width - 1))
        y_max = max(0, min(y_max, img_height - 1))

        # Draw rectangle (green color, 2px thickness)
        cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Add class label text
        label_text = f"class_{class_id}"
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        label_x = x_min
        label_y = y_min - 5 if y_min > 20 else y_min + label_size[1] + 5

        # Draw label background
        cv2.rectangle(
            result_image,
            (label_x, label_y - label_size[1]),
            (label_x + label_size[0], label_y + 5),
            (0, 255, 0),
            -1,
        )

        # Draw label text
        cv2.putText(
            result_image,
            label_text,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )

    return result_image


def augment_classification_dataset(
    source_dir: Path, target_dir: Path, augment_factor: int = 3
) -> None:
    """
    Augment classification dataset.

    Args:
        source_dir: Source dataset directory with split/class folders
        target_dir: Target directory for augmented dataset
        augment_factor: Number of augmented versions per original image
    """
    logger.info(
        f"Starting classification augmentation from {source_dir} to {target_dir}"
    )

    splits = ["train", "val", "test"]

    for split in splits:
        source_split_dir = source_dir / split
        target_split_dir = target_dir / split

        if not source_split_dir.exists():
            logger.warning(f"Source split directory not found: {source_split_dir}")
            continue

        # Get class directories
        class_dirs = [d for d in source_split_dir.iterdir() if d.is_dir()]

        for class_dir in class_dirs:
            class_name = class_dir.name
            target_class_dir = target_split_dir / class_name
            target_class_dir.mkdir(parents=True, exist_ok=True)

            # Get image files
            image_files = []
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]:
                image_files.extend(class_dir.glob(f"*{ext}"))
                image_files.extend(class_dir.glob(f"*{ext.upper()}"))

            logger.info(f"Processing {split}/{class_name}: {len(image_files)} images")

            # For val/test, just copy verbatim
            if split in ("val", "test"):
                processed_count = 0
                for img_path in image_files:
                    try:
                        image = cv2.imread(str(img_path))
                        if image is None:
                            logger.warning(f"Could not read image: {img_path}")
                            continue

                        original_target = target_class_dir / img_path.name
                        cv2.imwrite(str(original_target), image)

                        processed_count += 1
                        if processed_count % 100 == 0:
                            logger.info(
                                f"  Copied {processed_count}/{len(image_files)} images"
                            )

                    except Exception as e:
                        logger.error(f"Error copying {img_path}: {e}")
                        continue

                logger.info(
                    f"  Completed {split}/{class_name}: {processed_count} images copied"
                )
                continue

            transforms = get_classification_transforms(split)

            processed_count = 0
            for img_path in image_files:
                try:
                    # Read image
                    image = cv2.imread(str(img_path))
                    if image is None:
                        logger.warning(f"Could not read image: {img_path}")
                        continue

                    # Save original image
                    original_target = target_class_dir / img_path.name
                    cv2.imwrite(str(original_target), image)

                    # Generate augmented versions
                    for aug_idx in range(augment_factor):
                        augmented = transforms(image=image)
                        aug_image = augmented["image"]

                        # Create augmented filename
                        stem = img_path.stem
                        suffix = img_path.suffix
                        aug_filename = f"{stem}_aug_{aug_idx}{suffix}"
                        aug_path = target_class_dir / aug_filename

                        # Save augmented image
                        cv2.imwrite(str(aug_path), aug_image)

                    processed_count += 1
                    if processed_count % 100 == 0:
                        logger.info(
                            f"  Processed {processed_count}/{len(image_files)} images"
                        )

                except Exception as e:
                    logger.error(f"Error processing {img_path}: {e}")
                    continue

            logger.info(
                f"  Completed {split}/{class_name}: {processed_count} images processed"
            )


def augment_detection_dataset(
    source_dir: Path, target_dir: Path, augment_factor: int = 3
) -> None:
    """
    Augment detection dataset with YOLO label handling.
    Also creates visual bounding box outputs.

    Args:
        source_dir: Source dataset directory with split/images/labels structure
        target_dir: Target directory for augmented dataset
        augment_factor: Number of augmented versions per original image
    """
    logger.info(f"Starting detection augmentation from {source_dir} to {target_dir}")

    splits = ["train", "val", "test"]

    for split in splits:
        source_images_dir = source_dir / split / "images"
        source_labels_dir = source_dir / split / "labels"
        target_images_dir = target_dir / split / "images"
        target_labels_dir = target_dir / split / "labels"
        target_bbox_dir = target_dir / f"{split}_bbox"

        if not source_images_dir.exists():
            logger.warning(f"Source images directory not found: {source_images_dir}")
            continue

        # Create target directories
        target_images_dir.mkdir(parents=True, exist_ok=True)
        target_labels_dir.mkdir(parents=True, exist_ok=True)
        target_bbox_dir.mkdir(parents=True, exist_ok=True)

        # Get image files
        image_files = []
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]:
            image_files.extend(source_images_dir.glob(f"*{ext}"))
            image_files.extend(source_images_dir.glob(f"*{ext.upper()}"))

        logger.info(f"Processing {split}: {len(image_files)} images")

        # For val/test, just copy verbatim
        if split in ("val", "test"):
            processed_count = 0
            for img_path in image_files:
                try:
                    image = cv2.imread(str(img_path))
                    if image is None:
                        logger.warning(f"Could not read image: {img_path}")
                        continue

                    label_path = source_labels_dir / (img_path.stem + ".txt")
                    bboxes, class_labels = parse_yolo_label(label_path)

                    original_img_target = target_images_dir / img_path.name
                    original_label_target = target_labels_dir / (img_path.stem + ".txt")
                    bbox_img_target = target_bbox_dir / img_path.name

                    cv2.imwrite(str(original_img_target), image)
                    save_yolo_label(original_label_target, bboxes, class_labels)

                    # Save image with bounding boxes
                    if bboxes:
                        bbox_image = draw_bboxes_on_image(image, bboxes, class_labels)
                        cv2.imwrite(str(bbox_img_target), bbox_image)

                    processed_count += 1
                    if processed_count % 100 == 0:
                        logger.info(
                            f"  Copied {processed_count}/{len(image_files)} images"
                        )

                except Exception as e:
                    logger.error(f"Error copying {img_path}: {e}")
                    continue

            logger.info(f"  Completed {split}: {processed_count} images copied")
            continue

        transforms = get_detection_transforms(split)

        processed_count = 0
        for img_path in image_files:
            try:
                # Read image
                image = cv2.imread(str(img_path))
                if image is None:
                    logger.warning(f"Could not read image: {img_path}")
                    continue

                # Get corresponding label file
                label_path = source_labels_dir / (img_path.stem + ".txt")
                bboxes, class_labels = parse_yolo_label(label_path)

                # Save original image and labels
                original_img_target = target_images_dir / img_path.name
                original_label_target = target_labels_dir / (img_path.stem + ".txt")
                bbox_img_target = target_bbox_dir / img_path.name

                cv2.imwrite(str(original_img_target), image)
                save_yolo_label(original_label_target, bboxes, class_labels)

                # Save image with bounding boxes
                if bboxes:
                    bbox_image = draw_bboxes_on_image(image, bboxes, class_labels)
                    cv2.imwrite(str(bbox_img_target), bbox_image)

                # Generate augmented versions
                for aug_idx in range(augment_factor):
                    try:
                        # Apply augmentation
                        augmented = transforms(
                            image=image, bboxes=bboxes, class_labels=class_labels
                        )
                        aug_image = augmented["image"]
                        aug_bboxes = augmented["bboxes"]
                        aug_class_labels = [int(c) for c in augmented["class_labels"]]

                        # Create augmented filenames
                        stem = img_path.stem
                        img_suffix = img_path.suffix
                        aug_img_filename = f"{stem}_aug_{aug_idx}{img_suffix}"
                        aug_label_filename = f"{stem}_aug_{aug_idx}.txt"

                        aug_img_path = target_images_dir / aug_img_filename
                        aug_label_path = target_labels_dir / aug_label_filename
                        aug_bbox_path = target_bbox_dir / aug_img_filename

                        # Save augmented image and labels
                        cv2.imwrite(str(aug_img_path), aug_image)
                        save_yolo_label(aug_label_path, aug_bboxes, aug_class_labels)

                        # Save augmented image with bounding boxes
                        if aug_bboxes:
                            aug_bbox_image = draw_bboxes_on_image(
                                aug_image, aug_bboxes, aug_class_labels
                            )
                            cv2.imwrite(str(aug_bbox_path), aug_bbox_image)

                    except Exception as aug_error:
                        # Log augmentation error but continue with other images
                        logger.warning(
                            f"Skipping augmentation for {img_path.name} (aug {aug_idx}): {aug_error}"
                        )
                        continue

                processed_count += 1
                if processed_count % 100 == 0:
                    logger.info(
                        f"  Processed {processed_count}/{len(image_files)} images"
                    )

            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue

        logger.info(f"  Completed {split}: {processed_count} images processed")


def main() -> None:
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Augment datasets using Albumentations"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["cls", "detect", "both"],
        default="both",
        help="Task type to augment (default: both)",
    )

    args = parser.parse_args()

    # Clean up old outputs before regenerating
    if args.task in ["cls", "both"] and DATASET_CLS_AUGMENTED_DIR.exists():
        logger.info(f"Removing old {DATASET_CLS_AUGMENTED_DIR}...")
        shutil.rmtree(DATASET_CLS_AUGMENTED_DIR)
    if args.task in ["detect", "both"] and DATASET_DETECT_AUGMENTED_DIR.exists():
        logger.info(f"Removing old {DATASET_DETECT_AUGMENTED_DIR}...")
        shutil.rmtree(DATASET_DETECT_AUGMENTED_DIR)

    # Ensure directories exist
    ensure_directories()

    augment_factor = 4

    if args.task in ["cls", "both"]:
        logger.info("Starting classification dataset augmentation...")
        augment_classification_dataset(
            DATASET_CLS_REMAPPED_DIR, DATASET_CLS_AUGMENTED_DIR, augment_factor
        )
        logger.info("Classification augmentation completed!")

    if args.task in ["detect", "both"]:
        logger.info("Starting detection dataset augmentation...")
        augment_detection_dataset(
            DATASET_DETECT_REMAPPED_DIR, DATASET_DETECT_AUGMENTED_DIR, augment_factor
        )
        logger.info("Detection augmentation completed!")

    logger.info("All augmentation tasks completed!")


if __name__ == "__main__":
    main()
