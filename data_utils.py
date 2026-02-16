import os
import yaml
import random
import shutil
from pathlib import Path
from collections import defaultdict, Counter

DATASET_PATH = Path("data_resized")
SPLIT_PATH = Path("data_resized_split")

SPLITS = ["train", "val", "test"]
SPLIT_RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}

IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
RANDOM_SEED = 42


def load_class_names(dataset_path: Path):
    yaml_path = dataset_path / "data.yaml"
    if not yaml_path.exists():
        return None

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    if "names" in data:
        if isinstance(data["names"], list):
            return data["names"]
        elif isinstance(data["names"], dict):
            return [data["names"][k] for k in sorted(data["names"].keys())]
    return None


def get_images(images_dir: Path):
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(images_dir.glob(ext))
    return images


def read_label_counts(label_path: Path):
    counts = Counter()
    if not label_path.exists():
        return counts

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                counts[int(parts[0])] += 1
    return counts


def dataset_statistics(dataset_path: Path):
    class_names = load_class_names(dataset_path)

    overall_class_counts = defaultdict(int)
    overall_images = 0
    overall_objects = 0

    print("Dataset Statistics\n")

    for split in SPLITS:
        split_path = dataset_path / split
        if not split_path.exists():
            continue

        images_dir = split_path / "images"
        labels_dir = split_path / "labels"

        images = get_images(images_dir)
        label_files = list(labels_dir.glob("*.txt"))

        class_counts = defaultdict(int)
        total_objects = 0

        for lf in label_files:
            counts = read_label_counts(lf)
            for cid, c in counts.items():
                class_counts[cid] += c
                overall_class_counts[cid] += c
                total_objects += c
                overall_objects += c

        overall_images += len(images)

        print(f"[{split.upper()}]")
        print(f"  Images: {len(images)}")
        print(f"  Objects: {total_objects}")

        for cid in sorted(class_counts):
            name = (
                class_names[cid]
                if class_names and cid < len(class_names)
                else f"class_{cid}"
            )
            print(f"    {name} ({cid}): {class_counts[cid]}")
        print()

    print("OVERALL")
    print(f"Total images: {overall_images}")
    print(f"Total objects: {overall_objects}")
    for cid in sorted(overall_class_counts):
        name = (
            class_names[cid]
            if class_names and cid < len(class_names)
            else f"class_{cid}"
        )
        print(f"  {name} ({cid}): {overall_class_counts[cid]}")


def merge_splits(source_path: Path, output_path: Path):
    (output_path / "images").mkdir(parents=True, exist_ok=True)
    (output_path / "labels").mkdir(parents=True, exist_ok=True)

    total_images = 0

    for split in SPLITS:
        split_path = source_path / split
        if not split_path.exists():
            continue

        images = get_images(split_path / "images")

        for img_path in images:
            filename = img_path.name
            name, ext = os.path.splitext(filename)
            dest_img = output_path / "images" / filename

            counter = 1
            while dest_img.exists():
                dest_img = output_path / "images" / f"{name}_{split}_{counter}{ext}"
                counter += 1

            shutil.copy(img_path, dest_img)

            label_src = split_path / "labels" / f"{name}.txt"
            if label_src.exists():
                label_dest = output_path / "labels" / (dest_img.stem + ".txt")
                shutil.copy(label_src, label_dest)

            total_images += 1

        print(f"{split}: {len(images)} images merged")

    print(f"\nMerge complete. Total images merged: {total_images}")


def stratified_split(data_dir: Path, output_dir: Path):
    random.seed(RANDOM_SEED)

    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    image_files = []
    for img in get_images(images_dir):
        label = labels_dir / f"{img.stem}.txt"
        if label.exists():
            image_files.append((img, label))

    if not image_files:
        raise RuntimeError("No image-label pairs found.")

    print(f"Total images: {len(image_files)}")

    image_class_counts = {}
    total_class_counts = Counter()
    total_objects = 0

    for img, lbl in image_files:
        counts = read_label_counts(lbl)
        image_class_counts[img] = counts
        for cls, c in counts.items():
            total_class_counts[cls] += c
            total_objects += c

    print("Total objects per class:")
    for cls, cnt in total_class_counts.items():
        print(f"Class {cls}: {cnt}")

    target_objects = {
        split: total_objects * SPLIT_RATIOS[split] for split in SPLIT_RATIOS
    }

    current_objects = {split: 0 for split in SPLIT_RATIOS}
    current_class_counts = {split: Counter() for split in SPLIT_RATIOS}
    assignments = {}

    random.shuffle(image_files)

    for img, lbl in image_files:
        img_counts = image_class_counts[img]
        img_obj_count = sum(img_counts.values())

        best_split = None
        best_score = float("inf")

        for split in SPLIT_RATIOS:
            if current_objects[split] >= target_objects[split] * 1.05:
                continue

            projected_obj = current_objects[split] + img_obj_count
            obj_score = abs(projected_obj - target_objects[split])

            class_score = 0
            for cls, c in img_counts.items():
                projected_cls = current_class_counts[split][cls] + c
                target_cls = total_class_counts[cls] * SPLIT_RATIOS[split]
                class_score += abs(projected_cls - target_cls)

            score = obj_score + class_score

            if score < best_score:
                best_score = score
                best_split = split

        if best_split is None:
            best_split = min(current_objects, key=current_objects.get)

        assignments[img] = best_split
        current_objects[best_split] += img_obj_count
        for cls, c in img_counts.items():
            current_class_counts[best_split][cls] += c

    # Create folders
    for split in SPLIT_RATIOS:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Copy
    for img, lbl in image_files:
        split = assignments[img]
        shutil.copy(img, output_dir / split / "images" / img.name)
        shutil.copy(lbl, output_dir / split / "labels" / lbl.name)

    print("\nFinal object distribution:")
    for split in SPLIT_RATIOS:
        print(f"\n{split.upper()}")
        for cls in sorted(total_class_counts):
            actual = current_class_counts[split][cls]
            total = total_class_counts[cls]
            print(f"Class {cls}: {actual} ({actual / total:.2%})")


if __name__ == "__main__":
    # Choose ONE of these:

    # 1) Print dataset statistics
    # dataset_statistics(DATASET_PATH)

    # 2) Merge train/val/test into flat dataset
    # merge_splits(DATASET_PATH, DATASET_PATH)

    # 3) Perform stratified object-aware split
    # stratified_split(DATASET_PATH, OUTPUT_PATH)

    stratified_split(DATASET_PATH, SPLIT_PATH)
