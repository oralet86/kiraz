from pathlib import Path
import cv2
import numpy as np
from collections import Counter

DATASET_DIR = Path("data_split")
SPLITS = ["train", "val", "test"]

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


def collect_image_paths(dataset_dir: Path):
    image_paths = []
    for split in SPLITS:
        images_dir = dataset_dir / split / "images"
        if not images_dir.exists():
            continue
        for p in images_dir.rglob("*"):
            if p.suffix.lower() in IMAGE_EXTS:
                image_paths.append(p)
    return image_paths


def analyze_resolutions(image_paths):
    widths = []
    heights = []
    short_sides = []
    long_sides = []
    exact_resolutions = Counter()

    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue

        h, w = img.shape[:2]
        widths.append(w)
        heights.append(h)
        short_sides.append(min(w, h))
        long_sides.append(max(w, h))
        exact_resolutions[(w, h)] += 1

    return widths, heights, short_sides, long_sides, exact_resolutions


def print_stats(name, values):
    values = np.array(values)
    print(f"\n{name}:")
    print(f"  min:   {values.min():.1f}")
    print(f"  max:   {values.max():.1f}")
    print(f"  mean:  {values.mean():.1f}")
    print(f"  median:{np.median(values):.1f}")


def print_buckets(short_sides):
    buckets = {
        "<512": 0,
        "512–639": 0,
        "640–767": 0,
        "768–1023": 0,
        "1024+": 0,
    }

    for s in short_sides:
        if s < 512:
            buckets["<512"] += 1
        elif s < 640:
            buckets["512–639"] += 1
        elif s < 768:
            buckets["640–767"] += 1
        elif s < 1024:
            buckets["768–1023"] += 1
        else:
            buckets["1024+"] += 1

    print("\nShort-side distribution:")
    total = len(short_sides)
    for k, v in buckets.items():
        print(f"  {k:10s}: {v:4d} ({v/total:.1%})")


def main():
    image_paths = collect_image_paths(DATASET_DIR)

    if not image_paths:
        print("No images found.")
        return

    print(f"Total images analyzed: {len(image_paths)}")

    widths, heights, short_sides, long_sides, exact_resolutions = analyze_resolutions(image_paths)

    print_stats("Width", widths)
    print_stats("Height", heights)
    print_stats("Short side", short_sides)
    print_stats("Long side", long_sides)

    print_buckets(short_sides)

    print("\nTop 10 most common exact resolutions:")
    for (w, h), count in exact_resolutions.most_common(10):
        print(f"  {w}x{h}: {count}")


if __name__ == "__main__":
    main()
