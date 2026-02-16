from pathlib import Path
import cv2

INPUT_DIR = Path("data_split") / "train"
OUTPUT_DIR = Path("data_resized_split") / "train"
BUFFER_RATIO = 0.15

IMG_DIR = INPUT_DIR / "images"
LBL_DIR = INPUT_DIR / "labels"

OUT_IMG_DIR = OUTPUT_DIR / "images"
OUT_LBL_DIR = OUTPUT_DIR / "labels"

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)


def load_polygons(label_path, img_w, img_h):

    polygons = []

    with open(label_path, "r") as f:
        for line in f:
            parts = list(map(float, line.strip().split()))

            cls = int(parts[0])
            coords = parts[1:]

            pts = []

            for i in range(0, len(coords), 2):
                x = coords[i] * img_w
                y = coords[i + 1] * img_h

                pts.append([x, y])

            polygons.append((cls, pts))

    return polygons


def save_polygons(label_path, polygons, crop_x1, crop_y1, crop_w, crop_h):

    with open(label_path, "w") as f:
        for cls, pts in polygons:
            new_pts = []

            for x, y in pts:
                x -= crop_x1
                y -= crop_y1

                x = max(0, min(crop_w, x))
                y = max(0, min(crop_h, y))

                new_pts.append((x / crop_w, y / crop_h))

            line = str(cls)

            for x, y in new_pts:
                line += f" {x:.6f} {y:.6f}"

            f.write(line + "\n")


for img_path in IMG_DIR.glob("*"):
    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    label_path = LBL_DIR / (img_path.stem + ".txt")

    if not label_path.exists():
        continue

    img = cv2.imread(str(img_path))
    img_h, img_w = img.shape[:2]

    polygons = load_polygons(label_path, img_w, img_h)

    if len(polygons) == 0:
        continue

    xs = []
    ys = []

    for _, pts in polygons:
        for x, y in pts:
            xs.append(x)
            ys.append(y)

    buffer = int(BUFFER_RATIO * min(img_w, img_h))

    x1 = int(max(0, min(xs) - buffer))
    y1 = int(max(0, min(ys) - buffer))
    x2 = int(min(img_w, max(xs) + buffer))
    y2 = int(min(img_h, max(ys) + buffer))

    crop = img[y1:y2, x1:x2]

    crop_h, crop_w = crop.shape[:2]

    cv2.imwrite(str(OUT_IMG_DIR / img_path.name), crop)

    save_polygons(OUT_LBL_DIR / label_path.name, polygons, x1, y1, crop_w, crop_h)

print("Done.")
