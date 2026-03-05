import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

IMAGE_PATH = "data/images/275_jpg.rf.ea21c909d7b3158aa6ee9ec3e45c55d6.jpg"


def main():
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError("Görüntü bulunamadı")

    img = cv2.imread(IMAGE_PATH)

    B, G, R = cv2.split(img)

    # R-G farkı
    diff = cv2.subtract(R, G)

    # Otsu
    t, otsu = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    print("Otsu threshold (R-G):", t)

    # Largest component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(otsu)

    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.zeros_like(otsu)
        mask[labels == largest_label] = 255
    else:
        mask = otsu

    result = cv2.bitwise_and(img, img, mask=mask)

    # BGR -> RGB dönüşüm (matplotlib için)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # 2x2 grid
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Orijinal RGB Görüntü")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(diff, cmap="gray")
    plt.title("Red - Green Kanal Farkı")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(mask, cmap="gray")
    plt.title("En Büyük Birleşik Maske")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(result_rgb)
    plt.title("Maske Uygulanmış Segmentasyon")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
