#!/bin/bash

echo "Training all models..."
source ./.venv/bin/activate

# Detection models - seed 42
python train.py --mode detect --model yolov8n.pt --seed 42 --batch-mult 1.0
python train.py --mode detect --model yolov8s.pt --seed 42 --batch-mult 0.7
python train.py --mode detect --model yolov8m.pt --seed 42 --batch-mult 0.4
python train.py --mode detect --model yolo11n.pt --seed 42 --batch-mult 1.0
python train.py --mode detect --model yolo11s.pt --seed 42 --batch-mult 0.7
python train.py --mode detect --model yolo11m.pt --seed 42 --batch-mult 0.4
python train.py --mode detect --model yolov9t.pt --seed 42 --batch-mult 0.7
python train.py --mode detect --model yolov9s.pt --seed 42 --batch-mult 0.4
python train.py --mode detect --model yolov9m.pt --seed 42 --batch-mult 0.4
python train.py --mode detect --model yolo12n.pt --seed 42 --batch-mult 1.0
python train.py --mode detect --model yolo12s.pt --seed 42 --batch-mult 0.7
python train.py --mode detect --model yolo12m.pt --seed 42 --batch-mult 0.4
python train.py --mode detect --model yolov10n.pt --seed 42 --batch-mult 1.0
python train.py --mode detect --model yolov10s.pt --seed 42 --batch-mult 0.7
python train.py --mode detect --model yolov10m.pt --seed 42 --batch-mult 0.4
python train.py --mode detect --model yolo26n.pt --seed 42 --batch-mult 1.0
python train.py --mode detect --model yolo26s.pt --seed 42 --batch-mult 0.7
python train.py --mode detect --model yolo26m.pt --seed 42 --batch-mult 0.4

# Detection models - seed 737
python train.py --mode detect --model yolov8n.pt --seed 737 --batch-mult 1.0
python train.py --mode detect --model yolov8s.pt --seed 737 --batch-mult 0.7
python train.py --mode detect --model yolov8m.pt --seed 737 --batch-mult 0.4
python train.py --mode detect --model yolo11n.pt --seed 737 --batch-mult 1.0
python train.py --mode detect --model yolo11s.pt --seed 737 --batch-mult 0.7
python train.py --mode detect --model yolo11m.pt --seed 737 --batch-mult 0.4
python train.py --mode detect --model yolov9t.pt --seed 737 --batch-mult 0.7
python train.py --mode detect --model yolov9s.pt --seed 737 --batch-mult 0.4
python train.py --mode detect --model yolov9m.pt --seed 737 --batch-mult 0.4
python train.py --mode detect --model yolo12n.pt --seed 737 --batch-mult 1.0
python train.py --mode detect --model yolo12s.pt --seed 737 --batch-mult 0.7
python train.py --mode detect --model yolo12m.pt --seed 737 --batch-mult 0.4
python train.py --mode detect --model yolov10n.pt --seed 737 --batch-mult 1.0
python train.py --mode detect --model yolov10s.pt --seed 737 --batch-mult 0.7
python train.py --mode detect --model yolov10m.pt --seed 737 --batch-mult 0.4
python train.py --mode detect --model yolo26n.pt --seed 737 --batch-mult 1.0
python train.py --mode detect --model yolo26s.pt --seed 737 --batch-mult 0.7
python train.py --mode detect --model yolo26m.pt --seed 737 --batch-mult 0.4

# Detection models - seed 1573
python train.py --mode detect --model yolov8n.pt --seed 1573 --batch-mult 1.0
python train.py --mode detect --model yolov8s.pt --seed 1573 --batch-mult 0.7
python train.py --mode detect --model yolov8m.pt --seed 1573 --batch-mult 0.4
python train.py --mode detect --model yolo11n.pt --seed 1573 --batch-mult 1.0
python train.py --mode detect --model yolo11s.pt --seed 1573 --batch-mult 0.7
python train.py --mode detect --model yolo11m.pt --seed 1573 --batch-mult 0.4
python train.py --mode detect --model yolov9t.pt --seed 1573 --batch-mult 0.7
python train.py --mode detect --model yolov9s.pt --seed 1573 --batch-mult 0.4
python train.py --mode detect --model yolov9m.pt --seed 1573 --batch-mult 0.4
python train.py --mode detect --model yolo12n.pt --seed 1573 --batch-mult 1.0
python train.py --mode detect --model yolo12s.pt --seed 1573 --batch-mult 0.7
python train.py --mode detect --model yolo12m.pt --seed 1573 --batch-mult 0.4
python train.py --mode detect --model yolov10n.pt --seed 1573 --batch-mult 1.0
python train.py --mode detect --model yolov10s.pt --seed 1573 --batch-mult 0.7
python train.py --mode detect --model yolov10m.pt --seed 1573 --batch-mult 0.4
python train.py --mode detect --model yolo26n.pt --seed 1573 --batch-mult 1.0
python train.py --mode detect --model yolo26s.pt --seed 1573 --batch-mult 0.7
python train.py --mode detect --model yolo26m.pt --seed 1573 --batch-mult 0.4

# Detection models - seed 2468
python train.py --mode detect --model yolov8n.pt --seed 2468 --batch-mult 1.0
python train.py --mode detect --model yolov8s.pt --seed 2468 --batch-mult 0.7
python train.py --mode detect --model yolov8m.pt --seed 2468 --batch-mult 0.4
python train.py --mode detect --model yolo11n.pt --seed 2468 --batch-mult 1.0
python train.py --mode detect --model yolo11s.pt --seed 2468 --batch-mult 0.7
python train.py --mode detect --model yolo11m.pt --seed 2468 --batch-mult 0.4
python train.py --mode detect --model yolov9t.pt --seed 2468 --batch-mult 0.7
python train.py --mode detect --model yolov9s.pt --seed 2468 --batch-mult 0.4
python train.py --mode detect --model yolov9m.pt --seed 2468 --batch-mult 0.4
python train.py --mode detect --model yolo12n.pt --seed 2468 --batch-mult 1.0
python train.py --mode detect --model yolo12s.pt --seed 2468 --batch-mult 0.7
python train.py --mode detect --model yolo12m.pt --seed 2468 --batch-mult 0.4
python train.py --mode detect --model yolov10n.pt --seed 2468 --batch-mult 1.0
python train.py --mode detect --model yolov10s.pt --seed 2468 --batch-mult 0.7
python train.py --mode detect --model yolov10m.pt --seed 2468 --batch-mult 0.4
python train.py --mode detect --model yolo26n.pt --seed 2468 --batch-mult 1.0
python train.py --mode detect --model yolo26s.pt --seed 2468 --batch-mult 0.7
python train.py --mode detect --model yolo26m.pt --seed 2468 --batch-mult 0.4

# Detection models - seed 3456
python train.py --mode detect --model yolov8n.pt --seed 3456 --batch-mult 1.0
python train.py --mode detect --model yolov8s.pt --seed 3456 --batch-mult 0.7
python train.py --mode detect --model yolov8m.pt --seed 3456 --batch-mult 0.4
python train.py --mode detect --model yolo11n.pt --seed 3456 --batch-mult 1.0
python train.py --mode detect --model yolo11s.pt --seed 3456 --batch-mult 0.7
python train.py --mode detect --model yolo11m.pt --seed 3456 --batch-mult 0.4
python train.py --mode detect --model yolov9t.pt --seed 3456 --batch-mult 0.7
python train.py --mode detect --model yolov9s.pt --seed 3456 --batch-mult 0.4
python train.py --mode detect --model yolov9m.pt --seed 3456 --batch-mult 0.4
python train.py --mode detect --model yolo12n.pt --seed 3456 --batch-mult 1.0
python train.py --mode detect --model yolo12s.pt --seed 3456 --batch-mult 0.7
python train.py --mode detect --model yolo12m.pt --seed 3456 --batch-mult 0.4
python train.py --mode detect --model yolov10n.pt --seed 3456 --batch-mult 1.0
python train.py --mode detect --model yolov10s.pt --seed 3456 --batch-mult 0.7
python train.py --mode detect --model yolov10m.pt --seed 3456 --batch-mult 0.4
python train.py --mode detect --model yolo26n.pt --seed 3456 --batch-mult 1.0
python train.py --mode detect --model yolo26s.pt --seed 3456 --batch-mult 0.7
python train.py --mode detect --model yolo26m.pt --seed 3456 --batch-mult 0.4

# Classification models - seed 42
python train.py --mode cls --model yolov8n-cls.pt --seed 42 --batch-mult 1.0
python train.py --mode cls --model yolov8s-cls.pt --seed 42 --batch-mult 0.7
python train.py --mode cls --model yolov8m-cls.pt --seed 42 --batch-mult 0.4
python train.py --mode cls --model yolov8l-cls.pt --seed 42 --batch-mult 0.2
python train.py --mode cls --model yolo11n-cls.pt --seed 42 --batch-mult 1.0
python train.py --mode cls --model yolo11s-cls.pt --seed 42 --batch-mult 0.7
python train.py --mode cls --model yolo11m-cls.pt --seed 42 --batch-mult 0.4
python train.py --mode cls --model yolo11l-cls.pt --seed 42 --batch-mult 0.2
python train.py --mode cls --model yolo26n-cls.pt --seed 42 --batch-mult 1.0
python train.py --mode cls --model yolo26s-cls.pt --seed 42 --batch-mult 0.7
python train.py --mode cls --model yolo26m-cls.pt --seed 42 --batch-mult 0.4
python train.py --mode cls --model yolo26l-cls.pt --seed 42 --batch-mult 0.2

# Classification models - seed 737
python train.py --mode cls --model yolov8n-cls.pt --seed 737 --batch-mult 1.0
python train.py --mode cls --model yolov8s-cls.pt --seed 737 --batch-mult 0.7
python train.py --mode cls --model yolov8m-cls.pt --seed 737 --batch-mult 0.4
python train.py --mode cls --model yolov8l-cls.pt --seed 737 --batch-mult 0.2
python train.py --mode cls --model yolo11n-cls.pt --seed 737 --batch-mult 1.0
python train.py --mode cls --model yolo11s-cls.pt --seed 737 --batch-mult 0.7
python train.py --mode cls --model yolo11m-cls.pt --seed 737 --batch-mult 0.4
python train.py --mode cls --model yolo11l-cls.pt --seed 737 --batch-mult 0.2
python train.py --mode cls --model yolo26n-cls.pt --seed 737 --batch-mult 1.0
python train.py --mode cls --model yolo26s-cls.pt --seed 737 --batch-mult 0.7
python train.py --mode cls --model yolo26m-cls.pt --seed 737 --batch-mult 0.4
python train.py --mode cls --model yolo26l-cls.pt --seed 737 --batch-mult 0.2

# Classification models - seed 1573
python train.py --mode cls --model yolov8n-cls.pt --seed 1573 --batch-mult 1.0
python train.py --mode cls --model yolov8s-cls.pt --seed 1573 --batch-mult 0.7
python train.py --mode cls --model yolov8m-cls.pt --seed 1573 --batch-mult 0.4
python train.py --mode cls --model yolov8l-cls.pt --seed 1573 --batch-mult 0.2
python train.py --mode cls --model yolo11n-cls.pt --seed 1573 --batch-mult 1.0
python train.py --mode cls --model yolo11s-cls.pt --seed 1573 --batch-mult 0.7
python train.py --mode cls --model yolo11m-cls.pt --seed 1573 --batch-mult 0.4
python train.py --mode cls --model yolo11l-cls.pt --seed 1573 --batch-mult 0.2
python train.py --mode cls --model yolo26n-cls.pt --seed 1573 --batch-mult 1.0
python train.py --mode cls --model yolo26s-cls.pt --seed 1573 --batch-mult 0.7
python train.py --mode cls --model yolo26m-cls.pt --seed 1573 --batch-mult 0.4
python train.py --mode cls --model yolo26l-cls.pt --seed 1573 --batch-mult 0.2

# Classification models - seed 2468
python train.py --mode cls --model yolov8n-cls.pt --seed 2468 --batch-mult 1.0
python train.py --mode cls --model yolov8s-cls.pt --seed 2468 --batch-mult 0.7
python train.py --mode cls --model yolov8m-cls.pt --seed 2468 --batch-mult 0.4
python train.py --mode cls --model yolov8l-cls.pt --seed 2468 --batch-mult 0.2
python train.py --mode cls --model yolo11n-cls.pt --seed 2468 --batch-mult 1.0
python train.py --mode cls --model yolo11s-cls.pt --seed 2468 --batch-mult 0.7
python train.py --mode cls --model yolo11m-cls.pt --seed 2468 --batch-mult 0.4
python train.py --mode cls --model yolo11l-cls.pt --seed 2468 --batch-mult 0.2
python train.py --mode cls --model yolo26n-cls.pt --seed 2468 --batch-mult 1.0
python train.py --mode cls --model yolo26s-cls.pt --seed 2468 --batch-mult 0.7
python train.py --mode cls --model yolo26m-cls.pt --seed 2468 --batch-mult 0.4
python train.py --mode cls --model yolo26l-cls.pt --seed 2468 --batch-mult 0.2

# Classification models - seed 3456
python train.py --mode cls --model yolov8n-cls.pt --seed 3456 --batch-mult 1.0
python train.py --mode cls --model yolov8s-cls.pt --seed 3456 --batch-mult 0.7
python train.py --mode cls --model yolov8m-cls.pt --seed 3456 --batch-mult 0.4
python train.py --mode cls --model yolov8l-cls.pt --seed 3456 --batch-mult 0.2
python train.py --mode cls --model yolo11n-cls.pt --seed 3456 --batch-mult 1.0
python train.py --mode cls --model yolo11s-cls.pt --seed 3456 --batch-mult 0.7
python train.py --mode cls --model yolo11m-cls.pt --seed 3456 --batch-mult 0.4
python train.py --mode cls --model yolo11l-cls.pt --seed 3456 --batch-mult 0.2
python train.py --mode cls --model yolo26n-cls.pt --seed 3456 --batch-mult 1.0
python train.py --mode cls --model yolo26s-cls.pt --seed 3456 --batch-mult 0.7
python train.py --mode cls --model yolo26m-cls.pt --seed 3456 --batch-mult 0.4
python train.py --mode cls --model yolo26l-cls.pt --seed 3456 --batch-mult 0.2

echo "All models trained successfully!"
echo "Check the results/ directory for training results."
