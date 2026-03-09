#!/bin/bash

echo "Training all models..."
source ./.venv/bin/activate

# YOLO detection models - yolov8n.pt
python train.py --mode detect --model yolov8n.pt --seed 42 --batch-mult 1.0
python train.py --mode detect --model yolov8n.pt --seed 737 --batch-mult 1.0
python train.py --mode detect --model yolov8n.pt --seed 1573 --batch-mult 1.0
python train.py --mode detect --model yolov8n.pt --seed 2468 --batch-mult 1.0
python train.py --mode detect --model yolov8n.pt --seed 3456 --batch-mult 1.0

# YOLO detection models - yolov8s.pt
python train.py --mode detect --model yolov8s.pt --seed 42 --batch-mult 0.7
python train.py --mode detect --model yolov8s.pt --seed 737 --batch-mult 0.7
python train.py --mode detect --model yolov8s.pt --seed 1573 --batch-mult 0.7
python train.py --mode detect --model yolov8s.pt --seed 2468 --batch-mult 0.7
python train.py --mode detect --model yolov8s.pt --seed 3456 --batch-mult 0.7

# YOLO detection models - yolov8m.pt
python train.py --mode detect --model yolov8m.pt --seed 42 --batch-mult 0.4
python train.py --mode detect --model yolov8m.pt --seed 737 --batch-mult 0.4
python train.py --mode detect --model yolov8m.pt --seed 1573 --batch-mult 0.4
python train.py --mode detect --model yolov8m.pt --seed 2468 --batch-mult 0.4
python train.py --mode detect --model yolov8m.pt --seed 3456 --batch-mult 0.4

# YOLO detection models - yolo11n.pt
python train.py --mode detect --model yolo11n.pt --seed 42 --batch-mult 1.0
python train.py --mode detect --model yolo11n.pt --seed 737 --batch-mult 1.0
python train.py --mode detect --model yolo11n.pt --seed 1573 --batch-mult 1.0
python train.py --mode detect --model yolo11n.pt --seed 2468 --batch-mult 1.0
python train.py --mode detect --model yolo11n.pt --seed 3456 --batch-mult 1.0

# YOLO detection models - yolo11s.pt
python train.py --mode detect --model yolo11s.pt --seed 42 --batch-mult 0.7
python train.py --mode detect --model yolo11s.pt --seed 737 --batch-mult 0.7
python train.py --mode detect --model yolo11s.pt --seed 1573 --batch-mult 0.7
python train.py --mode detect --model yolo11s.pt --seed 2468 --batch-mult 0.7
python train.py --mode detect --model yolo11s.pt --seed 3456 --batch-mult 0.7

# YOLO detection models - yolo11m.pt
python train.py --mode detect --model yolo11m.pt --seed 42 --batch-mult 0.4
python train.py --mode detect --model yolo11m.pt --seed 737 --batch-mult 0.4
python train.py --mode detect --model yolo11m.pt --seed 1573 --batch-mult 0.4
python train.py --mode detect --model yolo11m.pt --seed 2468 --batch-mult 0.4
python train.py --mode detect --model yolo11m.pt --seed 3456 --batch-mult 0.4

# YOLO detection models - yolov9t.pt
python train.py --mode detect --model yolov9t.pt --seed 42 --batch-mult 1.0
python train.py --mode detect --model yolov9t.pt --seed 737 --batch-mult 1.0
python train.py --mode detect --model yolov9t.pt --seed 1573 --batch-mult 1.0
python train.py --mode detect --model yolov9t.pt --seed 2468 --batch-mult 1.0
python train.py --mode detect --model yolov9t.pt --seed 3456 --batch-mult 1.0

# YOLO detection models - yolov9s.pt
python train.py --mode detect --model yolov9s.pt --seed 42 --batch-mult 0.7
python train.py --mode detect --model yolov9s.pt --seed 737 --batch-mult 0.7
python train.py --mode detect --model yolov9s.pt --seed 1573 --batch-mult 0.7
python train.py --mode detect --model yolov9s.pt --seed 2468 --batch-mult 0.7
python train.py --mode detect --model yolov9s.pt --seed 3456 --batch-mult 0.7

# YOLO detection models - yolov9m.pt
python train.py --mode detect --model yolov9m.pt --seed 42 --batch-mult 0.4
python train.py --mode detect --model yolov9m.pt --seed 737 --batch-mult 0.4
python train.py --mode detect --model yolov9m.pt --seed 1573 --batch-mult 0.4
python train.py --mode detect --model yolov9m.pt --seed 2468 --batch-mult 0.4
python train.py --mode detect --model yolov9m.pt --seed 3456 --batch-mult 0.4

# YOLO detection models - yolo12n.pt
python train.py --mode detect --model yolo12n.pt --seed 42 --batch-mult 1.0
python train.py --mode detect --model yolo12n.pt --seed 737 --batch-mult 1.0
python train.py --mode detect --model yolo12n.pt --seed 1573 --batch-mult 1.0
python train.py --mode detect --model yolo12n.pt --seed 2468 --batch-mult 1.0
python train.py --mode detect --model yolo12n.pt --seed 3456 --batch-mult 1.0

# YOLO detection models - yolo12s.pt
python train.py --mode detect --model yolo12s.pt --seed 42 --batch-mult 0.7
python train.py --mode detect --model yolo12s.pt --seed 737 --batch-mult 0.7
python train.py --mode detect --model yolo12s.pt --seed 1573 --batch-mult 0.7
python train.py --mode detect --model yolo12s.pt --seed 2468 --batch-mult 0.7
python train.py --mode detect --model yolo12s.pt --seed 3456 --batch-mult 0.7

# YOLO detection models - yolo12m.pt
python train.py --mode detect --model yolo12m.pt --seed 42 --batch-mult 0.4
python train.py --mode detect --model yolo12m.pt --seed 737 --batch-mult 0.4
python train.py --mode detect --model yolo12m.pt --seed 1573 --batch-mult 0.4
python train.py --mode detect --model yolo12m.pt --seed 2468 --batch-mult 0.4
python train.py --mode detect --model yolo12m.pt --seed 3456 --batch-mult 0.4

# YOLO detection models - yolov10n.pt
python train.py --mode detect --model yolov10n.pt --seed 42 --batch-mult 1.0
python train.py --mode detect --model yolov10n.pt --seed 737 --batch-mult 1.0
python train.py --mode detect --model yolov10n.pt --seed 1573 --batch-mult 1.0
python train.py --mode detect --model yolov10n.pt --seed 2468 --batch-mult 1.0
python train.py --mode detect --model yolov10n.pt --seed 3456 --batch-mult 1.0

# YOLO detection models - yolov10s.pt
python train.py --mode detect --model yolov10s.pt --seed 42 --batch-mult 0.7
python train.py --mode detect --model yolov10s.pt --seed 737 --batch-mult 0.7
python train.py --mode detect --model yolov10s.pt --seed 1573 --batch-mult 0.7
python train.py --mode detect --model yolov10s.pt --seed 2468 --batch-mult 0.7
python train.py --mode detect --model yolov10s.pt --seed 3456 --batch-mult 0.7

# YOLO detection models - yolov10m.pt
python train.py --mode detect --model yolov10m.pt --seed 42 --batch-mult 0.4
python train.py --mode detect --model yolov10m.pt --seed 737 --batch-mult 0.4
python train.py --mode detect --model yolov10m.pt --seed 1573 --batch-mult 0.4
python train.py --mode detect --model yolov10m.pt --seed 2468 --batch-mult 0.4
python train.py --mode detect --model yolov10m.pt --seed 3456 --batch-mult 0.4

# YOLO detection models - yolo26n.pt
python train.py --mode detect --model yolo26n.pt --seed 42 --batch-mult 1.0
python train.py --mode detect --model yolo26n.pt --seed 737 --batch-mult 1.0
python train.py --mode detect --model yolo26n.pt --seed 1573 --batch-mult 1.0
python train.py --mode detect --model yolo26n.pt --seed 2468 --batch-mult 1.0
python train.py --mode detect --model yolo26n.pt --seed 3456 --batch-mult 1.0

# YOLO detection models - yolo26s.pt
python train.py --mode detect --model yolo26s.pt --seed 42 --batch-mult 0.7
python train.py --mode detect --model yolo26s.pt --seed 737 --batch-mult 0.7
python train.py --mode detect --model yolo26s.pt --seed 1573 --batch-mult 0.7
python train.py --mode detect --model yolo26s.pt --seed 2468 --batch-mult 0.7
python train.py --mode detect --model yolo26s.pt --seed 3456 --batch-mult 0.7

# YOLO detection models - yolo26m.pt
python train.py --mode detect --model yolo26m.pt --seed 42 --batch-mult 0.4
python train.py --mode detect --model yolo26m.pt --seed 737 --batch-mult 0.4
python train.py --mode detect --model yolo26m.pt --seed 1573 --batch-mult 0.4
python train.py --mode detect --model yolo26m.pt --seed 2468 --batch-mult 0.4
python train.py --mode detect --model yolo26m.pt --seed 3456 --batch-mult 0.4

# YOLO classification models - yolov8n-cls.pt
python train.py --mode cls --model yolov8n-cls.pt --seed 42 --batch-mult 1.0
python train.py --mode cls --model yolov8n-cls.pt --seed 737 --batch-mult 1.0
python train.py --mode cls --model yolov8n-cls.pt --seed 1573 --batch-mult 1.0
python train.py --mode cls --model yolov8n-cls.pt --seed 2468 --batch-mult 1.0
python train.py --mode cls --model yolov8n-cls.pt --seed 3456 --batch-mult 1.0

# YOLO classification models - yolov8s-cls.pt
python train.py --mode cls --model yolov8s-cls.pt --seed 42 --batch-mult 0.7
python train.py --mode cls --model yolov8s-cls.pt --seed 737 --batch-mult 0.7
python train.py --mode cls --model yolov8s-cls.pt --seed 1573 --batch-mult 0.7
python train.py --mode cls --model yolov8s-cls.pt --seed 2468 --batch-mult 0.7
python train.py --mode cls --model yolov8s-cls.pt --seed 3456 --batch-mult 0.7

# YOLO classification models - yolov8m-cls.pt
python train.py --mode cls --model yolov8m-cls.pt --seed 42 --batch-mult 0.4
python train.py --mode cls --model yolov8m-cls.pt --seed 737 --batch-mult 0.4
python train.py --mode cls --model yolov8m-cls.pt --seed 1573 --batch-mult 0.4
python train.py --mode cls --model yolov8m-cls.pt --seed 2468 --batch-mult 0.4
python train.py --mode cls --model yolov8m-cls.pt --seed 3456 --batch-mult 0.4

# YOLO classification models - yolov8l-cls.pt
python train.py --mode cls --model yolov8l-cls.pt --seed 42 --batch-mult 0.2
python train.py --mode cls --model yolov8l-cls.pt --seed 737 --batch-mult 0.2
python train.py --mode cls --model yolov8l-cls.pt --seed 1573 --batch-mult 0.2
python train.py --mode cls --model yolov8l-cls.pt --seed 2468 --batch-mult 0.2
python train.py --mode cls --model yolov8l-cls.pt --seed 3456 --batch-mult 0.2

# YOLO classification models - yolo11n-cls.pt
python train.py --mode cls --model yolo11n-cls.pt --seed 42 --batch-mult 1.0
python train.py --mode cls --model yolo11n-cls.pt --seed 737 --batch-mult 1.0
python train.py --mode cls --model yolo11n-cls.pt --seed 1573 --batch-mult 1.0
python train.py --mode cls --model yolo11n-cls.pt --seed 2468 --batch-mult 1.0
python train.py --mode cls --model yolo11n-cls.pt --seed 3456 --batch-mult 1.0

# YOLO classification models - yolo11s-cls.pt
python train.py --mode cls --model yolo11s-cls.pt --seed 42 --batch-mult 0.7
python train.py --mode cls --model yolo11s-cls.pt --seed 737 --batch-mult 0.7
python train.py --mode cls --model yolo11s-cls.pt --seed 1573 --batch-mult 0.7
python train.py --mode cls --model yolo11s-cls.pt --seed 2468 --batch-mult 0.7
python train.py --mode cls --model yolo11s-cls.pt --seed 3456 --batch-mult 0.7

# YOLO classification models - yolo11m-cls.pt
python train.py --mode cls --model yolo11m-cls.pt --seed 42 --batch-mult 0.4
python train.py --mode cls --model yolo11m-cls.pt --seed 737 --batch-mult 0.4
python train.py --mode cls --model yolo11m-cls.pt --seed 1573 --batch-mult 0.4
python train.py --mode cls --model yolo11m-cls.pt --seed 2468 --batch-mult 0.4
python train.py --mode cls --model yolo11m-cls.pt --seed 3456 --batch-mult 0.4

# YOLO classification models - yolo11l-cls.pt
python train.py --mode cls --model yolo11l-cls.pt --seed 42 --batch-mult 0.2
python train.py --mode cls --model yolo11l-cls.pt --seed 737 --batch-mult 0.2
python train.py --mode cls --model yolo11l-cls.pt --seed 1573 --batch-mult 0.2
python train.py --mode cls --model yolo11l-cls.pt --seed 2468 --batch-mult 0.2
python train.py --mode cls --model yolo11l-cls.pt --seed 3456 --batch-mult 0.2

# YOLO classification models - yolo26n-cls.pt
python train.py --mode cls --model yolo26n-cls.pt --seed 42 --batch-mult 1.0
python train.py --mode cls --model yolo26n-cls.pt --seed 737 --batch-mult 1.0
python train.py --mode cls --model yolo26n-cls.pt --seed 1573 --batch-mult 1.0
python train.py --mode cls --model yolo26n-cls.pt --seed 2468 --batch-mult 1.0
python train.py --mode cls --model yolo26n-cls.pt --seed 3456 --batch-mult 1.0

# YOLO classification models - yolo26s-cls.pt
python train.py --mode cls --model yolo26s-cls.pt --seed 42 --batch-mult 0.7
python train.py --mode cls --model yolo26s-cls.pt --seed 737 --batch-mult 0.7
python train.py --mode cls --model yolo26s-cls.pt --seed 1573 --batch-mult 0.7
python train.py --mode cls --model yolo26s-cls.pt --seed 2468 --batch-mult 0.7
python train.py --mode cls --model yolo26s-cls.pt --seed 3456 --batch-mult 0.7

# YOLO classification models - yolo26m-cls.pt
python train.py --mode cls --model yolo26m-cls.pt --seed 42 --batch-mult 0.4
python train.py --mode cls --model yolo26m-cls.pt --seed 737 --batch-mult 0.4
python train.py --mode cls --model yolo26m-cls.pt --seed 1573 --batch-mult 0.4
python train.py --mode cls --model yolo26m-cls.pt --seed 2468 --batch-mult 0.4
python train.py --mode cls --model yolo26m-cls.pt --seed 3456 --batch-mult 0.4

# YOLO classification models - yolo26l-cls.pt
python train.py --mode cls --model yolo26l-cls.pt --seed 42 --batch-mult 0.2
python train.py --mode cls --model yolo26l-cls.pt --seed 737 --batch-mult 0.2
python train.py --mode cls --model yolo26l-cls.pt --seed 1573 --batch-mult 0.2
python train.py --mode cls --model yolo26l-cls.pt --seed 2468 --batch-mult 0.2
python train.py --mode cls --model yolo26l-cls.pt --seed 3456 --batch-mult 0.2

# RT-DETR detection models - rtdetr-l.pt
python train.py --mode detect --model rtdetr-l.pt --seed 42 --batch-mult 0.4
python train.py --mode detect --model rtdetr-l.pt --seed 737 --batch-mult 0.4
python train.py --mode detect --model rtdetr-l.pt --seed 1573 --batch-mult 0.4
python train.py --mode detect --model rtdetr-l.pt --seed 2468 --batch-mult 0.4
python train.py --mode detect --model rtdetr-l.pt --seed 3456 --batch-mult 0.4

# RT-DETR detection models - rtdetr-x.pt
python train.py --mode detect --model rtdetr-x.pt --seed 42 --batch-mult 0.3
python train.py --mode detect --model rtdetr-x.pt --seed 737 --batch-mult 0.3
python train.py --mode detect --model rtdetr-x.pt --seed 1573 --batch-mult 0.3
python train.py --mode detect --model rtdetr-x.pt --seed 2468 --batch-mult 0.3
python train.py --mode detect --model rtdetr-x.pt --seed 3456 --batch-mult 0.3

# Faster R-CNN detection models - faster-rcnn-r50
python train_torch.py --mode detect --model faster-rcnn-r50  --seed 42 --batch-mult 0.5
python train_torch.py --mode detect --model faster-rcnn-r50  --seed 737 --batch-mult 0.5
python train_torch.py --mode detect --model faster-rcnn-r50  --seed 1573 --batch-mult 0.5
python train_torch.py --mode detect --model faster-rcnn-r50  --seed 2468 --batch-mult 0.5
python train_torch.py --mode detect --model faster-rcnn-r50  --seed 3456 --batch-mult 0.5

# Faster R-CNN detection models - faster-rcnn-r101
python train_torch.py --mode detect --model faster-rcnn-r101 --seed 42 --batch-mult 0.25
python train_torch.py --mode detect --model faster-rcnn-r101 --seed 737 --batch-mult 0.25
python train_torch.py --mode detect --model faster-rcnn-r101 --seed 1573 --batch-mult 0.25
python train_torch.py --mode detect --model faster-rcnn-r101 --seed 2468 --batch-mult 0.25
python train_torch.py --mode detect --model faster-rcnn-r101 --seed 3456 --batch-mult 0.25

# DETR detection models - detr-r50
python train_torch.py --mode detect --model detr-r50  --seed 42 --batch-mult 0.5
python train_torch.py --mode detect --model detr-r50  --seed 737 --batch-mult 0.5
python train_torch.py --mode detect --model detr-r50  --seed 1573 --batch-mult 0.5
python train_torch.py --mode detect --model detr-r50  --seed 2468 --batch-mult 0.5
python train_torch.py --mode detect --model detr-r50  --seed 3456 --batch-mult 0.5

# DETR detection models - detr-r101
python train_torch.py --mode detect --model detr-r101 --seed 42 --batch-mult 0.25
python train_torch.py --mode detect --model detr-r101 --seed 737 --batch-mult 0.25
python train_torch.py --mode detect --model detr-r101 --seed 1573 --batch-mult 0.25
python train_torch.py --mode detect --model detr-r101 --seed 2468 --batch-mult 0.25
python train_torch.py --mode detect --model detr-r101 --seed 3456 --batch-mult 0.25

# ResNet classification models - resnet50
python train_torch.py --mode cls --model resnet50  --seed 42 --batch-mult 0.7
python train_torch.py --mode cls --model resnet50  --seed 737 --batch-mult 0.7
python train_torch.py --mode cls --model resnet50  --seed 1573 --batch-mult 0.7
python train_torch.py --mode cls --model resnet50  --seed 2468 --batch-mult 0.7
python train_torch.py --mode cls --model resnet50  --seed 3456 --batch-mult 0.7

# ResNet classification models - resnet101
python train_torch.py --mode cls --model resnet101 --seed 42 --batch-mult 0.25
python train_torch.py --mode cls --model resnet101 --seed 737 --batch-mult 0.25
python train_torch.py --mode cls --model resnet101 --seed 1573 --batch-mult 0.25
python train_torch.py --mode cls --model resnet101 --seed 2468 --batch-mult 0.25
python train_torch.py --mode cls --model resnet101 --seed 3456 --batch-mult 0.25

# EfficientNet classification models - efficientnet-b0
python train_torch.py --mode cls --model efficientnet-b0 --seed 42 --batch-mult 1.0
python train_torch.py --mode cls --model efficientnet-b0 --seed 737 --batch-mult 1.0
python train_torch.py --mode cls --model efficientnet-b0 --seed 1573 --batch-mult 1.0
python train_torch.py --mode cls --model efficientnet-b0 --seed 2468 --batch-mult 1.0
python train_torch.py --mode cls --model efficientnet-b0 --seed 3456 --batch-mult 1.0

# EfficientNet classification models - efficientnet-b1
python train_torch.py --mode cls --model efficientnet-b1 --seed 42 --batch-mult 0.7
python train_torch.py --mode cls --model efficientnet-b1 --seed 737 --batch-mult 0.7
python train_torch.py --mode cls --model efficientnet-b1 --seed 1573 --batch-mult 0.7
python train_torch.py --mode cls --model efficientnet-b1 --seed 2468 --batch-mult 0.7
python train_torch.py --mode cls --model efficientnet-b1 --seed 3456 --batch-mult 0.7

# EfficientNet classification models - efficientnet-b2
python train_torch.py --mode cls --model efficientnet-b2 --seed 42 --batch-mult 0.5
python train_torch.py --mode cls --model efficientnet-b2 --seed 737 --batch-mult 0.5
python train_torch.py --mode cls --model efficientnet-b2 --seed 1573 --batch-mult 0.5
python train_torch.py --mode cls --model efficientnet-b2 --seed 2468 --batch-mult 0.5
python train_torch.py --mode cls --model efficientnet-b2 --seed 3456 --batch-mult 0.5

# EfficientNet classification models - efficientnet-b3
python train_torch.py --mode cls --model efficientnet-b3 --seed 42 --batch-mult 0.4
python train_torch.py --mode cls --model efficientnet-b3 --seed 737 --batch-mult 0.4
python train_torch.py --mode cls --model efficientnet-b3 --seed 1573 --batch-mult 0.4
python train_torch.py --mode cls --model efficientnet-b3 --seed 2468 --batch-mult 0.4
python train_torch.py --mode cls --model efficientnet-b3 --seed 3456 --batch-mult 0.4

# ConvNeXt classification models - convnext-tiny
python train_torch.py --mode cls --model convnext-tiny  --seed 42 --batch-mult 0.7
python train_torch.py --mode cls --model convnext-tiny  --seed 737 --batch-mult 0.7
python train_torch.py --mode cls --model convnext-tiny  --seed 1573 --batch-mult 0.7
python train_torch.py --mode cls --model convnext-tiny  --seed 2468 --batch-mult 0.7
python train_torch.py --mode cls --model convnext-tiny  --seed 3456 --batch-mult 0.7

# ConvNeXt classification models - convnext-small
python train_torch.py --mode cls --model convnext-small --seed 42 --batch-mult 0.5
python train_torch.py --mode cls --model convnext-small --seed 737 --batch-mult 0.5
python train_torch.py --mode cls --model convnext-small --seed 1573 --batch-mult 0.5
python train_torch.py --mode cls --model convnext-small --seed 2468 --batch-mult 0.5
python train_torch.py --mode cls --model convnext-small --seed 3456 --batch-mult 0.5

# ConvNeXt classification models - convnext-large
python train_torch.py --mode cls --model convnext-large --seed 42 --batch-mult 0.3
python train_torch.py --mode cls --model convnext-large --seed 737 --batch-mult 0.3
python train_torch.py --mode cls --model convnext-large --seed 1573 --batch-mult 0.3
python train_torch.py --mode cls --model convnext-large --seed 2468 --batch-mult 0.3
python train_torch.py --mode cls --model convnext-large --seed 3456 --batch-mult 0.3

# ViT classification models - vit-small
python train_torch.py --mode cls --model vit-small --seed 42 --batch-mult 0.6
python train_torch.py --mode cls --model vit-small --seed 737 --batch-mult 0.6
python train_torch.py --mode cls --model vit-small --seed 1573 --batch-mult 0.6
python train_torch.py --mode cls --model vit-small --seed 2468 --batch-mult 0.6
python train_torch.py --mode cls --model vit-small --seed 3456 --batch-mult 0.6

# ViT classification models - vit-base
python train_torch.py --mode cls --model vit-base  --seed 42 --batch-mult 0.4
python train_torch.py --mode cls --model vit-base  --seed 737 --batch-mult 0.4
python train_torch.py --mode cls --model vit-base  --seed 1573 --batch-mult 0.4
python train_torch.py --mode cls --model vit-base  --seed 2468 --batch-mult 0.4
python train_torch.py --mode cls --model vit-base  --seed 3456 --batch-mult 0.4

# DeiT classification models - deit-small
python train_torch.py --mode cls --model deit-small --seed 42 --batch-mult 0.6
python train_torch.py --mode cls --model deit-small --seed 737 --batch-mult 0.6
python train_torch.py --mode cls --model deit-small --seed 1573 --batch-mult 0.6
python train_torch.py --mode cls --model deit-small --seed 2468 --batch-mult 0.6
python train_torch.py --mode cls --model deit-small --seed 3456 --batch-mult 0.6

# DeiT classification models - deit-base
python train_torch.py --mode cls --model deit-base  --seed 42 --batch-mult 0.4
python train_torch.py --mode cls --model deit-base  --seed 737 --batch-mult 0.4
python train_torch.py --mode cls --model deit-base  --seed 1573 --batch-mult 0.4
python train_torch.py --mode cls --model deit-base  --seed 2468 --batch-mult 0.4
python train_torch.py --mode cls --model deit-base  --seed 3456 --batch-mult 0.4

# MobileNet classification models - mobilenet-v2
python train_torch.py --mode cls --model mobilenet-v2        --seed 42 --batch-mult 1.0
python train_torch.py --mode cls --model mobilenet-v2        --seed 737 --batch-mult 1.0
python train_torch.py --mode cls --model mobilenet-v2        --seed 1573 --batch-mult 1.0
python train_torch.py --mode cls --model mobilenet-v2        --seed 2468 --batch-mult 1.0
python train_torch.py --mode cls --model mobilenet-v2        --seed 3456 --batch-mult 1.0

# MobileNet classification models - mobilenet-v3-small
python train_torch.py --mode cls --model mobilenet-v3-small  --seed 42 --batch-mult 1.0
python train_torch.py --mode cls --model mobilenet-v3-small  --seed 737 --batch-mult 1.0
python train_torch.py --mode cls --model mobilenet-v3-small  --seed 1573 --batch-mult 1.0
python train_torch.py --mode cls --model mobilenet-v3-small  --seed 2468 --batch-mult 1.0
python train_torch.py --mode cls --model mobilenet-v3-small  --seed 3456 --batch-mult 1.0

# MobileNet classification models - mobilenet-v3-large
python train_torch.py --mode cls --model mobilenet-v3-large  --seed 42 --batch-mult 0.6
python train_torch.py --mode cls --model mobilenet-v3-large  --seed 737 --batch-mult 0.6
python train_torch.py --mode cls --model mobilenet-v3-large  --seed 1573 --batch-mult 0.6
python train_torch.py --mode cls --model mobilenet-v3-large  --seed 2468 --batch-mult 0.6
python train_torch.py --mode cls --model mobilenet-v3-large  --seed 3456 --batch-mult 0.6

echo "All models trained successfully!"
echo "Check the results/ directory for training results."