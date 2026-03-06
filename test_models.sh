#!/bin/bash

echo "Testing models..."

python train.py --mode cls --model yolo11l-cls.pt --epoch 1 --batch-mult 0.2

python train.py --mode detect --model yolo11m.pt --epoch 1 --batch-mult 0.2

echo "All models tested successfully!"
echo "Check the runs/cls/results and runs/detect/results directories for results."

# SEEDS=(42 737 1573 2468 3456)
# DETECT_MODELS=(yolov8n yolov8s yolov8m yolo11n yolo11s yolo11m yolov9t yolov9s yolov9m yolo12n yolov12s yolov12m yolov10n yolov10s yolov10m yolo26n yolo26s yolo26m)
# CLS_MODELS=(yolov8n-cls yolov8s-cls yolov8m-cls yolov8l-cls yolo11n-cls yolo11s-cls yolo11m-cls yolo11l-cls yolo26n-cls yolo26s-cls yolo26m-cls yolo26l-cls)
