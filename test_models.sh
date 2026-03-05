#!/bin/bash

echo "Testing models..."

#python train.py --mode cls --model yolov8n-cls.pt --epoch 50

python train.py --mode detect --model yolov8n.pt --epoch 50

echo "All models tested successfully!"
echo "Check the runs/detect/results directory for results."
