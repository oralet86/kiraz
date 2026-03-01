@echo off
echo Testing models...
python train.py --mode cls --model yolo8n-cls
python train.py --mode detect --model yolo8n
python train.py --mode cls --model yolo11l-cls --hpo
python train.py --mode detect --model yolo11m --hpo
echo Done.
