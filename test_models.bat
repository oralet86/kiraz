@echo off
echo Testing models...
python train.py --mode cls --model yolov8n-cls.pt
python train.py --mode detect --model yolov8n.pt
python train.py --mode cls --model yolo11l-cls.pt --hpo
python train.py --mode detect --model yolo11m.pt --hpo
echo Done.
