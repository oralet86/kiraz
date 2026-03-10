#!/bin/bash

echo "Training ViT and ConvNeXt models..."
source ./.venv/bin/activate

python train_torch.py --mode cls --model convnextv2-femto --seed 1573 --batch-mult 1.0
python train_torch.py --mode cls --model convnextv2-femto --seed 2468 --batch-mult 1.0
python train_torch.py --mode cls --model convnextv2-femto --seed 3456 --batch-mult 1.0

# ConvNeXt V2 classification models - convnextv2-pico
python train_torch.py --mode cls --model convnextv2-pico --seed 42 --batch-mult 1.0
python train_torch.py --mode cls --model convnextv2-pico --seed 737 --batch-mult 1.0
python train_torch.py --mode cls --model convnextv2-pico --seed 1573 --batch-mult 1.0
python train_torch.py --mode cls --model convnextv2-pico --seed 2468 --batch-mult 1.0
python train_torch.py --mode cls --model convnextv2-pico --seed 3456 --batch-mult 1.0

# ConvNeXt V2 classification models - convnextv2-nano
python train_torch.py --mode cls --model convnextv2-nano --seed 42 --batch-mult 0.8
python train_torch.py --mode cls --model convnextv2-nano --seed 737 --batch-mult 0.8
python train_torch.py --mode cls --model convnextv2-nano --seed 1573 --batch-mult 0.8
python train_torch.py --mode cls --model convnextv2-nano --seed 2468 --batch-mult 0.8
python train_torch.py --mode cls --model convnextv2-nano --seed 3456 --batch-mult 0.8

# ConvNeXt V2 classification models - convnextv2-tiny
python train_torch.py --mode cls --model convnextv2-tiny --seed 42 --batch-mult 0.6
python train_torch.py --mode cls --model convnextv2-tiny --seed 737 --batch-mult 0.6
python train_torch.py --mode cls --model convnextv2-tiny --seed 1573 --batch-mult 0.6
python train_torch.py --mode cls --model convnextv2-tiny --seed 2468 --batch-mult 0.6
python train_torch.py --mode cls --model convnextv2-tiny --seed 3456 --batch-mult 0.6

# ConvNeXt V2 classification models - convnextv2-base
python train_torch.py --mode cls --model convnextv2-base --seed 42 --batch-mult 0.3
python train_torch.py --mode cls --model convnextv2-base --seed 737 --batch-mult 0.3
python train_torch.py --mode cls --model convnextv2-base --seed 1573 --batch-mult 0.3
python train_torch.py --mode cls --model convnextv2-base --seed 2468 --batch-mult 0.3
python train_torch.py --mode cls --model convnextv2-base --seed 3456 --batch-mult 0.3

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

# MobileViT V1 classification models - mobilevit-xxs
python train_torch.py --mode cls --model mobilevit-xxs --seed 42 --batch-mult 1.0
python train_torch.py --mode cls --model mobilevit-xxs --seed 737 --batch-mult 1.0
python train_torch.py --mode cls --model mobilevit-xxs --seed 1573 --batch-mult 1.0
python train_torch.py --mode cls --model mobilevit-xxs --seed 2468 --batch-mult 1.0
python train_torch.py --mode cls --model mobilevit-xxs --seed 3456 --batch-mult 1.0

# MobileViT V1 classification models - mobilevit-xs
python train_torch.py --mode cls --model mobilevit-xs --seed 42 --batch-mult 0.8
python train_torch.py --mode cls --model mobilevit-xs --seed 737 --batch-mult 0.8
python train_torch.py --mode cls --model mobilevit-xs --seed 1573 --batch-mult 0.8
python train_torch.py --mode cls --model mobilevit-xs --seed 2468 --batch-mult 0.8
python train_torch.py --mode cls --model mobilevit-xs --seed 3456 --batch-mult 0.8

# MobileViT V1 classification models - mobilevit-s
python train_torch.py --mode cls --model mobilevit-s --seed 42 --batch-mult 0.6
python train_torch.py --mode cls --model mobilevit-s --seed 737 --batch-mult 0.6
python train_torch.py --mode cls --model mobilevit-s --seed 1573 --batch-mult 0.6
python train_torch.py --mode cls --model mobilevit-s --seed 2468 --batch-mult 0.6
python train_torch.py --mode cls --model mobilevit-s --seed 3456 --batch-mult 0.6

# MobileViT V2 classification models - mobilevitv2-050
python train_torch.py --mode cls --model mobilevitv2-050 --seed 42 --batch-mult 1.0
python train_torch.py --mode cls --model mobilevitv2-050 --seed 737 --batch-mult 1.0
python train_torch.py --mode cls --model mobilevitv2-050 --seed 1573 --batch-mult 1.0
python train_torch.py --mode cls --model mobilevitv2-050 --seed 2468 --batch-mult 1.0
python train_torch.py --mode cls --model mobilevitv2-050 --seed 3456 --batch-mult 1.0

# MobileViT V2 classification models - mobilevitv2-075
python train_torch.py --mode cls --model mobilevitv2-075 --seed 42 --batch-mult 0.8
python train_torch.py --mode cls --model mobilevitv2-075 --seed 737 --batch-mult 0.8
python train_torch.py --mode cls --model mobilevitv2-075 --seed 1573 --batch-mult 0.8
python train_torch.py --mode cls --model mobilevitv2-075 --seed 2468 --batch-mult 0.8
python train_torch.py --mode cls --model mobilevitv2-075 --seed 3456 --batch-mult 0.8

# MobileViT V2 classification models - mobilevitv2-100
python train_torch.py --mode cls --model mobilevitv2-100 --seed 42 --batch-mult 0.6
python train_torch.py --mode cls --model mobilevitv2-100 --seed 737 --batch-mult 0.6
python train_torch.py --mode cls --model mobilevitv2-100 --seed 1573 --batch-mult 0.6
python train_torch.py --mode cls --model mobilevitv2-100 --seed 2468 --batch-mult 0.6
python train_torch.py --mode cls --model mobilevitv2-100 --seed 3456 --batch-mult 0.6

# Swin Transformer classification models - swin-tiny
python train_torch.py --mode cls --model swin-tiny --seed 42 --batch-mult 0.5
python train_torch.py --mode cls --model swin-tiny --seed 737 --batch-mult 0.5
python train_torch.py --mode cls --model swin-tiny --seed 1573 --batch-mult 0.5
python train_torch.py --mode cls --model swin-tiny --seed 2468 --batch-mult 0.5
python train_torch.py --mode cls --model swin-tiny --seed 3456 --batch-mult 0.5

echo "All models trained successfully!"
echo "Check the results/ directory for training results."