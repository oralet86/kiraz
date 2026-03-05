# Kiraz

Kiraz is a cherry fruit detection and classification machine learning research project.

## Goal

Train and evaluate multiple architectures on a cherry dataset across 5 random seeds for reproducibility, covering both object detection and image classification tasks.

## Files

### Core Infrastructure

- `paths.py` - All path constants and path-generation functions
- `hyperparams.py` - All hyperparameters and training configurations
- `log.py` - Project-wide logging system

### Training Scripts

- `train.py` - Main training entry point for both detection and classification
- `hpo_yolo.py` - Hyperparameter optimization using Optuna for both YOLO detection and classification models

### Data Utilities

- `data_utils.py` - Comprehensive dataset utility functions
- `data_augmentation.py` - Data augmentation techniques

### Evaluation and Testing

- `compile_metrics.py` - Parse and compile training results from logs
- `test_models.sh` - Batch testing script

### Miscellaneous

- `aruco_markers.py` - ArUco marker generation and pose estimation demo
- `otsu.py` - Otsu thresholding exploration for cherry segmentation
- `train_all_models.sh` - Run all training jobs across multiple seeds

### Configuration

- `requirements.txt` - Python dependencies
