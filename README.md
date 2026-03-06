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
- `download_datasets.py` - Dataset download and extraction from Google Drive

### Evaluation and Metrics

- `metrics.py` - Additional metrics computation for classification and detection
- `compile_metrics.py` - Parse and compile training results from logs

### Testing

- `test_models.sh` - Batch testing script
- `train_all_models.sh` - Run all training jobs across multiple seeds

### Miscellaneous

- `aruco_markers.py` - ArUco marker generation and pose estimation demo
- `otsu.py` - Otsu thresholding exploration for cherry segmentation
- `cloud_setup.sh` - Cloud environment setup script

### Configuration

- `requirements.txt` - Python dependencies

### paths.py

**Path Generation Functions:**
- `get_dataset_paths(dataset_root, splits)` - Generate standardized dataset structure paths for images/labels directories
- `get_results_paths(mode, timestamp)` - Generate paths for training results based on mode and timestamp
- `ensure_directories()` - Create all core project directories if they don't exist

### hyperparams.py

**VRAM and Batch Management:**
- `get_vram_gb()` - Get available GPU VRAM in GB
- `get_scaled_batch_size(base_batch, vram_gb)` - Scale batch size based on available VRAM

**Hyperparameter Management:**
- `get_default_cls_hyperparams()` - Get default hyperparameters for classification tasks
- `get_default_detect_hyperparams()` - Get default hyperparameters for detection tasks
- `get_training_config(task)` - Get training configuration constants for a specific task
- `get_hyperparams(model_name, hpo)` - Get hyperparameters for a specific model with optional HPO optimization
- `list_available_models()` - List all models with HPO support status
- `validate_hyperparams(params)` - Validate hyperparameter dictionary for required fields

### log.py

**Logging Setup:**
- `setup_logging()` - Setup simple logging to console and universal log file
- `get_logger()` - Get the singleton logger instance
- `add_log_file(file_path)` - Add additional log file to logger

## Training Scripts

### train.py

**Setup and Validation:**
- `setup_logging(log_file)` - Setup additional logging for training session
- `set_seed(seed)` - Set random seeds for all libraries to ensure reproducible results
- `ensure_dataset_layout(dataset_dir, mode)` - Validate dataset structure based on training mode
- `ensure_data_yaml(dataset_dir, yaml_path)` - Create data.yaml for detection mode if missing

**Training and Evaluation:**
- `_extract_cls_metrics(results_obj)` - Extract classification metrics from validation results
- `log_results_to_file(model_name, metrics, train_time, mode)` - Append results to unified results.csv
- `train_model(ts, mode)` - Train model based on mode (cls/detect) with hyperparameters
- `evaluate_on_splits(model, ts, mode)` - Evaluate trained model on validation and test splits
- `cleanup_cuda()` - Clean up CUDA memory with aggressive garbage collection
- `main()` - Main training pipeline orchestrator

### hpo_yolo.py

**Dataset Validation:**
- `ensure_dataset_layout()` - Validate dataset structure for HPO
- `ensure_data_yaml()` - Create data.yaml for detection mode

**Hyperparameter Optimization:**
- `_suggest_cls_params(trial)` - Suggest classification hyperparameters for Optuna trial
- `_suggest_detect_params(trial)` - Suggest detection hyperparameters for Optuna trial
- `_build_trial_params(suggested)` - Build complete trial parameters from suggestions
- `objective(trial)` - Optuna objective function for single HPO trial
- `_extract_metrics(results_obj)` - Extract metrics from training results
- `train_final_model(best_params, ts)` - Train final model with best hyperparameters
- `evaluate_on_splits(best_weights, ts)` - Evaluate final model on all splits
- `save_trial_results(study, ts)` - Save HPO trial results to CSV files
- `main()` - Main HPO pipeline orchestrator

## Data Utilities

### data_utils.py

**Resolution Analysis:**
- `collect_image_paths(dataset_dir, splits)` - Collect all image paths from dataset splits
- `analyze_image_resolutions(image_paths)` - Analyze resolutions of collected images
- `print_resolution_statistics(name, values)` - Print statistical summary of resolution values
- `print_resolution_buckets(short_sides)` - Print distribution of images by short-side resolution
- `analyze_dataset_resolution(dataset_dir, splits)` - Complete resolution analysis for a dataset

**Dataset Statistics:**
- `load_class_names(dataset_path)` - Load class names from YOLO data.yaml file
- `read_label_counts(label_path)` - Count objects per class in a YOLO label file
- `get_image_files(images_dir)` - Get all image files from directory
- `dataset_statistics(dataset_dir, splits)` - Print comprehensive dataset statistics including class distributions

**Dataset Merging and Splitting:**
- `merge_dataset_splits(source_path, output_path, splits)` - Merge train/val/test splits into single flat dataset
- `stratified_dataset_split(data_dir, output_dir, split_ratios, random_seed)` - Perform object-aware stratified dataset splitting
- `stratified_cls_split(input_dir, output_dir, split_ratios, random_seed)` - Perform stratified split of classification dataset

**YOLO Label Manipulation:**
- `load_yolo_polygons(label_path, img_w, img_h)` - Load YOLO format polygons from label file
- `save_yolo_polygons(label_path, polygons, crop_x1, crop_y1, crop_w, crop_h)` - Save YOLO polygons with crop adjustment
- `remap_yolo_labels(dataset_path, label_mapping, splits)` - Remap YOLO class labels according to mapping
- `remap_flat_labels(dataset_path, label_mapping)` - Remap labels in flat dataset structure
- `filter_flat_labels_by_class(source_path, target_path, allowed_classes, class_mapping)` - Filter flat dataset to specific classes

**Image Cropping:**
- `calculate_crop_bounds(points, buffer_ratio, img_w, img_h)` - Calculate crop bounds around points with buffer
- `crop_single_objects(input_dir, output_dir, buffer_ratio, splits)` - Crop images to individual objects (one image per object)
- `crop_multi_objects(input_dir, output_dir, buffer_ratio, splits)` - Crop images to contain all objects (one crop per image)
- `crop_flat_dataset(input_dir, output_dir, buffer_ratio)` - Crop flat dataset around all objects per image
- `crop_flat_dataset_to_cls_format(input_dir, output_dir, class_mapping, buffer_ratio)` - Crop objects directly to classification folder structure

**Dataset Conversion:**
- `convert_detection_to_classification(src_root, dest_root, class_mapping, splits)` - Convert detection dataset to classification format
- `filter_labels_by_class(source_path, target_path, allowed_classes, class_mapping)` - Filter YOLO dataset to specific classes

**Pipeline and Configuration:**
- `create_data_yaml(dataset_path)` - Create data.yaml file for YOLO dataset
- `run_full_dataset_pipeline()` - Run complete dataset processing pipeline from original to augmented datasets

### data_augmentation.py

**Transform Configuration:**
- `get_classification_transforms(split)` - Get Albumentations transforms for classification tasks
- `get_detection_transforms(split)` - Get Albumentations transforms for detection with bbox handling

**Label Processing:**
- `validate_and_fix_bbox(bbox)` - Validate and fix bounding box coordinates
- `parse_yolo_label(label_path)` - Parse YOLO label file into bboxes and class labels
- `save_yolo_label(label_path, bboxes, class_labels)` - Save YOLO label file
- `draw_bboxes_on_image(image, bboxes, class_labels)` - Draw bounding boxes on image

**Augmentation Functions:**
- `augment_classification_dataset(source_dir, target_dir, augment_factor)` - Augment classification dataset with transforms
- `augment_detection_dataset(source_dir, target_dir, augment_factor)` - Augment detection dataset with label handling
- `main()` - CLI entry point for augmentation tasks

## Evaluation and Metrics

### metrics.py

**Classification Metrics:**
- `cls_precision_recall_f1(val_result)` - Compute macro-averaged precision, recall and F1 from confusion matrix

**Detection Metrics:**
- `detect_metrics(model, data, split, device, batch)` - Run validation capturing mAP50-95 and mean IoU simultaneously

**Testing Functions:**
- `_test_classification(model_name, device)` - Smoke-test classification metrics
- `_test_detection(model_name, device)` - Smoke-test detection metrics
- `main()` - CLI entry point for metrics testing

### compile_metrics.py

**Log Parsing:**
- `extract_model_and_seed(log_content)` - Extract model name and seed from log content
- `extract_metrics(log_content)` - Extract all metrics from log content using regex
- `is_training_successful(log_content)` - Check if training completed successfully
- `parse_log_file(log_path)` - Parse single log file and extract training information

**Compilation:**
- `compile_all_metrics()` - Compile metrics from all log files in results directory
- `write_compiled_results(runs, output_path)` - Write compiled results to CSV file
- `main()` - Main entry point for metric compilation

## Miscellaneous Utilities

### download_datasets.py

**Dataset Download:**
- `load_env()` - Load environment variables from .env file
- `download_and_extract()` - Download and extract datasets from Google Drive
- `main()` - Main function orchestrating download process

### aruco_markers.py

**Marker Generation:**
- `generate_clean_aruco_pdf(filename, marker_ids, marker_size_cm)` - Generate PDF with ArUco markers
- `generate_chessboard_pdf(filename, square_size_cm, rows, cols)` - Generate chessboard calibration pattern PDF

**Live Demo:**
- `start_live_demo(target_ids, marker_size_cm)` - Start live camera demo with 3D axis visualization

### otsu.py

**Image Processing:**
- `main()` - Apply Otsu thresholding on R-G channel difference for cherry segmentation (exploratory script)

---

## Summary

The Kiraz project contains approximately 80+ functions organized across:
- **Core Infrastructure** (3 files): Path management, hyperparameters, logging
- **Training Scripts** (2 files): Main training and HPO pipelines  
- **Data Utilities** (2 files): Dataset processing, augmentation, and conversion
- **Evaluation** (2 files): Metrics computation and result compilation
- **Miscellaneous** (3 files): Dataset download, ArUco markers, exploratory analysis

Each function is designed to be modular and reusable, with clear separation of concerns between data processing, training, evaluation, and utility operations.
