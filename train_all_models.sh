#!/bin/bash

echo "Comprehensive Model Training with Seeds"
echo

# Define 5 seeds for reproducible training
SEEDS=(42 737 1573 2468 3456)

# Detection models (without HPO)
DETECT_MODELS=(yolov8n yolov8s yolov8m yolo11n yolo11s yolo11m yolov9t yolov9s yolov9m yolo12n yolov12s yolov12m yolov10n yolov10s yolov10m yolo26n yolo26s yolo26m)

# Detection models with HPO (only those supported in hyperparams.py)
DETECT_HPO_MODELS=(yolov10m yolo11m)

# Classification models with HPO
CLS_HPO_MODELS=(yolo26l-cls yolo11l-cls)

# Classification models (without HPO)
CLS_MODELS=(yolov8n-cls yolov8s-cls yolov8m-cls yolov8l-cls yolo11n-cls yolo11s-cls yolo11m-cls yolo11l-cls yolo26n-cls yolo26s-cls yolo26m-cls yolo26l-cls)

echo "Training with seeds: ${SEEDS[*]}"
echo

# Calculate total jobs
detect_jobs=$((${#DETECT_MODELS[@]} + ${#DETECT_HPO_MODELS[@]}))
cls_jobs=$((${#CLS_HPO_MODELS[@]} + ${#CLS_MODELS[@]}))
total_jobs=$(((${detect_jobs} + ${cls_jobs}) * ${#SEEDS[@]}))

echo "Total training jobs: $total_jobs"
echo

# Create log directory if not exists
mkdir -p training_logs

current_job=0

echo "========================================"
echo "Training Detection Models (Standard)"
echo "========================================"
for seed in "${SEEDS[@]}"; do
    echo
    echo "=== Training Detection Models with Seed $seed ==="
    
    for model in "${DETECT_MODELS[@]}"; do
        ((current_job++))
        echo "[$current_job/$total_jobs] Training $model with seed $seed..."
        echo "[$current_job/$total_jobs] Training $model with seed $seed..." >> "training_logs/training_log_$seed.txt"
        
        if python train.py --mode detect --model "$model" --seed "$seed" >> "training_logs/training_log_$seed.txt" 2>&1; then
            echo "  [SUCCESS] $model with seed $seed completed"
            echo "  [SUCCESS] $model with seed $seed completed" >> "training_logs/training_log_$seed.txt"
        else
            echo "  [ERROR] $model with seed $seed failed (check training_log_$seed.txt)"
            echo "  [ERROR] $model with seed $seed failed" >> "training_logs/training_log_$seed.txt"
        fi
    done
done

echo
echo "========================================"
echo "Training Detection Models (HPO)"
echo "========================================"
for seed in "${SEEDS[@]}"; do
    echo
    echo "=== Training Detection Models with HPO and Seed $seed ==="
    
    for model in "${DETECT_HPO_MODELS[@]}"; do
        ((current_job++))
        echo "[$current_job/$total_jobs] Training $model with HPO and seed $seed..."
        echo "[$current_job/$total_jobs] Training $model with HPO and seed $seed..." >> "training_logs/training_log_$seed.txt"
        
        if python train.py --mode detect --model "$model" --hpo --seed "$seed" >> "training_logs/training_log_$seed.txt" 2>&1; then
            echo "  [SUCCESS] $model with HPO and seed $seed completed"
            echo "  [SUCCESS] $model with HPO and seed $seed completed" >> "training_logs/training_log_$seed.txt"
        else
            echo "  [ERROR] $model with HPO and seed $seed failed (check training_log_$seed.txt)"
            echo "  [ERROR] $model with HPO and seed $seed failed" >> "training_logs/training_log_$seed.txt"
        fi
    done
done

echo
echo "========================================"
echo "Training Classification Models (HPO)"
echo "========================================"
for seed in "${SEEDS[@]}"; do
    echo
    echo "=== Training Classification Models with HPO and Seed $seed ==="
    
    for model in "${CLS_HPO_MODELS[@]}"; do
        ((current_job++))
        echo "[$current_job/$total_jobs] Training $model with HPO and seed $seed..."
        echo "[$current_job/$total_jobs] Training $model with HPO and seed $seed..." >> "training_logs/training_log_$seed.txt"
        
        if python train.py --mode cls --model "$model" --hpo --seed "$seed" >> "training_logs/training_log_$seed.txt" 2>&1; then
            echo "  [SUCCESS] $model with HPO and seed $seed completed"
            echo "  [SUCCESS] $model with HPO and seed $seed completed" >> "training_logs/training_log_$seed.txt"
        else
            echo "  [ERROR] $model with HPO and seed $seed failed (check training_log_$seed.txt)"
            echo "  [ERROR] $model with HPO and seed $seed failed" >> "training_logs/training_log_$seed.txt"
        fi
    done
done

echo
echo "========================================"
echo "Training Classification Models (Standard)"
echo "========================================"
for seed in "${SEEDS[@]}"; do
    echo
    echo "=== Training Classification Models with Seed $seed ==="
    
    for model in "${CLS_MODELS[@]}"; do
        ((current_job++))
        echo "[$current_job/$total_jobs] Training $model with seed $seed..."
        echo "[$current_job/$total_jobs] Training $model with seed $seed..." >> "training_logs/training_log_$seed.txt"
        
        if python train.py --mode cls --model "$model" --seed "$seed" >> "training_logs/training_log_$seed.txt" 2>&1; then
            echo "  [SUCCESS] $model with seed $seed completed"
            echo "  [SUCCESS] $model with seed $seed completed" >> "training_logs/training_log_$seed.txt"
        else
            echo "  [ERROR] $model with seed $seed failed (check training_log_$seed.txt)"
            echo "  [ERROR] $model with seed $seed failed" >> "training_logs/training_log_$seed.txt"
        fi
    done
done

echo
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo "Total jobs processed: $current_job/$total_jobs"
echo
echo "Results are saved in the results/ directory"
echo "Detailed logs are available in training_logs/"
echo
echo "Check results.csv for comprehensive results with seed information"
echo

# Generate summary report
echo "Generating summary report..."
{
    echo "Training Summary Report"
    echo "Generated on: $(date)"
    echo "Total jobs: $current_job/$total_jobs"
    echo
    echo "Seeds used: ${SEEDS[*]}"
    echo
    echo "Model categories:"
    echo "- Detection (Standard): ${DETECT_MODELS[*]}"
    echo "- Detection (HPO): ${DETECT_HPO_MODELS[*]}"
    echo "- Classification (HPO): ${CLS_HPO_MODELS[*]}"
    echo "- Classification (Standard): ${CLS_MODELS[*]}"
    echo
    echo "For detailed results, see: results/results.csv"
    echo "For training logs, see: training_logs/training_log_*.txt"
} > training_logs/summary_report.txt

echo "Summary report saved to: training_logs/summary_report.txt"
echo
