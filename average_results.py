#!/usr/bin/env python3
"""
Script to group results by model name and calculate statistics for test metrics only.
Includes model_name, sample_count, and test_* columns with their standard deviations and standard errors.
Excludes train_*, val_*, mode, seed, and timestamp columns.
"""

import pandas as pd
import sys
from pathlib import Path
from typing import Set


def get_exclude_columns(df: pd.DataFrame) -> Set[str]:
    """Get columns to exclude from processing."""
    # We want to keep: model_name, sample_count, and test_* columns
    # We want to exclude: all other columns (train_*, val_*, mode, seed, timestamp)
    exclude_cols = {"mode", "seed", "timestamp"}
    # Add all non-test metric columns (train_*, val_*, and their std/std_err variants)
    for col in df.columns:
        if col.startswith("train_") or col.startswith("val_"):
            exclude_cols.add(col)
        elif col.startswith("train_") or col.startswith("val_"):
            exclude_cols.add(col)
    return {col for col in exclude_cols if col in df.columns}


def get_columns_to_keep(df: pd.DataFrame) -> list:
    """Get columns to keep in the final output."""
    columns_to_keep = ["model_name"]
    if "sample_count" in df.columns:
        columns_to_keep.append("sample_count")

    # Add test_* columns and their std_err variants
    for col in df.columns:
        if col.startswith("test_") or (
            col.startswith("test_") and col.endswith("_std_err")
        ):
            columns_to_keep.append(col)

    return [col for col in columns_to_keep if col in df.columns]


def average_metrics_by_model(csv_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Group results by model name and calculate statistics for test metrics only.

    Args:
        csv_path: Path to the input CSV file
        output_path: Optional path to save the averaged results

    Returns:
        DataFrame with test metrics, standard deviations, and standard errors grouped by model name
    """
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from {csv_path}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    # Check if required columns exist
    if "model_name" not in df.columns:
        print("Error: 'model_name' column not found in CSV")
        sys.exit(1)

    # Get test metric columns only
    test_cols = [col for col in df.columns if col.startswith("test_")]
    if not test_cols:
        print("Warning: No test_* columns found in the data")
        test_cols = []

    print(f"Processing {len(test_cols)} test metric columns")

    # Group by model_name and calculate mean and std for test columns only
    if test_cols:
        grouped_stats = (
            df.groupby("model_name")[test_cols].agg(["mean", "std"]).reset_index()
        )

        # Flatten column names (multi-level index from agg)
        grouped_stats.columns = [
            f"{col}_{stat}" if stat != "mean" else col
            for col, stat in grouped_stats.columns.values
        ]

        # Rename the model_name column back
        grouped_stats = grouped_stats.rename(columns={"model_name_": "model_name"})
    else:
        # No test columns, just group by model_name
        grouped_stats = df.groupby("model_name").size().reset_index(name="sample_count")

    # Add count of samples per model if not already present
    if "sample_count" not in grouped_stats.columns:
        counts = df.groupby("model_name").size().reset_index(name="sample_count")
        if test_cols:
            grouped_stats = grouped_stats.merge(counts, on="model_name")
        else:
            grouped_stats = counts

    # Calculate standard error for each test metric (std / sqrt(n))
    for col in test_cols:
        std_col = f"{col}_std"
        se_col = f"{col}_std_err"
        if std_col in grouped_stats.columns:
            grouped_stats[se_col] = grouped_stats[std_col] / grouped_stats[
                "sample_count"
            ].pow(0.5)

    # Keep only model_name, sample_count, and test columns with their std and std_err
    final_columns = ["model_name", "sample_count"]

    # Add test means, standard deviations, and standard errors
    for col in test_cols:
        if col in grouped_stats.columns:
            final_columns.append(col)
        std_col = f"{col}_std"
        se_col = f"{col}_std_err"
        if std_col in grouped_stats.columns:
            final_columns.append(std_col)
        if se_col in grouped_stats.columns:
            final_columns.append(se_col)

    # Filter to only existing columns
    final_columns = [col for col in final_columns if col in grouped_stats.columns]
    grouped_stats = grouped_stats[final_columns]

    # Sort by model_name for consistency
    grouped_stats = grouped_stats.sort_values("model_name").reset_index(drop=True)

    # Save to output path if provided
    if output_path:
        try:
            grouped_stats.to_csv(output_path, index=False)
            print(
                f"Saved test results with standard deviations and standard errors to {output_path}"
            )
        except Exception as e:
            print(f"Error saving to {output_path}: {e}")

    return grouped_stats


def main():
    """Main function to run the averaging script."""
    # Default paths
    default_input = "results_hpo_detect/results.csv"
    default_output = "results_hpo_detect/results_averaged.csv"

    # Parse command line arguments
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = default_input

    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        output_path = default_output

    # Check if input file exists
    if not Path(input_path).exists():
        print(f"Error: Input file {input_path} does not exist")
        print(f"Usage: {sys.argv[0]} [input_csv] [output_csv]")
        sys.exit(1)

    print(f"Processing test metrics from: {input_path}")
    print(f"Output will be saved to: {output_path}")
    print("-" * 50)

    # Average the metrics
    result_df = average_metrics_by_model(input_path, output_path)

    print("-" * 50)
    print(
        f"Test results with standard deviations and standard errors for {len(result_df)} models:"
    )
    print(result_df[["model_name", "sample_count"]].to_string(index=False))
    print(
        f"\nFull test results with standard deviations and standard errors saved to {output_path}"
    )


if __name__ == "__main__":
    main()
