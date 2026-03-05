import re
import csv
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, field

from paths import RESULTS_DIR


@dataclass
class TrainingRun:
    model_name: str
    seed: int
    log_file: str
    successful: bool = False
    metrics: Dict[str, float] = field(default_factory=dict)


def extract_model_and_seed(log_content: str) -> tuple[Optional[str], Optional[int]]:
    """Extract model name and seed from log content."""
    model_pattern = (
        r"Starting (?:cls|detect) training with model: (\S+) \(seed: (\d+)\)"
    )
    match = re.search(model_pattern, log_content)
    if match:
        return match.group(1), int(match.group(2))
    return None, None


def extract_metrics(log_content: str) -> Dict[str, float]:
    """Extract all metrics from log content."""
    metrics = {}

    metric_pattern = r"INFO -   (\S+):\s+([\d.]+)"

    for match in re.finditer(metric_pattern, log_content):
        metric_name = match.group(1)
        metric_value = float(match.group(2))
        metrics[metric_name] = metric_value

    return metrics


def is_training_successful(log_content: str) -> bool:
    """Check if training completed successfully."""
    return "Training completed in" in log_content


def parse_log_file(log_path: Path) -> Optional[TrainingRun]:
    """Parse a single log file and extract training information."""
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()

        model_name, seed = extract_model_and_seed(content)

        if model_name is None or seed is None:
            return None

        successful = is_training_successful(content)
        metrics = extract_metrics(content) if successful else {}

        return TrainingRun(
            model_name=model_name,
            seed=seed,
            log_file=log_path.name,
            successful=successful,
            metrics=metrics,
        )

    except Exception as e:
        print(f"Error parsing {log_path}: {e}")
        return None


def compile_all_metrics() -> List[TrainingRun]:
    """Compile metrics from all log files in results directory."""
    runs = []

    if not RESULTS_DIR.exists():
        print(f"Results directory not found: {RESULTS_DIR}")
        return runs

    log_files = list(RESULTS_DIR.glob("*-run.log"))
    print(f"Found {len(log_files)} log files")

    for log_file in log_files:
        run = parse_log_file(log_file)
        if run:
            runs.append(run)

    runs.sort(key=lambda x: (x.model_name, x.seed))

    return runs


def write_compiled_results(runs: List[TrainingRun], output_path: Path) -> None:
    """Write compiled results to CSV file."""
    all_metric_names = set()
    for run in runs:
        if run.successful:
            all_metric_names.update(run.metrics.keys())

    metric_names = sorted(all_metric_names)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        header = ["model_name", "seed", "log_file", "status"] + metric_names
        writer.writerow(header)

        for run in runs:
            row = [
                run.model_name,
                run.seed,
                run.log_file,
                "SUCCESS" if run.successful else "FAILED",
            ]

            if run.successful:
                for metric_name in metric_names:
                    value = run.metrics.get(metric_name, "")
                    row.append(f"{value:.4f}" if value != "" else "")
            else:
                row.extend([""] * len(metric_names))

            writer.writerow(row)

    print(f"Compiled results written to: {output_path}")


def main() -> None:
    """Main entry point for metric compilation."""
    print("Compiling metrics from training logs...")

    runs = compile_all_metrics()

    if not runs:
        print("No valid training runs found.")
        return

    print(f"Parsed {len(runs)} training runs")
    successful_runs = sum(1 for run in runs if run.successful)
    failed_runs = len(runs) - successful_runs
    print(f"  Successful: {successful_runs}")
    print(f"  Failed: {failed_runs}")

    output_path = RESULTS_DIR / "compiled_results.csv"
    write_compiled_results(runs, output_path)


if __name__ == "__main__":
    main()
