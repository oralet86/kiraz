#!/usr/bin/env python3
"""
Model Performance Visualization Script

Generates performance visualization graphs for object detection and classification models
from CSV data, grouping by model generation.

Usage examples:
  python plot_results.py                                  # all graphs
  python plot_results.py --task detect                    # detection only
  python plot_results.py --plot family pareto             # family + pareto plots only
  python plot_results.py --task cls --family convnextv2   # one family
  python plot_results.py --plot efficiency normalized     # efficiency graphs only
"""

from __future__ import annotations

import argparse
import itertools
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskConfig:
    task_type: str  # "detection" or "classification"
    csv_file: str
    y_col: str
    y_err_col: str
    y_label: str
    title_suffix: str
    family_color: str  # single-family line/marker color
    pareto_title: str
    pareto_filename: str


DETECTION = TaskConfig(
    task_type="detection",
    csv_file="object_aggregates.csv",
    y_col="test_box_f1",
    y_err_col="test_box_f1_std_err",
    y_label="F1 Score",
    title_suffix="Object Detection",
    family_color="#1f77b4",
    pareto_title="Object Detection, Pareto Frontier",
    pareto_filename="detection_pareto.pdf",
)

CLASSIFICATION = TaskConfig(
    task_type="classification",
    csv_file="cls_aggregates.csv",
    y_col="test_f1",
    y_err_col="test_f1_std_err",
    y_label="Mean F1 Score",
    title_suffix="Classification",
    family_color="#ff7f0e",
    pareto_title="Classification, Pareto Frontier",
    pareto_filename="classification_pareto.pdf",
)

X_COL = "Median Latency (ms)"
X_LABEL = "Average processing latency (ms/img)"

_SIZE_RE = re.compile(
    r"[-_]?(atto|femto|pico|nano|tiny|small|medium|large|base|\d{3}|[ntsmbl])$",
    re.IGNORECASE,
)
_SIZE_NORM: dict[str, str] = {
    "n": "nano",
    "t": "tiny",
    "s": "small",
    "m": "medium",
    "l": "large",
    "b": "base",
}
# Ordered from smallest to largest for a consistent legend
_SIZE_MARKERS: dict[str, str] = {
    "atto": "v",
    "femto": "<",
    "pico": ">",
    "nano": "o",
    "050": "^",
    "tiny": "^",
    "075": "s",
    "small": "s",
    "100": "D",
    "medium": "D",
    "large": "P",
    "base": "h",  # hexagon — distinct from all other shapes
}
_SIZE_ORDER = list(_SIZE_MARKERS)

# Per-marker legend size so all shapes appear visually equal
_LEGEND_MARKER_SIZE: dict[str, int] = {
    "v": 10,
    "<": 10,
    ">": 10,
    "o": 9,
    "^": 10,
    "s": 9,
    "D": 9,
    "P": 11,
    "h": 13,
}

_FAMILY_DISPLAY: dict[str, str] = {
    # Detection
    "yolov8": "YOLOv8",
    "yolov9": "YOLOv9",
    "yolov10": "YOLOv10",
    "yolo11": "YOLO11",
    "yolo12": "YOLO12",
    "yolo26": "YOLO26",
    # Classification
    "yolov8-cls": "YOLOv8",
    "yolo11-cls": "YOLO11",
    "yolo26-cls": "YOLO26",
    "convnext": "ConvNeXt",
    "convnextv2": "ConvNeXt V2",
    "deit": "DeiT",
    "vit": "Vision Transformer",
    "mobilenet-v2": "MobileNet V2",
    "mobilenet-v3": "MobileNet V3",
    "mobilevit": "MobileViT",
    "mobilevitv2": "MobileViT V2",
    "efficientnet": "EfficientNet",
    "resnet": "ResNet",
    "swin": "Swin Transformer",
}

_SIZE_DISPLAY: dict[str, str] = {
    "atto": "Atto",
    "femto": "Femto",
    "pico": "Pico",
    "nano": "Nano",
    "050": "050",
    "tiny": "Tiny",
    "075": "075",
    "small": "Small",
    "100": "100",
    "medium": "Medium",
    "large": "Large",
    "base": "Base",
}


def extract_model_size(model_name: str) -> str:
    """Return a canonical size string (e.g. 'small', 'medium') for a model name."""
    name = model_name.lower().replace("-cls", "")
    m = _SIZE_RE.search(name)
    if not m:
        return "unknown"
    raw = m.group(1).lower()
    return _SIZE_NORM.get(raw, raw)


_FAMILY_COLORS: dict[str, str] = {
    # Detection
    "yolov8": "#1f77b4",
    "yolov9": "#ff7f0e",
    "yolov10": "#2ca02c",
    "yolo11": "#d62728",
    "yolo12": "#9467bd",
    "yolo26": "#8c564b",
    # Classification
    "yolov8-cls": "#1f77b4",
    "yolo11-cls": "#d62728",
    "yolo26-cls": "#8c564b",
    "convnext": "#e377c2",
    "convnextv2": "#17becf",
    "deit": "#bcbd22",
    "vit": "#e74c3c",
    "mobilenet-v2": "#ff7f0e",
    "mobilenet-v3": "#2ca02c",
    "mobilevit": "#9467bd",
    "mobilevitv2": "#8c564b",
    "efficientnet": "#e377c2",
    "resnet": "#17becf",
    "swin": "#bcbd22",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def extract_model_family(model_name: str) -> str:
    """Extract model family (generation) from model name."""
    size_pattern = (
        r"[-_]?[nsmlt ]+$"
        r"|[-_]?small|[-_]?medium|[-_]?large|[-_]?tiny|[-_]?base"
        r"|[-_]?nano|[-_]?pico|[-_]?femto|[-_]?atto$"
    )
    if "-cls" in model_name:
        base = model_name.replace("-cls", "")
        return re.sub(size_pattern, "", base) + "-cls"
    return re.sub(size_pattern, "", model_name)


def group_models_by_family(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Group models by family; discard families with only one model."""
    buckets: dict[str, list[str]] = {}
    for name in df["model_name"]:
        family = extract_model_family(name)
        buckets.setdefault(family, []).append(name)
    return {
        fam: df[df["model_name"].isin(models)].copy()
        for fam, models in buckets.items()
        if len(models) > 1
    }


def compute_pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
    """Return boolean mask of Pareto-optimal rows (lower latency OR higher performance)."""
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    weakly_better_x = x[None, :] <= x[:, None]
    weakly_better_y = y[None, :] >= y[:, None]
    strictly_better = (x[None, :] < x[:, None]) | (y[None, :] > y[:, None])
    np.fill_diagonal(weakly_better_x, False)
    dominated = (weakly_better_x & weakly_better_y & strictly_better).any(axis=1)
    return pd.Series(~dominated, index=df.index)


def compute_shared_y_range(
    family_dfs: dict[str, pd.DataFrame],
    y_col: str,
    y_err_col: str,
) -> float:
    """Return the widest y-axis span (with 10% buffer each side) across all families."""
    max_span = 0.0
    for fdf in family_dfs.values():
        raw_span = (fdf[y_col] + fdf[y_err_col]).max() - (
            fdf[y_col] - fdf[y_err_col]
        ).min()
        max_span = max(max_span, raw_span * 1.20)
    return max_span


def compute_efficiency_scores(
    pareto_df: pd.DataFrame, x_col: str, y_col: str, latency_scale: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_norm, y_norm, scores) for Pareto models.

    Coordinates are normalised in latency / linear-performance space.
    latency_scale controls whether latency is normalised on a log or linear scale.
    Ideal point is (0, 1); score = 1 - dist_to_ideal / sqrt(2).
    """
    raw_x = pareto_df[x_col].to_numpy()
    x = np.log10(raw_x) if latency_scale == "log" else raw_x
    y_p = pareto_df[y_col].to_numpy()
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_norm = (y_p - y_p.min()) / (y_p.max() - y_p.min() + 1e-12)
    dist = np.sqrt(x_norm**2 + (1 - y_norm) ** 2)
    return x_norm, y_norm, 1 - dist / np.sqrt(2)


def build_family_color_map(pareto_df: pd.DataFrame) -> dict[str, str]:
    """Assign a color to each family present in pareto_df."""
    pareto_df = pareto_df.copy()
    pareto_df["family"] = (
        pareto_df["model_name"]
        .apply(extract_model_family)
        .str.replace(r"-\d{3}$", "", regex=True)
    )
    families = pareto_df["family"].unique().tolist()
    color_iter = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    return {fam: _FAMILY_COLORS.get(fam, next(color_iter)) for fam in families}


# ---------------------------------------------------------------------------
# Plot style helpers
# ---------------------------------------------------------------------------

_CHAR_W = 0.55  # character width as fraction of font size in points
_LINE_H = 1.35  # line height as fraction of font size in points
_PAD = 18.0  # extra pixel padding around each estimated bbox


def _estimate_bbox(
    ax: plt.Axes,
    x_data: float,
    y_data: float,
    text: str,
    fontsize: float,
    offset: tuple[float, float],
    ha: str,
    va: str,
) -> tuple[float, float, float, float]:
    """Return estimated (x0, y0, x1, y1) bounding box in display pixels."""
    px_per_pt = ax.get_figure().get_dpi() / 72.0
    lines = text.split("\n")
    w = max(len(ln) for ln in lines) * fontsize * px_per_pt * _CHAR_W
    h = len(lines) * fontsize * px_per_pt * _LINE_H
    xd, yd = ax.transData.transform((x_data, y_data))
    ax_x, ax_y = xd + offset[0], yd + offset[1]
    x0 = ax_x - w if ha == "right" else (ax_x - w / 2 if ha == "center" else ax_x)
    y0 = ax_y - h if va == "top" else (ax_y - h / 2 if va == "center" else ax_y)
    return x0 - _PAD, y0 - _PAD, x0 + w + _PAD, y0 + h + _PAD


def _boxes_overlap(
    a: tuple[float, float, float, float],
    boxes: list[tuple[float, float, float, float]],
) -> bool:
    ax0, ay0, ax1, ay1 = a
    return any(
        ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0
        for bx0, by0, bx1, by1 in boxes
    )


def annotate_with_fallback(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    fontsize: float,
    color: str,
    placed: list[tuple[float, float, float, float]],
) -> None:
    """Annotate a point; prefer top-left, fall back to bottom-right if it overlaps."""
    candidates = [
        ((-4, 8), "right", "bottom"),
        ((4, -8), "left", "top"),
    ]
    chosen_offset, chosen_ha, chosen_va = candidates[0]  # default
    for offset, ha, va in candidates:
        box = _estimate_bbox(ax, x, y, text, fontsize, offset, ha, va)
        if not _boxes_overlap(box, placed):
            chosen_offset, chosen_ha, chosen_va = offset, ha, va
            placed.append(box)
            break
    else:
        # Both overlap — still place at top-left, just register it
        placed.append(
            _estimate_bbox(
                ax, x, y, text, fontsize, candidates[0][0], *candidates[0][1:]
            )
        )

    ax.annotate(
        text,
        (x, y),
        xytext=chosen_offset,
        textcoords="offset points",
        fontsize=fontsize,
        fontweight="bold",
        color=color,
        va=chosen_va,
        ha=chosen_ha,
    )


def setup_plot_style() -> None:
    """Set up clean, publication-quality plotting style."""
    sns.set_style(
        "whitegrid",
        {
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.5,
            "grid.color": "#cccccc",
        },
    )
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 15,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.titlesize": 17,
            "axes.linewidth": 1.2,
            "axes.edgecolor": "#aaaaaa",
            "axes.labelcolor": "#222222",
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "text.color": "#222222",
            "figure.facecolor": "white",
            "axes.facecolor": "#f8f8f8",
        }
    )


def apply_spine_style(ax: plt.Axes) -> None:
    """Apply consistent spine styling to an axes."""
    for spine in ax.spines.values():
        spine.set_edgecolor("#aaaaaa")
        spine.set_linewidth(1.2)


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save figure as PDF and close it."""
    fig.savefig(
        path,
        format="pdf",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


def plot_family_performance(
    family_df: pd.DataFrame,
    family_name: str,
    cfg: TaskConfig,
    output_dir: Path,
    shared_y_range: float,
) -> None:
    """Plot performance vs latency for a single model family."""
    family_df = family_df.sort_values(X_COL)
    x_vals = family_df[X_COL]
    y_vals = family_df[cfg.y_col]
    y_err = family_df[cfg.y_err_col]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    ax.plot(x_vals, y_vals, color=cfg.family_color, linewidth=3.0, zorder=3)
    ax.errorbar(
        x_vals,
        y_vals,
        yerr=y_err,
        fmt="none",
        ecolor=cfg.family_color,
        capsize=4,
        capthick=1.8,
        alpha=0.7,
        linewidth=1.5,
        zorder=5,
    )

    # Per-point markers by size; build legend handles while iterating
    legend_handles: list[plt.Artist] = []
    seen_markers: set[str] = set()
    for x, y, name in zip(x_vals, y_vals, family_df["model_name"]):
        size = extract_model_size(name)
        marker = _SIZE_MARKERS.get(size, "o")
        ax.scatter(
            x,
            y,
            color=cfg.family_color,
            marker=marker,
            s=90,
            zorder=4,
            edgecolors="white",
            linewidths=1.8,
        )
        if marker not in seen_markers:
            seen_markers.add(marker)
            ms = _LEGEND_MARKER_SIZE.get(marker, 9)
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color="w",
                    markerfacecolor=cfg.family_color,
                    markersize=ms,
                    label=_SIZE_DISPLAY.get(size, size),
                )
            )
        ax.annotate(
            name,
            (x, y),
            xytext=(5, -14),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            color="#222222",
            va="top",
            ha="left",
        )
        ax.annotate(
            f"{y:.6f}\n{x:.0f} ms",
            (x, y),
            xytext=(-5, 10),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            color="#222222",
            va="bottom",
            ha="right",
        )

    if len(legend_handles) > 1:
        ax.legend(
            handles=legend_handles,
            fontsize=10,
            framealpha=0.8,
            loc="lower right",
            title="Size",
            title_fontsize=10,
        )

    ax.set_xlabel(X_LABEL, fontsize=13, fontweight="600", color="#222222")
    ax.set_ylabel(cfg.y_label, fontsize=13, fontweight="600", color="#222222")

    x_min, x_max = x_vals.min(), x_vals.max()
    x_buf = (x_max - x_min) * 0.10
    ax.set_xlim(x_min - x_buf, x_max + x_buf)

    y_mid = ((y_vals - y_err).min() + (y_vals + y_err).max()) / 2
    ax.set_ylim(y_mid - shared_y_range / 2, y_mid + shared_y_range / 2)

    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.grid(True, alpha=0.5, linewidth=0.6, color="#cccccc")
    apply_spine_style(ax)
    fig.tight_layout()
    save_figure(fig, output_dir / f"{family_name}_{cfg.task_type}.pdf")


def plot_pareto_frontier(
    df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    non_pareto_df: pd.DataFrame,
    family_color_map: dict[str, str],
    cfg: TaskConfig,
    output_dir: Path,
) -> None:
    """Scatter plot of the Pareto efficiency frontier."""
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    y_min = (df[cfg.y_col] - df[cfg.y_err_col]).min()
    y_max = (df[cfg.y_col] + df[cfg.y_err_col]).max()
    y_buf = (y_max - y_min) * 0.10
    ax.set_ylim(y_min - y_buf, y_max + y_buf)

    x_min, x_max = df[X_COL].min(), df[X_COL].max()
    x_buf = (x_max - x_min) * 0.10
    ax.set_xlim(x_min - x_buf, x_max + x_buf)

    # Dominated dots — per-point shape by size, faded gray
    for _, row in non_pareto_df.iterrows():
        marker = _SIZE_MARKERS.get(extract_model_size(row["model_name"]), "o")
        ax.scatter(
            row[X_COL],
            row[cfg.y_col],
            color="#aaaaaa",
            marker=marker,
            alpha=0.3,
            s=40,
            zorder=2,
            edgecolors="none",
        )

    ax.plot(
        pareto_df[X_COL],
        pareto_df[cfg.y_col],
        color="#333333",
        linewidth=2.0,
        zorder=3,
        alpha=0.7,
    )

    # Pareto dots — color = family, marker = size
    sizes_present: list[str] = []
    for _, row in pareto_df.iterrows():
        color = family_color_map.get(row["family"], "#555555")
        size = extract_model_size(row["model_name"])
        marker = _SIZE_MARKERS.get(size, "o")
        ax.scatter(
            row[X_COL],
            row[cfg.y_col],
            color=color,
            marker=marker,
            s=110,
            zorder=5,
            edgecolors="white",
            linewidths=1.8,
        )
        if size not in sizes_present:
            sizes_present.append(size)

    # Legend — section 1: family colors (circles, professional names)
    legend_handles: list[plt.Artist] = []
    for family, color in family_color_map.items():
        label = _FAMILY_DISPLAY.get(family, family.replace("-cls", ""))
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=8,
                label=label,
            )
        )
    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#aaaaaa",
            markersize=8,
            alpha=0.5,
            label="Dominated Models",
        )
    )
    legend_handles.append(
        plt.Line2D(
            [0], [1], color="#333333", linewidth=2.0, alpha=0.7, label="Pareto Frontier"
        )
    )
    # Blank separator
    legend_handles.append(plt.Line2D([0], [0], linewidth=0, label=""))
    # Legend — section 2: size shapes (gray, ordered small → large)
    # Deduplicate by marker symbol so e.g. tiny and 050 (both "^") only appear once
    seen_markers: set[str] = set()
    for size in _SIZE_ORDER:
        if size in sizes_present:
            marker = _SIZE_MARKERS[size]
            if marker in seen_markers:
                continue
            seen_markers.add(marker)
            ms = _LEGEND_MARKER_SIZE.get(marker, 9)
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color="w",
                    markerfacecolor="#666666",
                    markersize=ms,
                    label=_SIZE_DISPLAY.get(size, size),
                )
            )

    ax.set_xlabel(X_LABEL, fontsize=13, fontweight="600", color="#222222")
    ax.set_ylabel(cfg.y_label, fontsize=13, fontweight="600", color="#222222")
    ax.legend(handles=legend_handles, fontsize=10, framealpha=0.8, loc="lower right")
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.grid(True, alpha=0.5, linewidth=0.6, color="#cccccc")
    apply_spine_style(ax)

    # Finalise layout and force a full render pass so ax.transData gives accurate display coords
    fig.tight_layout()
    fig.canvas.draw()

    pareto_fs = 11.5
    placed: list[tuple[float, float, float, float]] = []
    for _, row in pareto_df.iterrows():
        label = f"{row[cfg.y_col]:.6f}\n{row[X_COL]:.0f} ms"
        annotate_with_fallback(
            ax, row[X_COL], row[cfg.y_col], label, pareto_fs, "#222222", placed
        )

    save_figure(fig, output_dir / cfg.pareto_filename)


def plot_efficiency_bar(
    pareto_df: pd.DataFrame,
    scores: np.ndarray,
    family_color_map: dict[str, str],
    cfg: TaskConfig,
    output_dir: Path,
) -> None:
    """Bar chart of efficiency scores for Pareto-optimal models."""
    bar_colors = [
        family_color_map.get(pareto_df.iloc[i]["family"], "#aaaaaa")
        for i in range(len(pareto_df))
    ]
    display_labels = [
        f"{_FAMILY_DISPLAY.get(pareto_df.iloc[i]['family'], pareto_df.iloc[i]['family'])}\n"
        f"{_SIZE_DISPLAY.get(extract_model_size(pareto_df.iloc[i]['model_name']), '')}"
        for i in range(len(pareto_df))
    ]
    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    bars = ax.bar(
        display_labels,
        scores,
        color=bar_colors,
        edgecolor="#333333",
        linewidth=0.6,
        zorder=3,
    )

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{score:.6f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#222222",
        )

    ax.set_ylabel("Efficiency Score", fontsize=13, fontweight="600", color="#222222")
    ax.tick_params(axis="x", labelsize=10, rotation=30)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, axis="y", alpha=0.5, linewidth=0.6, color="#cccccc")
    ax.set_facecolor("#f8f8f8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    apply_spine_style(ax)
    fig.tight_layout()
    filename = cfg.pareto_filename.replace("_pareto.pdf", "_efficiency.pdf")
    save_figure(fig, output_dir / filename)


def plot_normalized_space(
    pareto_df: pd.DataFrame,
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    scores: np.ndarray,
    family_color_map: dict[str, str],
    cfg: TaskConfig,
    output_dir: Path,
) -> None:
    """Plot models in normalised latency/performance space with ideal/worst corners."""
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    ax.set_facecolor("#f8f8f8")
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.08)

    best_idx = int(np.argmax(scores))

    for idx, (xn, yn) in enumerate(zip(x_norm, y_norm)):
        line_color = "#e74c3c" if idx == best_idx else "#aaaaaa"
        line_width = 1.8 if idx == best_idx else 1.0
        ax.plot(
            [xn, 0],
            [yn, 1],
            color=line_color,
            linewidth=line_width,
            linestyle="--",
            zorder=2,
            alpha=0.85,
        )

    legend_handles: list[plt.Artist] = []
    seen_families: set[str] = set()
    seen_markers: set[str] = set()
    for i, row in pareto_df.reset_index(drop=True).iterrows():
        color = family_color_map.get(row["family"], "#555555")
        size = extract_model_size(row["model_name"])
        marker = _SIZE_MARKERS.get(size, "o")
        ax.scatter(
            x_norm[i],
            y_norm[i],
            color=color,
            marker=marker,
            s=110,
            zorder=4,
            edgecolors="white",
            linewidths=1.5,
        )
        ax.annotate(
            f"  {row['model_name']}",
            (x_norm[i], y_norm[i]),
            xytext=(6, -4),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="#222222",
            va="center",
            ha="left",
        )
        family = row["family"]
        if family not in seen_families:
            seen_families.add(family)
            label = _FAMILY_DISPLAY.get(family, family.replace("-cls", ""))
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=8,
                    label=label,
                )
            )
        if marker not in seen_markers:
            seen_markers.add(marker)

    legend_handles.append(plt.Line2D([0], [0], linewidth=0, label=""))
    for size in _SIZE_ORDER:
        marker = _SIZE_MARKERS[size]
        if marker in seen_markers:
            ms = _LEGEND_MARKER_SIZE.get(marker, 9)
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color="w",
                    markerfacecolor="#666666",
                    markersize=ms,
                    label=_SIZE_DISPLAY.get(size, size),
                )
            )
            seen_markers.discard(marker)  # prevent duplicates for shared markers

    ax.legend(handles=legend_handles, fontsize=9, framealpha=0.8, loc="lower right")

    ax.scatter(0, 1, marker="*", s=300, color="#27ae60", zorder=5)
    ax.annotate(
        "Ideal\n(fastest + best)",
        (0, 1),
        xytext=(8, -8),
        textcoords="offset points",
        fontsize=9,
        color="#27ae60",
        fontweight="bold",
        va="top",
    )

    ax.scatter(1, 0, marker="X", s=180, color="#e74c3c", zorder=5)
    ax.annotate(
        "Worst\n(slowest + weakest)",
        (1, 0),
        xytext=(-8, 8),
        textcoords="offset points",
        fontsize=9,
        color="#e74c3c",
        fontweight="bold",
        va="bottom",
        ha="right",
    )

    ax.set_xlabel("Normalised Latency", fontsize=12, fontweight="600", color="#222222")
    ax.set_ylabel(
        "Normalised Performance", fontsize=12, fontweight="600", color="#222222"
    )
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, alpha=0.4, linewidth=0.6, color="#cccccc")
    apply_spine_style(ax)
    fig.tight_layout()
    filename = cfg.pareto_filename.replace("_pareto.pdf", "_normalized_space.pdf")
    save_figure(fig, output_dir / filename)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_family_plots(
    df: pd.DataFrame,
    families: dict[str, pd.DataFrame],
    cfg: TaskConfig,
    output_dir: Path,
    only_families: list[str] | None,
) -> None:
    shared_y_range = compute_shared_y_range(families, cfg.y_col, cfg.y_err_col)
    for family_name, family_df in families.items():
        if only_families and family_name not in only_families:
            continue
        plot_family_performance(family_df, family_name, cfg, output_dir, shared_y_range)


def run_pareto_plots(
    df: pd.DataFrame,
    cfg: TaskConfig,
    output_dir: Path,
    plots: set[str],
) -> None:
    pareto_mask = compute_pareto_frontier(df, X_COL, cfg.y_col)
    pareto_df = df[pareto_mask].copy()
    non_pareto_df = df[~pareto_mask].copy()

    pareto_df["family"] = (
        pareto_df["model_name"]
        .apply(extract_model_family)
        .str.replace(r"-\d{3}$", "", regex=True)
    )
    pareto_df = pareto_df.sort_values(X_COL).reset_index(drop=True)
    family_color_map = build_family_color_map(pareto_df)

    x_norm, y_norm, scores = compute_efficiency_scores(
        pareto_df, X_COL, cfg.y_col, "linear"
    )
    if "pareto" in plots:
        plot_pareto_frontier(
            df, pareto_df, non_pareto_df, family_color_map, cfg, output_dir
        )
    if "efficiency" in plots:
        plot_efficiency_bar(pareto_df, scores, family_color_map, cfg, output_dir)
    if "normalized" in plots:
        plot_normalized_space(
            pareto_df, x_norm, y_norm, scores, family_color_map, cfg, output_dir
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate model performance visualizations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--task",
        choices=["detect", "cls", "all"],
        default="all",
        help="Which task type to generate graphs for (default: all)",
    )
    parser.add_argument(
        "--plot",
        nargs="+",
        choices=["family", "pareto", "efficiency", "normalized", "all"],
        default=["all"],
        help="Which plot types to generate (default: all)",
    )
    parser.add_argument(
        "--family",
        nargs="+",
        metavar="NAME",
        help="Limit family plots to these family names (e.g. yolo11 convnextv2)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("graphs"),
        help="Output directory (default: graphs/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plots: set[str] = (
        {"family", "pareto", "efficiency", "normalized"}
        if "all" in args.plot
        else set(args.plot)
    )
    run_detection = args.task in ("detect", "all")
    run_classification = args.task in ("cls", "all")

    setup_plot_style()
    args.output.mkdir(exist_ok=True)

    if run_detection:
        det_df = pd.read_csv(DETECTION.csv_file)
        det_families = group_models_by_family(det_df)
        print(f"Detection families: {', '.join(det_families)}")
        if "family" in plots:
            run_family_plots(det_df, det_families, DETECTION, args.output, args.family)
        if plots & {"pareto", "efficiency", "normalized"}:
            run_pareto_plots(det_df, DETECTION, args.output, plots)

    if run_classification:
        cls_df = pd.read_csv(CLASSIFICATION.csv_file)
        cls_families = group_models_by_family(cls_df)
        print(f"Classification families: {', '.join(cls_families)}")
        if "family" in plots:
            run_family_plots(
                cls_df, cls_families, CLASSIFICATION, args.output, args.family
            )
        if plots & {"pareto", "efficiency", "normalized"}:
            cls_pareto_df = cls_df[
                ~cls_df["model_name"].str.contains(r"^yolo", case=False, regex=True)
            ]
            run_pareto_plots(cls_pareto_df, CLASSIFICATION, args.output, plots)

    print(f"\nDone. Graphs saved to {args.output}/")


if __name__ == "__main__":
    main()
