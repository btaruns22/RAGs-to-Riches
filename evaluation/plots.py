"""Generate evaluation plots from comparison and summary artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_inputs(
    comparison_path: str = "data/generated/comparison_results.csv",
    summary_path: str = "data/generated/evaluation_summary.json",
) -> tuple[pd.DataFrame, dict]:
    comparison_df = pd.read_csv(comparison_path)
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return comparison_df, summary


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_bars(summary: dict, output_dir: Path) -> None:
    metrics = summary["metrics"]
    labels = ["Baseline", "Manual RAG", "Vector RAG"]
    values = [
        metrics["baseline_accuracy"],
        metrics["rag_manual_accuracy"],
        metrics["rag_vector_accuracy"],
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=["#6c8ebf", "#d79b00", "#82b366"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")
    ax.bar_label(bars, fmt="%.3f")
    _save(fig, output_dir / "accuracy_comparison.png")


def plot_correct_vs_wrong(comparison_df: pd.DataFrame, output_dir: Path) -> None:
    total_days = len(comparison_df)
    correct_counts = [
        int(comparison_df["baseline_correct"].sum()),
        int(comparison_df["rag_manual_correct"].sum()),
        int(comparison_df["rag_vector_correct"].sum()),
    ]
    wrong_counts = [total_days - value for value in correct_counts]

    labels = ["Baseline", "Manual RAG", "Vector RAG"]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, correct_counts, width, label="Correct", color="#82b366")
    ax.bar(x + width / 2, wrong_counts, width, label="Wrong", color="#e06666")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Days")
    ax.set_title("Correct vs Wrong Days")
    ax.legend()
    _save(fig, output_dir / "correct_vs_wrong_days.png")


def plot_confidence_histograms(comparison_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    series = [
        ("Baseline", comparison_df["baseline_confidence"], "#6c8ebf"),
        ("Manual RAG", comparison_df["rag_manual_confidence"], "#d79b00"),
        ("Vector RAG", comparison_df["rag_vector_confidence"], "#82b366"),
    ]

    for ax, (title, values, color) in zip(axes, series):
        clean = values.dropna()
        ax.hist(clean, bins=10, color=color, edgecolor="black")
        ax.set_title(title)
        ax.set_xlabel("Confidence")
        ax.set_xlim(0, 100)
    axes[0].set_ylabel("Count")
    fig.suptitle("Confidence Distribution by Model")
    _save(fig, output_dir / "confidence_histograms.png")


def _plot_confusion_matrix(
    ax: plt.Axes,
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str,
) -> None:
    labels = ["PASS", "TAKE"]
    matrix = pd.crosstab(
        pd.Categorical(y_true, categories=labels),
        pd.Categorical(y_pred, categories=labels),
        dropna=False,
    )

    image = ax.imshow(matrix.values, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix.iloc[i, j]), ha="center", va="center", color="black")

    return image


def plot_confusion_matrices(comparison_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    y_true = comparison_df["ground_truth"]

    image = _plot_confusion_matrix(
        axes[0], y_true, comparison_df["baseline_predicted_label"], "Baseline"
    )
    _plot_confusion_matrix(
        axes[1], y_true, comparison_df["rag_manual_predicted_label"], "Manual RAG"
    )
    _plot_confusion_matrix(
        axes[2], y_true, comparison_df["rag_vector_predicted_label"], "Vector RAG"
    )
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8)
    fig.suptitle("Confusion Matrices")
    _save(fig, output_dir / "confusion_matrices.png")


def main(
    comparison_path: str = "data/generated/comparison_results.csv",
    summary_path: str = "data/generated/evaluation_summary.json",
    output_dir: str = "data/generated/plots",
) -> None:
    comparison_df, summary = _load_inputs(comparison_path=comparison_path, summary_path=summary_path)
    output_root = Path(output_dir)

    plot_accuracy_bars(summary, output_root)
    plot_correct_vs_wrong(comparison_df, output_root)
    plot_confidence_histograms(comparison_df, output_root)
    plot_confusion_matrices(comparison_df, output_root)

    print(f"Saved plots to {output_root}/")


if __name__ == "__main__":
    main()
