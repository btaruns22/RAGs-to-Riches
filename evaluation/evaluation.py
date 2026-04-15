"""Metrics for comparing LLM predictions against deterministic labels."""
import json
from pathlib import Path

import pandas as pd

from evaluation.plots import main as generate_plots
from llm.baseline import MODEL as BASELINE_MODEL
from llm.rag import MODEL as RAG_MODEL
from project_config import TEST_START_DATE, TRAIN_END_DATE, TRAIN_START_DATE
from rag.vector_store import DEFAULT_COLLECTION_NAME, DEFAULT_VECTOR_DIR
from services.openrouter_embeddings import get_embedding_model


def normalize_decision_label(value):
    """Normalize model output labels like 'TAKE TRADE' to dataset labels like 'TAKE'."""
    if pd.isna(value):
        return value
    text = str(value).strip().upper()
    if text == "TAKE TRADE":
        return "TAKE"
    if text == "PASS TRADE":
        return "PASS"
    return text


def compute_accuracy(df: pd.DataFrame) -> float:
    return float((df["true_label"] == df["predicted_label"]).mean())


def compute_parse_error_rate(df: pd.DataFrame) -> float:
    return float(df["parse_error"].mean())


def compute_confidence_stats(df: pd.DataFrame) -> pd.Series:
    return df["confidence"].describe()


def evaluate(file_path: str = "data/generated/baseline_results.csv") -> dict:
    """Load an experiment CSV and print a few headline metrics."""
    df = pd.read_csv(file_path)
    df["predicted_label"] = df["predicted_label"].map(normalize_decision_label)

    accuracy = compute_accuracy(df)
    parse_error_rate = compute_parse_error_rate(df)
    confidence_stats = compute_confidence_stats(df)

    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Parse Error Rate: {parse_error_rate:.4f}")
    print("\nConfidence Stats:")
    print(confidence_stats)

    return {
        "accuracy": accuracy,
        "parse_error_rate": parse_error_rate,
        "confidence_stats": confidence_stats,
    }


def compare_runs(
    baseline_path: str = "data/generated/baseline_results.csv",
    rag_path: str = "data/generated/rag_results.csv",
    features_path: str = "data/generated/spy_open_setup_features.csv",
    output_path: str = "data/generated/comparison_results.csv",
) -> pd.DataFrame:
    """Compare baseline and RAG predictions side by side with ground truth."""
    baseline_df = pd.read_csv(baseline_path).rename(
        columns={
            "predicted_label": "baseline_predicted_label",
            "confidence": "baseline_confidence",
            "explanation": "baseline_explanation",
            "parse_error": "baseline_parse_error",
        }
    )
    rag_df = pd.read_csv(rag_path).rename(
        columns={
            "predicted_label": "rag_predicted_label",
            "confidence": "rag_confidence",
            "explanation": "rag_explanation",
            "parse_error": "rag_parse_error",
        }
    )
    features_df = pd.read_csv(features_path)[["date", "label", "outcome_label"]].rename(
        columns={"label": "ground_truth", "outcome_label": "ground_truth_outcome"}
    )

    baseline_df = baseline_df.drop(columns=["true_label"], errors="ignore")
    rag_df = rag_df.drop(columns=["true_label"], errors="ignore")

    comparison_df = features_df.merge(baseline_df, on="date", how="left").merge(rag_df, on="date", how="left")

    comparison_df["baseline_correct"] = (
        comparison_df["baseline_predicted_label"] == comparison_df["ground_truth"]
    )
    comparison_df["rag_correct"] = comparison_df["rag_predicted_label"] == comparison_df["ground_truth"]

    baseline_accuracy = float(comparison_df["baseline_correct"].mean())
    rag_accuracy = float(comparison_df["rag_correct"].mean())
    baseline_parse_error_rate = float(comparison_df["baseline_parse_error"].mean())
    rag_parse_error_rate = float(comparison_df["rag_parse_error"].mean())

    print("\n=== Side-by-Side Comparison ===")
    print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
    print(f"RAG Accuracy:      {rag_accuracy:.4f}")
    print(f"Baseline Parse Error Rate: {baseline_parse_error_rate:.4f}")
    print(f"RAG Parse Error Rate:      {rag_parse_error_rate:.4f}")

    comparison_df.to_csv(output_path, index=False)
    print(f"\nSaved comparison output to {output_path}")

    return comparison_df


def compare_three_runs(
    baseline_path: str = "data/generated/baseline_results.csv",
    rag_manual_path: str = "data/generated/rag_results_manual.csv",
    rag_vector_path: str = "data/generated/rag_results_vector.csv",
    features_path: str = "data/generated/spy_open_setup_features.csv",
    output_path: str = "data/generated/comparison_results.csv",
    summary_path: str = "data/generated/evaluation_summary.json",
) -> pd.DataFrame:
    """Compare baseline, manual RAG, and vector RAG side by side with ground truth."""
    features_df = pd.read_csv(features_path)[["date", "label", "outcome_label"]].rename(
        columns={"label": "ground_truth", "outcome_label": "ground_truth_outcome"}
    )

    baseline_df = pd.read_csv(baseline_path).rename(
        columns={
            "predicted_label": "baseline_predicted_label",
            "confidence": "baseline_confidence",
            "explanation": "baseline_explanation",
            "parse_error": "baseline_parse_error",
        }
    ).drop(columns=["true_label", "true_outcome_label"], errors="ignore")
    baseline_df["baseline_predicted_label"] = baseline_df["baseline_predicted_label"].map(
        normalize_decision_label
    )

    rag_manual_df = pd.read_csv(rag_manual_path).rename(
        columns={
            "predicted_label": "rag_manual_predicted_label",
            "confidence": "rag_manual_confidence",
            "explanation": "rag_manual_explanation",
            "parse_error": "rag_manual_parse_error",
        }
    ).drop(columns=["true_label", "true_outcome_label"], errors="ignore")
    rag_manual_df["rag_manual_predicted_label"] = rag_manual_df["rag_manual_predicted_label"].map(
        normalize_decision_label
    )

    rag_vector_df = pd.read_csv(rag_vector_path).rename(
        columns={
            "predicted_label": "rag_vector_predicted_label",
            "confidence": "rag_vector_confidence",
            "explanation": "rag_vector_explanation",
            "parse_error": "rag_vector_parse_error",
        }
    ).drop(columns=["true_label", "true_outcome_label"], errors="ignore")
    rag_vector_df["rag_vector_predicted_label"] = rag_vector_df["rag_vector_predicted_label"].map(
        normalize_decision_label
    )

    comparison_df = (
        features_df
        .merge(baseline_df, on="date", how="left")
        .merge(rag_manual_df, on="date", how="left")
        .merge(rag_vector_df, on="date", how="left")
    )

    comparison_df["baseline_correct"] = (
        comparison_df["baseline_predicted_label"] == comparison_df["ground_truth"]
    )
    comparison_df["rag_manual_correct"] = (
        comparison_df["rag_manual_predicted_label"] == comparison_df["ground_truth"]
    )
    comparison_df["rag_vector_correct"] = (
        comparison_df["rag_vector_predicted_label"] == comparison_df["ground_truth"]
    )

    baseline_accuracy = float(comparison_df["baseline_correct"].mean())
    rag_manual_accuracy = float(comparison_df["rag_manual_correct"].mean())
    rag_vector_accuracy = float(comparison_df["rag_vector_correct"].mean())
    baseline_parse_error_rate = float(comparison_df["baseline_parse_error"].mean())
    rag_manual_parse_error_rate = float(comparison_df["rag_manual_parse_error"].mean())
    rag_vector_parse_error_rate = float(comparison_df["rag_vector_parse_error"].mean())

    print("\n=== Three-Way Comparison ===")
    print(f"Baseline Accuracy:   {baseline_accuracy:.4f}")
    print(f"Manual RAG Accuracy: {rag_manual_accuracy:.4f}")
    print(f"Vector RAG Accuracy: {rag_vector_accuracy:.4f}")
    print(f"Baseline Parse Error Rate:   {baseline_parse_error_rate:.4f}")
    print(f"Manual RAG Parse Error Rate: {rag_manual_parse_error_rate:.4f}")
    print(f"Vector RAG Parse Error Rate: {rag_vector_parse_error_rate:.4f}")

    comparison_df.to_csv(output_path, index=False)
    print(f"\nSaved comparison output to {output_path}")

    summary = {
        "train_period": {
            "start_date": TRAIN_START_DATE,
            "end_date": TRAIN_END_DATE,
        },
        "test_period": {
            "start_date": TEST_START_DATE,
            "end_date": "present",
        },
        "models": {
            "baseline": {
                "provider_model": BASELINE_MODEL,
                "result_file": baseline_path,
            },
            "rag_manual": {
                "provider_model": RAG_MODEL,
                "retrieval_mode": "manual",
                "result_file": rag_manual_path,
            },
            "rag_vector": {
                "provider_model": RAG_MODEL,
                "retrieval_mode": "vector",
                "embedding_model": get_embedding_model(),
                "vector_store_dir": DEFAULT_VECTOR_DIR,
                "collection_name": DEFAULT_COLLECTION_NAME,
                "result_file": rag_vector_path,
            },
        },
        "metrics": {
            "baseline_accuracy": baseline_accuracy,
            "rag_manual_accuracy": rag_manual_accuracy,
            "rag_vector_accuracy": rag_vector_accuracy,
            "baseline_parse_error_rate": baseline_parse_error_rate,
            "rag_manual_parse_error_rate": rag_manual_parse_error_rate,
            "rag_vector_parse_error_rate": rag_vector_parse_error_rate,
            "num_evaluated_rows": int(len(comparison_df)),
        },
    }
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved evaluation summary to {summary_path}")
    return comparison_df


def summarize_comparison(comparison_df: pd.DataFrame) -> None:
    """Print a compact summary of where RAG helps or hurts versus baseline."""
    rag_only_correct = comparison_df[
        (~comparison_df["baseline_correct"]) & (comparison_df["rag_correct"])
    ]
    baseline_only_correct = comparison_df[
        (comparison_df["baseline_correct"]) & (~comparison_df["rag_correct"])
    ]
    both_wrong = comparison_df[
        (~comparison_df["baseline_correct"]) & (~comparison_df["rag_correct"])
    ]

    print("\n=== Outcome Breakdown ===")
    print(f"RAG-only correct days:      {len(rag_only_correct)}")
    print(f"Baseline-only correct days: {len(baseline_only_correct)}")
    print(f"Both wrong days:            {len(both_wrong)}")


def summarize_three_way_comparison(comparison_df: pd.DataFrame) -> None:
    """Print a simple per-model correct vs wrong day breakdown."""
    baseline_correct_days = int(comparison_df["baseline_correct"].sum())
    rag_manual_correct_days = int(comparison_df["rag_manual_correct"].sum())
    rag_vector_correct_days = int(comparison_df["rag_vector_correct"].sum())
    total_days = int(len(comparison_df))

    print("\n=== Correct vs Wrong Days ===")
    print(f"Baseline:   correct={baseline_correct_days} wrong={total_days - baseline_correct_days}")
    print(f"Manual RAG: correct={rag_manual_correct_days} wrong={total_days - rag_manual_correct_days}")
    print(f"Vector RAG: correct={rag_vector_correct_days} wrong={total_days - rag_vector_correct_days}")


def run_full_evaluation(
    baseline_path: str = "data/generated/baseline_results.csv",
    rag_manual_path: str = "data/generated/rag_results_manual.csv",
    rag_vector_path: str = "data/generated/rag_results_vector.csv",
    features_path: str = "data/generated/spy_open_setup_features.csv",
    comparison_path: str = "data/generated/comparison_results.csv",
    summary_path: str = "data/generated/evaluation_summary.json",
    plots_output_dir: str = "data/generated/plots",
) -> pd.DataFrame:
    """Run comparison, print summaries, and generate plots from the evaluation artifacts."""
    comparison_df = compare_three_runs(
        baseline_path=baseline_path,
        rag_manual_path=rag_manual_path,
        rag_vector_path=rag_vector_path,
        features_path=features_path,
        output_path=comparison_path,
        summary_path=summary_path,
    )
    summarize_three_way_comparison(comparison_df)
    generate_plots(
        comparison_path=comparison_path,
        summary_path=summary_path,
        output_dir=plots_output_dir,
    )
    return comparison_df


if __name__ == "__main__":
    run_full_evaluation()
