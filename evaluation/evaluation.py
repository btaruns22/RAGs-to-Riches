"""Metrics for comparing LLM predictions against deterministic labels."""
import pandas as pd


def compute_accuracy(df: pd.DataFrame) -> float:
    return float((df["true_label"] == df["predicted_label"]).mean())


def compute_parse_error_rate(df: pd.DataFrame) -> float:
    return float(df["parse_error"].mean())


def compute_confidence_stats(df: pd.DataFrame) -> pd.Series:
    return df["confidence"].describe()


def evaluate(file_path: str = "baseline_results.csv") -> dict:
    """Load an experiment CSV and print a few headline metrics."""
    df = pd.read_csv(file_path)

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
    baseline_path: str = "baseline_results.csv",
    rag_path: str = "rag_results.csv",
    features_path: str = "spy_open_features.csv",
    output_path: str = "comparison_results.csv",
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
    features_df = pd.read_csv(features_path)[["date", "label"]].rename(columns={"label": "ground_truth"})

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


if __name__ == "__main__":
    comparison_df = compare_runs()
    summarize_comparison(comparison_df)
