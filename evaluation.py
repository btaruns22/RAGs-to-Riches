import pandas as pd


def compute_accuracy(df):
    return (df["true_label"] == df["predicted_label"]).mean()


def compute_parse_error_rate(df):
    return df["parse_error"].mean()


def compute_confidence_stats(df):
    return df["confidence"].describe()


def evaluate(file_path="baseline_results.csv"):
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
        "confidence_stats": confidence_stats
    }


if __name__ == "__main__":
    evaluate()
