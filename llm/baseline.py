"""Run the baseline LLM without any retrieved context."""
import pandas as pd
from openai import OpenAI

from prompts.prompt_utils import SYSTEM_PROMPT, parse_llm_output, raw_minutes_to_text

MODEL = "gpt-4.1-mini"


def build_baseline_messages(raw_day: pd.DataFrame) -> list[dict]:
    """Return a simple system/user message pair for the baseline model."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": raw_minutes_to_text(raw_day)},
    ]


def run_baseline(
    raw_csv: str = "data/generated/spy_open_setup_raw.csv",
    features_csv: str = "data/generated/spy_open_setup_features.csv",
    output_csv: str = "data/generated/baseline_results.csv",
    sample_size: int | None = None,
    model: str = MODEL,
) -> pd.DataFrame:
    """Run the baseline model on raw 5-bar sequences and save predictions."""
    client = OpenAI()
    raw_df = pd.read_csv(raw_csv)
    features_df = pd.read_csv(features_csv)

    if sample_size:
        features_df = features_df.sample(sample_size, random_state=42)

    results = []
    for i, row in features_df.iterrows():
        print(f"[{i + 1}/{len(features_df)}] Processing {row['date']}")
        raw_day = raw_df[raw_df["date"] == row["date"]].copy()
        if raw_day.empty:
            print(f"Skipping {row['date']} - no raw bars found")
            continue

        messages = build_baseline_messages(raw_day)

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )

        raw_output = response.choices[0].message.content or ""
        parsed = parse_llm_output(raw_output)

        results.append(
            {
                "date": row["date"],
                "true_label": row["label"],
                "predicted_label": parsed["decision"],
                "confidence": parsed["confidence"],
                "explanation": parsed["explanation"],
                "parse_error": parsed["parse_error"],
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved results to {output_csv}")
    return results_df


if __name__ == "__main__":
    run_baseline(sample_size=100)
