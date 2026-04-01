"""Run the baseline LLM without any retrieved context."""
import pandas as pd
from openai import OpenAI

from prompts.prompt_utils import SYSTEM_PROMPT, features_to_text
from prompts.prompt_utils import parse_llm_output

MODEL = "gpt-4.1-mini"


def build_baseline_messages(row: dict) -> list[dict]:
    """Return a simple system/user message pair for the baseline model."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": features_to_text(row)},
    ]


def run_baseline(
    input_csv: str = "spy_open_features.csv",
    output_csv: str = "baseline_results.csv",
    sample_size: int | None = None,
    model: str = MODEL,
) -> pd.DataFrame:
    """Run the baseline model on the engineered dataset and save predictions."""
    client = OpenAI()
    df = pd.read_csv(input_csv)

    if sample_size:
        df = df.sample(sample_size, random_state=42)

    results = []
    for i, row in df.iterrows():
        print(f"[{i + 1}/{len(df)}] Processing {row['date']}")
        messages = build_baseline_messages(row)

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
