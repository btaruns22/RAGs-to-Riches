import pandas as pd
from openai import OpenAI
from prompt_utils import SYSTEM_PROMPT, features_to_text, parse_llm_output

client = OpenAI()

MODEL = "gpt-4o-mini"


def run_baseline(input_csv="spy_open_features.csv", output_csv="baseline_results.csv", sample_size=None):
    df = pd.read_csv(input_csv)

    if sample_size:
        df = df.sample(sample_size, random_state=42)

    results = []

    for i, row in df.iterrows():
        print(f"[{i+1}/{len(df)}] Processing {row['date']}")

        user_prompt = features_to_text(row)

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0  # VERY IMPORTANT for consistency
        )

        raw_output = response.choices[0].message.content
        parsed = parse_llm_output(raw_output)

        results.append({
            "date": row["date"],
            "true_label": row["label"],
            "predicted_label": parsed["decision"],
            "confidence": parsed["confidence"],
            "explanation": parsed["explanation"],
            "parse_error": parsed["parse_error"]
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

    print(f"\nSaved results to {output_csv}")
    return results_df


if __name__ == "__main__":
    run_baseline(sample_size=100)  # start small