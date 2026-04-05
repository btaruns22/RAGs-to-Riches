"""Build the RAG prompt using retrieved rules and similar examples."""
import pandas as pd
from openai import OpenAI

from prompts.rag_prompt import build_rag_user_prompt
from prompts.prompt_utils import SYSTEM_PROMPT, parse_llm_output
from rag.knowledge_base import load_examples, load_rules
from rag.retriever import retrieve_relevant_rules, retrieve_similar_examples

MODEL = "gpt-4.1-mini"


def build_rag_messages(
    raw_day: pd.DataFrame,
    feature_row: pd.Series,
    dataset_path: str = "spy_open_features.csv",
    rules_path: str | None = None,
    top_k: int = 3,
) -> list[dict]:
    """Return system/user messages augmented with retrieved context."""
    examples = load_examples(dataset_path)
    rules = load_rules(rules_path)

    similar_examples = retrieve_similar_examples(row=feature_row, examples=examples, top_k=top_k)
    relevant_rules = retrieve_relevant_rules(row=feature_row, rules=rules)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_rag_user_prompt(
                raw_day=raw_day,
                rules=relevant_rules,
                similar_examples=similar_examples,
            ),
        },
    ]


def run_rag(
    raw_csv: str = "spy_open_raw_minutes.csv",
    features_csv: str = "spy_open_features.csv",
    output_csv: str = "rag_results.csv",
    rules_path: str | None = None,
    sample_size: int | None = None,
    top_k: int = 3,
    model: str = MODEL,
) -> pd.DataFrame:
    """Run the RAG model on raw 5-bar sequences and save predictions."""
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

        messages = build_rag_messages(
            raw_day=raw_day,
            feature_row=row,
            dataset_path=features_csv,
            rules_path=rules_path,
            top_k=top_k,
        )

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
    run_rag(sample_size=100)
