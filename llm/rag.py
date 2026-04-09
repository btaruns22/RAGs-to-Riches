"""Build the RAG prompt using retrieved rules and similar examples."""
import pandas as pd

from project_config import TEST_START_DATE
from prompts.rag_prompt import build_rag_user_prompt
from prompts.prompt_utils import SYSTEM_PROMPT, parse_llm_output
from rag.knowledge_base import load_examples, load_rules
from rag.retriever import retrieve_relevant_rules, retrieve_similar_examples
from rag.vector_store import DEFAULT_VECTOR_DIR, ensure_vector_index, query_similar_examples
from services.llm_client import build_llm_client

MODEL = "openai/gpt-4o-mini"
DEFAULT_RETRIEVAL_MODE = "manual"


def build_rag_messages(
    raw_day: pd.DataFrame,
    feature_row: pd.Series,
    dataset_path: str = "data/generated/spy_open_setup_features.csv",
    raw_csv_path: str = "data/generated/spy_open_setup_raw.csv",
    rules_path: str | None = None,
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
    top_k: int = 3,
    vector_dir: str = DEFAULT_VECTOR_DIR,
) -> list[dict]:
    """Return system/user messages augmented with retrieved context."""
    examples = load_examples(dataset_path)
    rules = load_rules(rules_path)

    if retrieval_mode == "vector":
        similar_examples = query_similar_examples(
            row=feature_row,
            raw_day=raw_day,
            examples=examples,
            dataset_path=dataset_path,
            raw_csv_path=raw_csv_path,
            top_k=top_k,
            persist_dir=vector_dir,
        )
    elif retrieval_mode == "manual":
        similar_examples = retrieve_similar_examples(row=feature_row, examples=examples, top_k=top_k)
    else:
        raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode}")

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
    raw_csv: str = "data/generated/spy_open_setup_raw.csv",
    features_csv: str = "data/generated/spy_open_setup_features.csv",
    output_csv: str = "data/generated/rag_results.csv",
    rules_path: str | None = None,
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
    eval_start_date: str = TEST_START_DATE,
    sample_size: int | None = None,
    top_k: int = 3,
    model: str = MODEL,
    vector_dir: str = DEFAULT_VECTOR_DIR,
) -> pd.DataFrame:
    """Run the RAG model on raw 5-bar sequences and save predictions."""
    client = build_llm_client()
    raw_df = pd.read_csv(raw_csv)
    features_df = pd.read_csv(features_csv)
    features_df = (
        features_df[features_df["date"] >= eval_start_date]
        .sort_values("date")
        .reset_index(drop=True)
    )

    if retrieval_mode == "vector":
        ensure_vector_index(
            dataset_path=features_csv,
            raw_csv_path=raw_csv,
            persist_dir=vector_dir,
        )

    if sample_size:
        features_df = (
            features_df.sample(sample_size, random_state=42)
            .sort_values("date")
            .reset_index(drop=True)
        )

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
            raw_csv_path=raw_csv,
            rules_path=rules_path,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            vector_dir=vector_dir,
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
                "true_outcome_label": row.get("outcome_label"),
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
    run_rag(retrieval_mode=DEFAULT_RETRIEVAL_MODE)
