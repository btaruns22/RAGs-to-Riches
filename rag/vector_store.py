"""Local Chroma vector store for historical SPY setups."""
from __future__ import annotations

from pathlib import Path

import chromadb
import pandas as pd

from services.openrouter_embeddings import embed_texts, get_embedding_model
from prompts.prompt_utils import raw_minutes_to_text

DEFAULT_VECTOR_DIR = "data/generated/chroma"
DEFAULT_COLLECTION_NAME = "spy_open_setup_features"


def _format_feature_summary(row: pd.Series, include_label: bool = True) -> str:
    """Format the engineered summary section for one historical setup."""
    lines = [
        "Engineered Summary:",
        f"Previous Close: {row['previous_close']:.4f}",
        f"SPY Open: {row['spy_open']:.4f}",
        f"Gap %: {row['gap_pct']:.4f}",
        f"First 1m Return: {row['first_1m_return']:.4f}",
        f"Net Movement: {row['net_movement']:.4f}",
        f"Opening Range High: {row['opening_range_high']:.4f}",
        f"Opening Range Low: {row['opening_range_low']:.4f}",
        f"Opening Range Width: {row['opening_range_width']:.4f}",
        f"Volatility: {row['volatility']:.4f}",
        f"Volume: {int(row['volume'])}",
        f"Volume Ratio: {row['volume_ratio']:.4f}",
        f"Breakout Direction: {row['breakout_direction']}",
    ]
    if include_label and "label" in row:
        lines.append(f"Label: {row['label']}")
    return "\n".join(lines)


def format_retrieval_document(
    row: pd.Series,
    raw_day: pd.DataFrame,
    include_label: bool = True,
) -> str:
    """Format one historical setup as a combined raw-bars-plus-features document."""
    raw_text = raw_minutes_to_text(raw_day).strip()
    summary_text = _format_feature_summary(row, include_label=include_label)
    return f"{raw_text}\n\n{summary_text}"


def _get_client(persist_dir: str = DEFAULT_VECTOR_DIR):
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def _get_or_create_collection(persist_dir: str = DEFAULT_VECTOR_DIR):
    client = _get_client(persist_dir)
    return client.get_or_create_collection(name=DEFAULT_COLLECTION_NAME)


def build_vector_index(
    dataset_path: str = "data/generated/spy_open_setup_features.csv",
    raw_csv_path: str = "data/generated/spy_open_setup_raw.csv",
    persist_dir: str = DEFAULT_VECTOR_DIR,
) -> int:
    """Build or refresh the local Chroma collection from the feature dataset."""
    df = pd.read_csv(dataset_path)
    raw_df = pd.read_csv(raw_csv_path)
    client = _get_client(persist_dir)

    try:
        client.delete_collection(DEFAULT_COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(name=DEFAULT_COLLECTION_NAME)
    documents = []
    for _, row in df.iterrows():
        raw_day = raw_df[raw_df["date"] == row["date"]].copy()
        documents.append(format_retrieval_document(row, raw_day=raw_day, include_label=True))

    embeddings = embed_texts(documents)
    metadatas = [
        {
            "date": row["date"],
            "label": row["label"],
            "breakout_direction": row["breakout_direction"],
        }
        for _, row in df.iterrows()
    ]

    collection.add(
        ids=df["date"].tolist(),
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(df)


def ensure_vector_index(
    dataset_path: str = "data/generated/spy_open_setup_features.csv",
    raw_csv_path: str = "data/generated/spy_open_setup_raw.csv",
    persist_dir: str = DEFAULT_VECTOR_DIR,
) -> None:
    """Build the vector store if the local collection does not exist yet."""
    collection = _get_or_create_collection(persist_dir)
    if collection.count() == 0:
        build_vector_index(dataset_path=dataset_path, raw_csv_path=raw_csv_path, persist_dir=persist_dir)


def query_similar_examples(
    row: pd.Series,
    raw_day: pd.DataFrame,
    examples: pd.DataFrame,
    dataset_path: str = "data/generated/spy_open_setup_features.csv",
    raw_csv_path: str = "data/generated/spy_open_setup_raw.csv",
    top_k: int = 3,
    persist_dir: str = DEFAULT_VECTOR_DIR,
) -> pd.DataFrame:
    """Query local Chroma for the nearest historical setups, excluding the current date."""
    ensure_vector_index(
        dataset_path=dataset_path,
        raw_csv_path=raw_csv_path,
        persist_dir=persist_dir,
    )
    collection = _get_or_create_collection(persist_dir)

    query_text = format_retrieval_document(row, raw_day=raw_day, include_label=False)
    query_embedding = embed_texts([query_text])[0]
    query_size = min(max(top_k + 5, 10), max(collection.count(), 1))

    response = collection.query(
        query_embeddings=[query_embedding],
        n_results=query_size,
        include=["metadatas"],
    )

    ordered_dates = []
    for metadata in response.get("metadatas", [[]])[0]:
        if not metadata:
            continue
        date = metadata.get("date")
        if date and date != row["date"]:
            ordered_dates.append(date)
        if len(ordered_dates) >= top_k:
            break

    if not ordered_dates:
        return examples.iloc[0:0].copy()

    indexed = examples.set_index("date")
    result = indexed.loc[[date for date in ordered_dates if date in indexed.index]].reset_index()
    return result


if __name__ == "__main__":
    count = build_vector_index()
    print(
        f"Built local Chroma index with {count} setups "
        f"using {get_embedding_model()} in {DEFAULT_VECTOR_DIR}/"
    )
