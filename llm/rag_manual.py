"""CLI wrapper for RAG with manual feature-distance retrieval."""
from llm.rag import run_rag


if __name__ == "__main__":
    run_rag(
        output_csv="data/generated/rag_results_manual.csv",
        retrieval_mode="manual",
    )
