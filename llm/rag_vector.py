"""CLI wrapper for RAG with local Chroma vector retrieval."""
from llm.rag import run_rag


if __name__ == "__main__":
    run_rag(
        output_csv="data/generated/rag_results_vector.csv",
        retrieval_mode="vector",
        sample_size=100,
    )
