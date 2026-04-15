"""Entry point for running the full baseline/RAG evaluation sequence."""
from evaluation.evaluation import run_full_evaluation
from llm.baseline import run_baseline
from llm.rag import run_rag


def main() -> None:
    """Run baseline, manual RAG, vector RAG, then evaluate all three."""
    print("Running baseline...")
    run_baseline()

    print("\nRunning manual RAG...")
    run_rag(
        output_csv="data/generated/rag_results_manual.csv",
        retrieval_mode="manual",
    )

    print("\nRunning vector RAG...")
    run_rag(
        output_csv="data/generated/rag_results_vector.csv",
        retrieval_mode="vector",
    )

    print("\nRunning evaluation...")
    run_full_evaluation()


if __name__ == "__main__":
    main()
