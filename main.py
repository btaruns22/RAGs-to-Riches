"""Entry point for running the full baseline/RAG evaluation sequence."""


def main() -> None:
    """Run baseline, manual RAG, vector RAG, then evaluate all three."""
    print("Loading baseline runner...")
    from llm.baseline import run_baseline

    print("Running baseline...")
    run_baseline()

    print("\nLoading RAG runner...")
    from llm.rag import run_rag

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

    print("\nLoading evaluation module...")
    from evaluation.evaluation import compare_three_runs, summarize_three_way_comparison

    print("\nRunning evaluation...")
    comparison_df = compare_three_runs()
    summarize_three_way_comparison(comparison_df)


if __name__ == "__main__":
    main()
