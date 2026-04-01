"""Build the RAG prompt using retrieved rules and similar examples."""
from prompts.rag_prompt import build_rag_user_prompt
from prompts.prompt_utils import SYSTEM_PROMPT
from rag.knowledge_base import load_examples, load_rules
from rag.retriever import retrieve_relevant_rules, retrieve_similar_examples


def build_rag_messages(
    row: dict,
    dataset_path: str = "spy_open_features.csv",
    rules_path: str | None = None,
    top_k: int = 3,
) -> list[dict]:
    """Return system/user messages augmented with retrieved context."""
    examples = load_examples(dataset_path)
    rules = load_rules(rules_path)

    similar_examples = retrieve_similar_examples(row=row, examples=examples, top_k=top_k)
    relevant_rules = retrieve_relevant_rules(row=row, rules=rules)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_rag_user_prompt(
                row=row,
                rules=relevant_rules,
                similar_examples=similar_examples,
            ),
        },
    ]
