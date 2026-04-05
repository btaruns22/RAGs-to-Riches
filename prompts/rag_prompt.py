"""Prompt builders for the retrieval-augmented evaluation flow."""
import pandas as pd

from prompts.prompt_utils import raw_minutes_to_text


def _examples_to_text(similar_examples: pd.DataFrame) -> str:
    if similar_examples.empty:
        return "No similar historical examples were retrieved."

    lines = []
    for _, row in similar_examples.iterrows():
        lines.append(
            (
                f"- {row['date']}: direction={row['breakout_direction']}, "
                f"net_movement={row['net_movement']:+.2f}%, "
                f"volume_ratio={row['volume_ratio']:.2f}, "
                f"label={row.get('label', 'UNKNOWN')}"
            )
        )
    return "\n".join(lines)


def build_rag_user_prompt(
    raw_day: pd.DataFrame,
    rules: list[str],
    similar_examples: pd.DataFrame,
) -> str:
    """Combine one day's raw opening bars with retrieved context."""
    rules_text = "\n".join(f"- {rule}" for rule in rules) if rules else "- No rules retrieved."
    examples_text = _examples_to_text(similar_examples)

    return (
        f"{raw_minutes_to_text(raw_day)}\n"
        "Retrieved Strategy Rules:\n"
        f"{rules_text}\n\n"
        "Retrieved Similar Historical Examples:\n"
        f"{examples_text}\n"
    )
