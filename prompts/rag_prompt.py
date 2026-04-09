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
                f"rvol_10d={row['rvol_10d']:.2f}, "
                f"vix_at_open={row.get('vix_at_open', 'NA')}, "
                f"outcome={row.get('outcome_label', row.get('label', 'UNKNOWN'))}, "
                f"max_gain={row.get('max_gain_reached', 'NA')}, "
                f"max_drawdown={row.get('max_drawdown_reached', 'NA')}"
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
        "Instructions for using retrieved context:\n"
        "- First evaluate the current 09:30 to 09:34 setup on its own.\n"
        "- Use the strategy rules as the primary decision framework.\n"
        "- Use historical examples as analogies, not as automatic answers.\n"
        "- If rules and examples conflict, trust the rules more than the examples.\n"
        "- If the setup is mixed or unclear, choose PASS TRADE.\n"
        "- Do not copy a retrieved label blindly. Explain why the current setup is similar or different.\n\n"
        "Retrieved Strategy Rules:\n"
        f"{rules_text}\n\n"
        "Retrieved Similar Historical Examples:\n"
        f"{examples_text}\n"
    )
