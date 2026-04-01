"""Build the baseline prompt without any retrieved context."""
from prompts.prompt_utils import SYSTEM_PROMPT, features_to_text


def build_baseline_messages(row: dict) -> list[dict]:
    """Return a simple system/user message pair for the baseline model."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": features_to_text(row)},
    ]
