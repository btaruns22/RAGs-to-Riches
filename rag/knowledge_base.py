"""Load strategy rules and historical examples used by the RAG pipeline."""
from pathlib import Path
from typing import List

import pandas as pd

from trading_strategies.breakout_strategy import RULE_DOCUMENT


DEFAULT_RULES = [
    "A breakout setup needs a directional move, not a flat open.",
    "Higher opening volume strengthens conviction in the move.",
    "A small net movement is less reliable than a clear 5-minute move.",
    RULE_DOCUMENT,
]


def load_rules(rules_path: str | None = None) -> List[str]:
    """Load strategy rules from disk or fall back to a small default set."""
    if not rules_path:
        return DEFAULT_RULES

    path = Path(rules_path)
    if not path.exists():
        return DEFAULT_RULES

    rules = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return rules or DEFAULT_RULES


def load_examples(dataset_path: str = "spy_open_features.csv") -> pd.DataFrame:
    """Load historical labeled examples from the engineered dataset."""
    path = Path(dataset_path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
