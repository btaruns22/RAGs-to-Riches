"""Simple retrieval utilities for pulling similar historical examples."""
from __future__ import annotations

from typing import Iterable, List

import pandas as pd


SIMILARITY_COLUMNS = [
    "gap_pct",
    "first_1m_return",
    "net_movement",
    "volatility",
    "rvol_10d",
    "vix_at_open",
]


def _distance(row: pd.Series, candidate: pd.Series) -> float:
    total = 0.0
    for column in SIMILARITY_COLUMNS:
        total += abs(float(candidate[column]) - float(row[column]))
    return total


def retrieve_similar_examples(
    row: pd.Series,
    examples: pd.DataFrame,
    top_k: int = 3,
) -> pd.DataFrame:
    """Return the top-k most similar historical examples for a feature row."""
    if examples.empty:
        return examples

    comparable = examples.copy()
    comparable = comparable[comparable["date"] != row["date"]]
    if comparable.empty:
        return comparable

    comparable["_distance"] = comparable.apply(lambda candidate: _distance(row, candidate), axis=1)
    return comparable.nsmallest(top_k, "_distance").drop(columns="_distance")


def retrieve_relevant_rules(
    row: pd.Series,
    rules: Iterable[str],
) -> List[str]:
    """Return the subset of rules most relevant to the current opening setup."""
    selected = list(rules)

    if row["breakout_direction"] == "NONE":
        selected.append("No directional breakout should bias the system toward PASS.")
    if row["rvol_10d"] >= 1.2:
        selected.append("Above-average volume supports a breakout continuation thesis.")
    if abs(row["net_movement"]) < 0.2:
        selected.append("Weak 5-minute net movement reduces conviction in the setup.")
    if row.get("vix_at_open") is not None and float(row["vix_at_open"]) >= 25:
        selected.append("Elevated VIX suggests a high-volatility regime that can distort breakout reliability.")

    seen = set()
    ordered = []
    for rule in selected:
        if rule not in seen:
            seen.add(rule)
            ordered.append(rule)
    return ordered[:5]
