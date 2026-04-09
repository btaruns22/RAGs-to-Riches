"""Path-dependent breakout labeling for the opening-window setup."""
from typing import Tuple

import pandas as pd

from project_config import PROFIT_TARGET_PCT, STOP_LOSS_PCT

RULE_DOCUMENT = """
Opening Window Breakout Outcome Rules:

Feature window:
- Use SPY 09:30 through 09:34 bars to describe the setup.

Outcome window:
- Starting from the 09:34 close as the entry price, scan bars from 09:35 through 10:30.
- If price reaches entry + 0.30% before entry - 0.20%, label TAKE.
- If price reaches entry - 0.20% first, label FAIL_FAKEOUT.
- If neither threshold is reached by 10:30, label PASS.

Same-bar rule:
- If a single minute bar hits both the profit target and the stop-loss range, treat it as FAIL_FAKEOUT.
  This is the conservative assumption because intrabar order cannot be recovered from OHLC alone.
""".strip()


def label_outcome(entry_price: float, outcome_window: pd.DataFrame) -> Tuple[str, float, float]:
    """Label a setup using the 09:35-10:30 path after the 09:34 close entry."""
    if outcome_window.empty:
        return "PASS", 0.0, 0.0

    profit_level = entry_price * (1 + PROFIT_TARGET_PCT)
    stop_level = entry_price * (1 - STOP_LOSS_PCT)

    max_gain_reached = float(((outcome_window["high"].max() - entry_price) / entry_price) * 100)
    max_drawdown_reached = float(((outcome_window["low"].min() - entry_price) / entry_price) * 100)

    for _, bar in outcome_window.sort_values("ts").iterrows():
        hit_target = float(bar["high"]) >= profit_level
        hit_stop = float(bar["low"]) <= stop_level

        if hit_target and hit_stop:
            return "FAIL_FAKEOUT", round(max_gain_reached, 4), round(max_drawdown_reached, 4)
        if hit_stop:
            return "FAIL_FAKEOUT", round(max_gain_reached, 4), round(max_drawdown_reached, 4)
        if hit_target:
            return "TAKE", round(max_gain_reached, 4), round(max_drawdown_reached, 4)

    return "PASS", round(max_gain_reached, 4), round(max_drawdown_reached, 4)


def decision_label(outcome_label: str) -> str:
    """Collapse path-dependent outcomes to the binary LLM decision task."""
    return "TAKE" if outcome_label == "TAKE" else "PASS"


def format_for_retrieval(row: dict) -> str:
    """Format a labeled row as a compact retrieval document."""
    return (
        f"Date: {row['date']}\n"
        f"Gap %: {row['gap_pct']:.2f}\n"
        f"First 1m Return: {row['first_1m_return']:.2f}\n"
        f"Net Movement: {row['net_movement']:.2f}\n"
        f"Opening Range Width: {row['opening_range_width']:.2f}\n"
        f"Volatility: {row['volatility']:.2f}\n"
        f"RVOL 10D: {row['rvol_10d']:.2f}\n"
        f"VIX At Open: {row.get('vix_at_open', 'UNKNOWN')}\n"
        f"Breakout Direction: {row['breakout_direction']}\n"
        f"Outcome Label: {row.get('outcome_label', 'UNKNOWN')}\n"
    )
