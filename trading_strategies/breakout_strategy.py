"""Momentum breakout strategy used for deterministic labeling."""

THRESHOLDS = {
    "breakout_direction": "UP",
    "min_net_movement": 0.25,
    "min_volume_ratio": 1.2,
    "min_opening_range_width": 0.3,
    "min_first_1m_return": 0.0,
}

RULE_DOCUMENT = """
Momentum Breakout Strategy:

TAKE if:
- breakout_direction = UP
- net_movement >= 0.25%
- volume_ratio >= 1.2
- opening_range_width >= 0.3
- first_1m_return >= 0

Otherwise PASS.
""".strip()


def label(row: dict) -> str:
    """Apply the momentum breakout strategy to one engineered feature row."""
    if row["breakout_direction"] != THRESHOLDS["breakout_direction"]:
        return "PASS"
    if row["volume_ratio"] < THRESHOLDS["min_volume_ratio"]:
        return "PASS"
    if row["net_movement"] < THRESHOLDS["min_net_movement"]:
        return "PASS"
    if row["opening_range_width"] < THRESHOLDS["min_opening_range_width"]:
        return "PASS"
    if row["first_1m_return"] < THRESHOLDS["min_first_1m_return"]:
        return "PASS"
    return "TAKE"


def format_for_retrieval(row: dict) -> str:
    """Format a labeled row as a compact retrieval document."""
    return (
        f"Date: {row['date']}\n"
        f"Gap %: {row['gap_pct']:.2f}\n"
        f"First 1m Return: {row['first_1m_return']:.2f}\n"
        f"Net Movement: {row['net_movement']:.2f}\n"
        f"Opening Range Width: {row['opening_range_width']:.2f}\n"
        f"Volatility: {row['volatility']:.2f}\n"
        f"Volume Ratio: {row['volume_ratio']:.2f}\n"
        f"Breakout Direction: {row['breakout_direction']}\n"
        f"Decision: {row.get('label', 'UNKNOWN')}\n"
    )
