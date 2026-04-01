STRATEGY_RULES = {
    "breakout_direction": "UP",
    "net_movement_min": 0.25,
    "volume_ratio_min": 1.2,
    "opening_range_width_min": 0.3,
    "first_1m_return_min": 0.0
}


def label_trade(row, rules=STRATEGY_RULES):
    if (
        row["breakout_direction"] == rules["breakout_direction"] and
        row["net_movement"] >= rules["net_movement_min"] and
        row["volume_ratio"] >= rules["volume_ratio_min"] and
        row["opening_range_width"] >= rules["opening_range_width_min"] and
        row["first_1m_return"] >= rules["first_1m_return_min"]
    ):
        return "TAKE"
    else:
        return "PASS"


def format_for_retrieval(row):
    return f"""
Date: {row['date']}
Gap %: {row['gap_pct']:.2f}
First 1m Return: {row['first_1m_return']:.2f}
Net Movement: {row['net_movement']:.2f}
Opening Range Width: {row['opening_range_width']:.2f}
Volatility: {row['volatility']:.2f}
Volume Ratio: {row['volume_ratio']:.2f}
Breakout Direction: {row['breakout_direction']}
Decision: {row['label']}
"""


RULE_DOCUMENT = """
Momentum Breakout Strategy:

TAKE if:
- breakout_direction = UP
- net_movement ≥ 0.25%
- volume_ratio ≥ 1.2
- opening_range_width ≥ 0.3
- first_1m_return ≥ 0

Otherwise PASS.
"""