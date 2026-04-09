"""Prompt helpers for formatting raw opening bars and parsing LLM output."""
import re

import pandas as pd

SYSTEM_PROMPT = """
You are a trading strategy evaluator. Your job is to look at
market signals from the first 5 minutes of the NYSE trading
session for SPY and decide whether the conditions are good
enough to take a trade that day.

You will be given the five 1-minute bars from the opening
window (09:30 through 09:34 ET). Based on that sequence, you
need to output exactly three things:

1. DECISION: Either "TAKE TRADE" or "PASS TRADE". These are
   the only two valid answers.

2. CONFIDENCE: A number from 0 to 100 showing how sure you
   are. 0 means you have no idea, 100 means you are
   completely certain.

3. EXPLANATION: 2 to 4 sentences explaining why you made
   that decision. You must reference the actual signal
   values you were given. Do not make up information that
   was not in the input.

Reply in this exact format every single time, with no extra
text before or after:

DECISION: [TAKE TRADE or PASS TRADE]
CONFIDENCE: [0-100]
EXPLANATION: [Your reasoning here]
"""


def raw_minutes_to_text(raw_day: pd.DataFrame) -> str:
    """Format one trading day's 5 opening bars as the model input."""
    ordered = raw_day.sort_values("time").reset_index(drop=True)
    if ordered.empty:
        return "No opening bars were provided."

    date = ordered.iloc[0]["date"]
    bars = "\n".join(
        (
            f"- {bar['time']}: "
            f"open={bar['open']:.2f}, high={bar['high']:.2f}, "
            f"low={bar['low']:.2f}, close={bar['close']:.2f}, "
            f"volume={int(bar['volume'])}"
        )
        for _, bar in ordered.iterrows()
    )

    return f"""Date: {date}
Instrument: SPY
Opening Sequence (09:30 to 09:34 ET):
{bars}
"""


def features_to_text(row):
    """Format engineered features for retrieval summaries and legacy prompts."""
    if row["volatility"] > 0.015:
        vol_label = "HIGH"
    elif row["volatility"] > 0.007:
        vol_label = "MODERATE"
    else:
        vol_label = "LOW"

    if row["volume_ratio"] > 1.1:
        volume_context = "above"
    elif row["volume_ratio"] < 0.9:
        volume_context = "below"
    else:
        volume_context = "in line with"

    breakout = row["breakout_direction"].upper()

    return f"""Date: {row['date']}
Instrument: SPY
Prior Close: {row['previous_close']:.2f}
Today's Open: {row['spy_open']:.2f}
Gap from Prior Close: {row['gap_pct']:+.2f}%

Opening Window Signals (09:30 to 09:34):
- Breakout Direction: {breakout}
- First 1-Minute Return: {row['first_1m_return']:+.2f}%
- Net Movement over Full 5-Minute Window: {row['net_movement']:+.2f}%
- Opening Range High: {row['opening_range_high']:.2f}
- Opening Range Low: {row['opening_range_low']:.2f}
- Opening Range Width: {row['opening_range_width']:.2f} points
- Volatility: {vol_label} (average intrabar range across 5 candles)
- Volume: {row['volume']/1e6:.2f}M shares, {volume_context} the 20-day average
"""


def parse_llm_output(raw_response):
    decision_match = re.search(r"DECISION:\s*(TAKE TRADE|PASS TRADE)", raw_response, re.IGNORECASE)
    confidence_match = re.search(r"CONFIDENCE:\s*(\d+)", raw_response)
    explanation_match = re.search(r"EXPLANATION:\s*(.+)", raw_response, re.DOTALL)

    return {
        "decision": decision_match.group(1).upper() if decision_match else None,
        "confidence": int(confidence_match.group(1)) if confidence_match else None,
        "explanation": explanation_match.group(1).strip() if explanation_match else None,
        "parse_error": not all([decision_match, confidence_match, explanation_match]),
    }
