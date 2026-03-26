# prompt_utils.py
import re

SYSTEM_PROMPT = """
You are a trading strategy evaluator. Your job is to look at 
market signals from the first 5 minutes of the NYSE trading 
session for SPY and decide whether the conditions are good 
enough to take a trade that day.

You will be given a summary of signals from the opening 
5-minute window. Based on those signals, you need to output 
exactly three things:

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


def features_to_text(row):
    # Volatility: convert average intrabar range to a readable label
    if row["volatility"] > 0.015:
        vol_label = "HIGH"
    elif row["volatility"] > 0.007:
        vol_label = "MODERATE"
    else:
        vol_label = "LOW"

    # Volume ratio: compare opening window volume to 20-day average
    if row["volume_ratio"] > 1.1:
        vol_rel = "above"
    elif row["volume_ratio"] < 0.9:
        vol_rel = "below"
    else:
        vol_rel = "in line with"

    breakout = row["breakout_direction"].upper()  # UP, DOWN, or NONE

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
- Volume: {row['volume']/1e6:.2f}M shares, {vol_rel} the 20-day average
"""


def parse_llm_output(raw_response):
    decision_match = re.search(r"DECISION:\s*(TAKE TRADE|PASS TRADE)", raw_response, re.IGNORECASE)
    confidence_match = re.search(r"CONFIDENCE:\s*(\d+)", raw_response)
    explanation_match = re.search(r"EXPLANATION:\s*(.+)", raw_response, re.DOTALL)

    return {
        "decision": decision_match.group(1).upper() if decision_match else None,
        "confidence": int(confidence_match.group(1)) if confidence_match else None,
        "explanation": explanation_match.group(1).strip() if explanation_match else None,
        "parse_error": not all([decision_match, confidence_match, explanation_match])
    }