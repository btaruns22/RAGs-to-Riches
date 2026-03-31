# Breakout Strategy
# Labels a 9:30–9:34 SPY opening window as TAKE or PASS.
#
# Ground truth: close of the 5-minute candle (09:34 close = net_movement basis).
# A setup is TAKE if there is a clear directional move, confirmed by above-average
# volume, and a meaningful price move by the end of the window.
#
# To add a new strategy, create a new file (e.g. mean_reversion_strategy.py)
# following the same interface: a module-level THRESHOLDS dict and a label(row) function.

# TODO: finalize these thresholds based on the defined strategy rules.
THRESHOLDS = {
    "min_volume_ratio": 1.2,   # opening volume must be at least 20% above 20-day avg
    "min_net_movement": 0.2,   # price must move at least 0.2% across the 5-min window
}


def label(row: dict) -> str:
    """
    Apply breakout strategy rules to a feature row.

    Parameters
    ----------
    row : dict
        One row from spy_open_features.csv. Must contain:
        breakout_direction, volume_ratio, net_movement.

    Returns
    -------
    "TAKE" or "PASS"
    """
    if row["breakout_direction"] == "NONE":
        return "PASS"
    if row["volume_ratio"] < THRESHOLDS["min_volume_ratio"]:
        return "PASS"
    if abs(row["net_movement"]) < THRESHOLDS["min_net_movement"]:
        return "PASS"
    return "TAKE"
