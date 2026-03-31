"""Build engineered SPY opening-window datasets from Massive flat files."""
from collections import deque
from typing import Deque, Optional

import pandas as pd

import trading_strategies.breakout_strategy as strategy
from pipeline.features import (
    compute_features,
    extract_open_window,
    get_regular_close,
    list_trading_dates,
    load_spy_day,
)


def build_dataset(
    start_date: str = "2024-03-01",
    end_date: str = "2025-03-14",
    features_path: str = "spy_open_features.csv",
    raw_path: str = "spy_open_raw_minutes.csv",
) -> pd.DataFrame:
    """Build raw and feature CSV outputs for the requested trading-date range."""
    print(f"Listing trading dates {start_date} to {end_date}...")
    dates = list_trading_dates(start_date, end_date)
    print(f"Found {len(dates)} trading days.\n")

    feature_rows: list = []
    raw_rows: list = []
    prev_close: Optional[float] = None
    vol_history: Deque = deque(maxlen=20)

    for i, trade_date in enumerate(dates, 1):
        print(f"[{i:3d}/{len(dates)}] {trade_date}", end="  ")
        try:
            spy_day = load_spy_day(trade_date)

            if spy_day.empty:
                print("no SPY data - skip")
                continue

            day_close = get_regular_close(spy_day)

            if prev_close is None:
                prev_close = day_close
                print("no prev close yet - skip")
                continue

            window = extract_open_window(spy_day)
            if len(window) < 5:
                print(f"incomplete window ({len(window)} bars) - skip")
                prev_close = day_close
                continue

            for _, bar in window.iterrows():
                raw_rows.append(
                    {
                        "date": trade_date,
                        "time": bar["ts"].strftime("%H:%M"),
                        "open": bar["open"],
                        "high": bar["high"],
                        "low": bar["low"],
                        "close": bar["close"],
                        "volume": int(bar["volume"]),
                    }
                )

            features = compute_features(trade_date, window, prev_close, vol_history)
            if features:
                features["label"] = strategy.label(features)
                feature_rows.append(features)
                vol_history.append(features["volume"])
                print(
                    f"OK gap={features['gap_pct']:+.2f}% "
                    f"dir={features['breakout_direction']} "
                    f"net={features['net_movement']:+.2f}%"
                )

            prev_close = day_close

        except RuntimeError as exc:
            print(f"ERROR - {exc}")

    features_df = pd.DataFrame(feature_rows)
    features_df.to_csv(features_path, index=False)
    print(f"\nSaved {len(features_df)} rows to {features_path}")

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(raw_path, index=False)
    print(f"Saved {len(raw_df)} rows to {raw_path}")

    return features_df
