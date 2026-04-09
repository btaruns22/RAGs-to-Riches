"""Build engineered SPY opening-window datasets from Massive flat files."""
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Deque, Optional

import pandas as pd

from project_config import DEFAULT_DATA_END_DATE, RVOL_LOOKBACK_DAYS, TRAIN_START_DATE
import trading_strategies.breakout_strategy as strategy
from pipeline.features import (
    compute_features,
    extract_open_window,
    extract_outcome_window,
    get_entry_price,
    get_regular_close,
    get_vix_at_open,
    list_trading_dates,
    load_spy_day,
    load_vix_day,
)


def build_chunk_output_paths(
    start_date: str,
    end_date: str,
    output_dir: str = "data/generated",
) -> tuple[str, str]:
    """Return chunk-specific raw/features output paths for a date range."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    prefix = f"{start_date}_{end_date}"
    features_path = output_root / f"{prefix}_spy_open_setup_features.csv"
    raw_path = output_root / f"{prefix}_spy_open_setup_raw.csv"
    return str(features_path), str(raw_path)


def _calendar_days_before(date_str: str, days: int = 14) -> str:
    """Return an ISO date a fixed number of calendar days before date_str."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").date()
    return (dt - timedelta(days=days)).isoformat()


def _load_previous_close(start_date: str, end_date: str) -> Optional[float]:
    """Load the most recent close before start_date so chunked runs keep continuity."""
    lookback_start = _calendar_days_before(start_date, days=14)
    candidate_dates = list_trading_dates(lookback_start, end_date)
    previous_dates = [trade_date for trade_date in candidate_dates if trade_date < start_date]
    if not previous_dates:
        return None

    seed_date = previous_dates[-1]
    seed_day = load_spy_day(seed_date)
    if seed_day.empty:
        return None
    return get_regular_close(seed_day)


def build_dataset(
    start_date: str = TRAIN_START_DATE,
    end_date: str = DEFAULT_DATA_END_DATE,
    features_path: str = "data/generated/spy_open_setup_features.csv",
    raw_path: str = "data/generated/spy_open_setup_raw.csv",
) -> pd.DataFrame:
    """Build raw and feature CSV outputs for the requested trading-date range."""
    print(f"Listing trading dates {start_date} to {end_date}...")
    dates = list_trading_dates(start_date, end_date)
    print(f"Found {len(dates)} trading days.\n")

    feature_rows: list = []
    raw_rows: list = []
    prev_close: Optional[float] = _load_previous_close(start_date, end_date)
    vol_history: Deque = deque(maxlen=RVOL_LOOKBACK_DAYS)

    for i, trade_date in enumerate(dates, 1):
        print(f"[{i:3d}/{len(dates)}] {trade_date}", end="  ")
        try:
            spy_day = load_spy_day(trade_date)
            vix_day = load_vix_day(trade_date)

            if spy_day.empty:
                print("no SPY data - skip")
                continue

            day_close = get_regular_close(spy_day)

            if prev_close is None:
                prev_close = day_close
                print("no prev close yet - skip")
                continue

            window = extract_open_window(spy_day)
            outcome_window = extract_outcome_window(spy_day)
            if len(window) < 5:
                print(f"incomplete window ({len(window)} bars) - skip")
                prev_close = day_close
                continue

            vix_at_open = get_vix_at_open(vix_day)
            entry_price = get_entry_price(window)
            if entry_price is None:
                print("missing 09:34 entry price - skip")
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

            features = compute_features(
                trade_date,
                window,
                prev_close,
                vol_history,
                vix_at_open=vix_at_open,
            )
            if features:
                outcome_label, max_gain_reached, max_drawdown_reached = strategy.label_outcome(
                    entry_price=entry_price,
                    outcome_window=outcome_window,
                )
                features["entry_price"] = round(entry_price, 4)
                features["outcome_label"] = outcome_label
                features["label"] = strategy.decision_label(outcome_label)
                features["max_gain_reached"] = max_gain_reached
                features["max_drawdown_reached"] = max_drawdown_reached
                feature_rows.append(features)
                vol_history.append(features["volume"])
                print(
                    f"OK gap={features['gap_pct']:+.2f}% "
                    f"dir={features['breakout_direction']} "
                    f"net={features['net_movement']:+.2f}% "
                    f"outcome={features['outcome_label']} "
                    f"vix={features['vix_at_open'] if features['vix_at_open'] is not None else 'NA'}"
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
