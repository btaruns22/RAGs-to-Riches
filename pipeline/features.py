"""SPY data loading and feature engineering for the 9:30-9:34 opening window."""
from datetime import datetime
from typing import Deque, List, Optional

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from project_config import (
    FEATURE_WINDOW_END_MINUTE,
    FEATURE_WINDOW_START_MINUTE,
    OUTCOME_WINDOW_END_MINUTE,
    OUTCOME_WINDOW_START_MINUTE,
)
from services.s3_client import list_available_keys, read_daily_file

ET = ZoneInfo("America/New_York")

_STOCKS_PREFIX = "us_stocks_sip/minute_aggs_v1"
_INDICES_PREFIX = "us_indices/minute_aggs_v1"


def list_trading_dates(start_date: str, end_date: str) -> List[str]:
    """Return sorted trading dates in [start_date, end_date] by scanning S3."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    dates = []
    year, month = start_dt.year, start_dt.month

    while (year, month) <= (end_dt.year, end_dt.month):
        prefix = f"{_STOCKS_PREFIX}/{year}/{month:02d}/"
        for key in list_available_keys(prefix=prefix, limit=31):
            date_str = key.split("/")[-1].replace(".csv.gz", "")
            if start_date <= date_str <= end_date:
                dates.append(date_str)
        month += 1
        if month > 12:
            year, month = year + 1, 1

    return sorted(dates)


def load_spy_day(trade_date: str) -> pd.DataFrame:
    """Load all SPY minute bars for a trading day with ET timestamps attached."""
    frames = []
    for chunk in read_daily_file(trade_date, dataset_prefix=_STOCKS_PREFIX):
        spy = chunk[chunk["ticker"] == "SPY"]
        if not spy.empty:
            frames.append(spy.copy())

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["window_start"], unit="ns", utc=True).dt.tz_convert(ET)
    df["minute"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    return df.sort_values("ts").reset_index(drop=True)


def load_vix_day(trade_date: str) -> pd.DataFrame:
    """Load all VIX minute bars for a trading day with ET timestamps attached."""
    frames = []
    for chunk in read_daily_file(trade_date, dataset_prefix=_INDICES_PREFIX):
        vix = chunk[chunk["ticker"] == "I:VIX"]
        if not vix.empty:
            frames.append(vix.copy())

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["window_start"], unit="ns", utc=True).dt.tz_convert(ET)
    df["minute"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    return df.sort_values("ts").reset_index(drop=True)


def extract_open_window(spy_day: pd.DataFrame) -> pd.DataFrame:
    """Return the 5 bars covering 9:30-9:34 ET."""
    mask = (spy_day["minute"] >= FEATURE_WINDOW_START_MINUTE) & (
        spy_day["minute"] <= FEATURE_WINDOW_END_MINUTE
    )
    return spy_day[mask].reset_index(drop=True)


def extract_outcome_window(spy_day: pd.DataFrame) -> pd.DataFrame:
    """Return the bars covering 9:35-10:30 ET."""
    mask = (spy_day["minute"] >= OUTCOME_WINDOW_START_MINUTE) & (
        spy_day["minute"] <= OUTCOME_WINDOW_END_MINUTE
    )
    return spy_day[mask].reset_index(drop=True)


def get_regular_close(spy_day: pd.DataFrame) -> Optional[float]:
    """Return SPY close price of the last regular-session bar on or before 4:00 PM."""
    regular = spy_day[spy_day["minute"] <= 16 * 60]
    if regular.empty:
        return None
    return float(regular.iloc[-1]["close"])


def get_vix_at_open(vix_day: pd.DataFrame) -> Optional[float]:
    """Return the VIX open price from the 09:30 minute bar."""
    if vix_day.empty:
        return None

    bar_930 = vix_day[vix_day["minute"] == FEATURE_WINDOW_START_MINUTE]
    if bar_930.empty:
        return None
    return float(bar_930.iloc[0]["open"])


def get_entry_price(window: pd.DataFrame) -> Optional[float]:
    """Return the 09:34 close, used as the simulated entry price."""
    bar_934 = window[window["minute"] == FEATURE_WINDOW_END_MINUTE]
    if bar_934.empty:
        return None
    return float(bar_934.iloc[0]["close"])


def compute_features(
    trade_date: str,
    window: pd.DataFrame,
    prev_close: float,
    vol_history: Deque,
    vix_at_open: Optional[float] = None,
) -> Optional[dict]:
    """Compute the agreed feature schema for one trading day's opening window."""
    bar_930 = window[window["minute"] == FEATURE_WINDOW_START_MINUTE]
    bar_934 = window[window["minute"] == FEATURE_WINDOW_END_MINUTE]

    if bar_930.empty or bar_934.empty:
        return None

    spy_open = float(bar_930.iloc[0]["open"])
    close_930 = float(bar_930.iloc[0]["close"])
    close_934 = float(bar_934.iloc[0]["close"])

    gap_pct = (spy_open - prev_close) / prev_close * 100
    first_1m_return = (close_930 - spy_open) / spy_open * 100
    net_movement = (close_934 - spy_open) / spy_open * 100

    opening_range_high = float(window["high"].max())
    opening_range_low = float(window["low"].min())

    if net_movement > 0:
        breakout_direction = "UP"
    elif net_movement < 0:
        breakout_direction = "DOWN"
    else:
        breakout_direction = "NONE"

    volatility = float((window["high"] - window["low"]).mean())
    volume = int(window["volume"].sum())
    avg_vol = float(np.mean(vol_history)) if vol_history else float(volume)
    rvol_10d = volume / avg_vol if avg_vol > 0 else 1.0

    return {
        "date": trade_date,
        "spy_open": spy_open,
        "previous_close": prev_close,
        "gap_pct": round(gap_pct, 4),
        "first_1m_return": round(first_1m_return, 4),
        "net_movement": round(net_movement, 4),
        "opening_range_high": opening_range_high,
        "opening_range_low": opening_range_low,
        "opening_range_width": round(opening_range_high - opening_range_low, 4),
        "breakout_direction": breakout_direction,
        "volatility": round(volatility, 4),
        "volume": volume,
        "rvol_10d": round(rvol_10d, 4),
        "vix_at_open": round(vix_at_open, 4) if vix_at_open is not None else None,
    }
