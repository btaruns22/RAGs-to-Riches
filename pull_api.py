import io
import os
from collections import deque
from datetime import datetime
from typing import Deque, Iterator, List, Optional
from strategy import label_trade

import boto3
import numpy as np
import pandas as pd
import requests
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from zoneinfo import ZoneInfo

load_dotenv()

ET = ZoneInfo("America/New_York")

# 9:30–9:34 inclusive (5 one-minute bars)
_OPEN_START = 9 * 60 + 30   # 570
_OPEN_END   = 9 * 60 + 34   # 574


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def build_s3_client():
    session = boto3.Session(
        aws_access_key_id=require_env("MASSIVE_ACCESS_KEY"),
        aws_secret_access_key=require_env("MASSIVE_SECRET_KEY"),
    )
    return session.client(
        "s3",
        endpoint_url=require_env("MASSIVE_S3_ENDPOINT"),
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def build_daily_object_key(trade_date: str) -> str:
    year, month, _ = trade_date.split("-")
    return f"us_stocks_sip/minute_aggs_v1/{year}/{month}/{trade_date}.csv.gz"


def list_available_keys(prefix: str, limit: int = 31) -> List[str]:
    s3_client = build_s3_client()
    bucket = require_env("MASSIVE_S3_BUCKET")
    paginator = s3_client.get_paginator("list_objects_v2")

    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
            if len(keys) >= limit:
                return keys
    return keys


def read_daily_file(
    trade_date: str,
    chunksize: int = 100_000,
) -> Iterator[pd.DataFrame]:
    bucket = require_env("MASSIVE_S3_BUCKET")
    object_key = build_daily_object_key(trade_date)
    s3_client = build_s3_client()

    try:
        presigned_url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": object_key},
            ExpiresIn=300,
        )
        http_response = requests.get(presigned_url, stream=True, timeout=60)
        http_response.raise_for_status()
    except (ClientError, requests.HTTPError) as exc:
        raise RuntimeError(
            f"Failed to read {object_key}: {exc}"
        ) from exc

    return pd.read_csv(
        io.BytesIO(http_response.content),
        compression="gzip",
        chunksize=chunksize,
    )


def list_trading_dates(start_date: str, end_date: str) -> List[str]:
    """Return sorted trading dates in [start_date, end_date] by scanning S3."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt   = datetime.strptime(end_date,   "%Y-%m-%d")

    dates = []
    year, month = start_dt.year, start_dt.month

    while (year, month) <= (end_dt.year, end_dt.month):
        prefix = f"us_stocks_sip/minute_aggs_v1/{year}/{month:02d}/"
        for key in list_available_keys(prefix=prefix, limit=31):
            date_str = key.split("/")[-1].replace(".csv.gz", "")
            if start_date <= date_str <= end_date:
                dates.append(date_str)
        month += 1
        if month > 12:
            year, month = year + 1, 1

    return sorted(dates)


def load_spy_day(trade_date: str) -> pd.DataFrame:
    """Load all SPY minute bars for a trading day with ET timestamps."""
    frames = []
    for chunk in read_daily_file(trade_date):
        spy = chunk[chunk["ticker"] == "SPY"]
        if not spy.empty:
            frames.append(spy.copy())

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["window_start"], unit="ns", utc=True).dt.tz_convert(ET)
    df["minute"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    return df.sort_values("ts").reset_index(drop=True)


def extract_open_window(spy_day: pd.DataFrame) -> pd.DataFrame:
    """Return the 9:30–9:34 bars."""
    mask = (spy_day["minute"] >= _OPEN_START) & (spy_day["minute"] <= _OPEN_END)
    return spy_day[mask].reset_index(drop=True)


def get_regular_close(spy_day: pd.DataFrame) -> Optional[float]:
    """Return SPY close price of the last regular-session bar (<= 4:00 PM)."""
    regular = spy_day[spy_day["minute"] <= 16 * 60]
    if regular.empty:
        return None
    return float(regular.iloc[-1]["close"])




def compute_features(
    trade_date: str,
    window: pd.DataFrame,
    prev_close: float,
    vol_history: Deque,
) -> Optional[dict]:
    bar_930 = window[window["minute"] == _OPEN_START]
    bar_934 = window[window["minute"] == _OPEN_END]

    if bar_930.empty or bar_934.empty:
        return None

    spy_open  = float(bar_930.iloc[0]["open"])
    close_930 = float(bar_930.iloc[0]["close"])
    close_934 = float(bar_934.iloc[0]["close"])

    gap_pct         = (spy_open - prev_close) / prev_close * 100
    first_1m_return = (close_930 - spy_open) / spy_open * 100
    net_movement    = (close_934 - spy_open) / spy_open * 100

    or_high  = float(window["high"].max())
    or_low   = float(window["low"].min())

    if net_movement > 0:
        direction = "UP"
    elif net_movement < 0:
        direction = "DOWN"
    else:
        direction = "NONE"

    volatility = float((window["high"] - window["low"]).mean())
    volume     = int(window["volume"].sum())
    avg_vol    = float(np.mean(vol_history)) if vol_history else float(volume)
    vol_ratio  = volume / avg_vol if avg_vol > 0 else 1.0

    return {
        "date":                trade_date,
        "spy_open":            spy_open,
        "previous_close":      prev_close,
        "gap_pct":             round(gap_pct, 4),
        "first_1m_return":     round(first_1m_return, 4),
        "net_movement":        round(net_movement, 4),
        "opening_range_high":  or_high,
        "opening_range_low":   or_low,
        "opening_range_width": round(or_high - or_low, 4),
        "breakout_direction":  direction,
        "volatility":          round(volatility, 4),
        "volume":              volume,
        "volume_ratio":        round(vol_ratio, 4),
    }


def label_row(row: dict) -> str:
    # TODO: replace with finalized strategy rules from the RAG & Evaluation Lead.
    # Ground truth is the close of the 5-minute candle (9:34 close = net_movement).
    # TAKE requires a clear directional move with above-average volume by end of window.
    if row["breakout_direction"] == "NONE":
        return "PASS"
    if row["volume_ratio"] < 1.2:
        return "PASS"
    if abs(row["net_movement"]) < 0.2:
        return "PASS"
    return "TAKE"


def build_dataset(
    start_date: str = "2023-03-01",
    end_date: str   = "2025-03-14",
    features_path: str = "spy_open_features.csv",
    raw_path: str      = "spy_open_raw_minutes.csv",
) -> pd.DataFrame:
    print(f"Listing trading dates {start_date} to {end_date}...")
    dates = list_trading_dates(start_date, end_date)
    print(f"Found {len(dates)} trading days.\n")

    feature_rows: list = []
    raw_rows: list     = []
    prev_close: Optional[float] = None
    vol_history: Deque = deque(maxlen=20)

    for i, trade_date in enumerate(dates, 1):
        print(f"[{i:3d}/{len(dates)}] {trade_date}", end="  ")
        try:
            spy_day = load_spy_day(trade_date)

            if spy_day.empty:
                print("no SPY data — skip")
                continue

            day_close = get_regular_close(spy_day)

            if prev_close is None:
                prev_close = day_close
                print("no prev close yet — skip")
                continue

            window = extract_open_window(spy_day)
            if len(window) < 5:
                print(f"incomplete window ({len(window)} bars) — skip")
                prev_close = day_close
                continue

            # Collect raw minute bars
            for _, bar in window.iterrows():
                raw_rows.append({
                    "date":   trade_date,
                    "time":   bar["ts"].strftime("%H:%M"),
                    "open":   bar["open"],
                    "high":   bar["high"],
                    "low":    bar["low"],
                    "close":  bar["close"],
                    "volume": int(bar["volume"]),
                })

            features = compute_features(trade_date, window, prev_close, vol_history)
            if features:
                features["label"] = label_trade(features)
                feature_rows.append(features)
                vol_history.append(features["volume"])
                print(f"OK  gap={features['gap_pct']:+.2f}%  dir={features['breakout_direction']}  net={features['net_movement']:+.2f}%")

            prev_close = day_close

        except RuntimeError as exc:
            print(f"ERROR — {exc}")

    features_df = pd.DataFrame(feature_rows)
    features_df.to_csv(features_path, index=False)
    print(f"\nSaved {len(features_df)} rows to {features_path}")

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(raw_path, index=False)
    print(f"Saved {len(raw_df)} rows to {raw_path}")

    return features_df


if __name__ == "__main__":
    build_dataset()
