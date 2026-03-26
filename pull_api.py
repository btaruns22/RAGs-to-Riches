import os
from typing import Iterator

import boto3
import pandas as pd
from botocore.config import Config
from dotenv import load_dotenv


# Load Massive credentials from the local .env file.
load_dotenv()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def build_s3_client():
    access_key = require_env("MASSIVE_ACCESS_KEY")
    secret_key = require_env("MASSIVE_SECRET_KEY")
    endpoint = require_env("MASSIVE_S3_ENDPOINT")

    # Massive flat files are exposed through an S3-compatible endpoint.
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    return session.client(
        "s3",
        endpoint_url=endpoint,
        config=Config(signature_version="s3v4"),
    )


def build_daily_object_key(trade_date: str) -> str:
    # Flat files are organized by year/month/date under minute_aggs_v1.
    year, month, _ = trade_date.split("-")
    return f"us_stocks_sip/minute_aggs_v1/{year}/{month}/{trade_date}.csv.gz"


def read_daily_file(
    trade_date: str,
    chunksize: int = 100_000,
) -> Iterator[pd.DataFrame]:
    bucket = require_env("MASSIVE_S3_BUCKET")
    object_key = build_daily_object_key(trade_date)
    s3_client = build_s3_client()
    response = s3_client.get_object(Bucket=bucket, Key=object_key)

    # Stream the gzipped CSV in chunks so large daily files stay manageable.
    return pd.read_csv(
        response["Body"],
        compression="gzip",
        chunksize=chunksize,
    )


def load_symbol_for_day(
    trade_date: str,
    ticker: str = "SPY",
    chunksize: int = 100_000,
) -> pd.DataFrame:
    output_frames = []

    for chunk in read_daily_file(trade_date=trade_date, chunksize=chunksize):
        # Daily files contain many tickers, so narrow the dataset here.
        symbol_rows = chunk.loc[chunk["ticker"] == ticker].copy()
        if not symbol_rows.empty:
            output_frames.append(symbol_rows)

    if not output_frames:
        return pd.DataFrame()

    return pd.concat(output_frames, ignore_index=True)


if __name__ == "__main__":
    example_date = "2024-03-01"
    df = load_symbol_for_day(trade_date=example_date, ticker="SPY")

    print(f"Pulled {len(df)} rows for SPY on {example_date}.")
    if not df.empty:
        print(df.head())
