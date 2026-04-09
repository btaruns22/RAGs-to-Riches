"""S3 client and raw flat-file access for Massive's S3-compatible endpoint."""
import io
import os
import time
from typing import Iterator, List

import boto3
import pandas as pd
import requests
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

CONNECT_TIMEOUT_SECONDS = 15
READ_TIMEOUT_SECONDS = 180
MAX_DOWNLOAD_RETRIES = 3
RETRY_BACKOFF_SECONDS = 3


def require_env(name: str) -> str:
    """Return the value of a required environment variable or raise ValueError."""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def build_s3_client():
    """Create a boto3 S3 client configured for Massive's S3-compatible endpoint."""
    session = boto3.Session(
        aws_access_key_id=require_env("MASSIVE_ACCESS_KEY"),
        aws_secret_access_key=require_env("MASSIVE_SECRET_KEY"),
    )
    return session.client(
        "s3",
        endpoint_url=require_env("MASSIVE_S3_ENDPOINT"),
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def build_daily_object_key(
    trade_date: str,
    dataset_prefix: str = "us_stocks_sip/minute_aggs_v1",
) -> str:
    """Return the S3 key for a given trade date's minute aggregate flat file."""
    year, month, _ = trade_date.split("-")
    return f"{dataset_prefix}/{year}/{month}/{trade_date}.csv.gz"


def list_available_keys(prefix: str, limit: int = 31) -> List[str]:
    """List S3 object keys under the given prefix, up to limit results."""
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
    dataset_prefix: str = "us_stocks_sip/minute_aggs_v1",
    chunksize: int = 100_000,
) -> Iterator[pd.DataFrame]:
    """Download and stream a daily minute-aggregate flat file as chunked DataFrames."""
    bucket = require_env("MASSIVE_S3_BUCKET")
    object_key = build_daily_object_key(trade_date, dataset_prefix=dataset_prefix)
    s3_client = build_s3_client()

    try:
        presigned_url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": object_key},
            ExpiresIn=300,
        )
    except ClientError as exc:
        raise RuntimeError(f"Failed to sign {object_key}: {exc}") from exc

    last_exc: Exception | None = None
    content: bytes | None = None
    for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
        try:
            http_response = requests.get(
                presigned_url,
                timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
            )
            http_response.raise_for_status()
            content = http_response.content
            break
        except requests.RequestException as exc:
            last_exc = exc
            if attempt == MAX_DOWNLOAD_RETRIES:
                break
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    if content is None:
        raise RuntimeError(
            f"Failed to read {object_key} after {MAX_DOWNLOAD_RETRIES} attempts: {last_exc}"
        ) from last_exc

    return pd.read_csv(
        io.BytesIO(content),
        compression="gzip",
        chunksize=chunksize,
    )
