"""Merge chunked raw/features CSV outputs into final combined files."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load_csvs(paths: list[Path]) -> list[pd.DataFrame]:
    frames = []
    for path in paths:
        if path.is_file():
            frames.append(pd.read_csv(path))
    return frames


def merge_chunk_files(
    input_dir: str = "data/generated",
    features_pattern: str = "*_spy_open_setup_features.csv",
    raw_pattern: str = "*_spy_open_setup_raw.csv",
    features_output: str = "data/generated/spy_open_setup_features.csv",
    raw_output: str = "data/generated/spy_open_setup_raw.csv",
) -> None:
    """Merge chunk CSVs created by date-range runs into combined raw/features files."""
    input_root = Path(input_dir)

    feature_paths = sorted(input_root.glob(features_pattern))
    raw_paths = sorted(input_root.glob(raw_pattern))

    features_frames = _load_csvs(feature_paths)
    raw_frames = _load_csvs(raw_paths)

    if not features_frames:
        raise ValueError(f"No feature chunk files matched {features_pattern!r} in {input_dir}")
    if not raw_frames:
        raise ValueError(f"No raw chunk files matched {raw_pattern!r} in {input_dir}")

    features_df = (
        pd.concat(features_frames, ignore_index=True)
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    raw_df = (
        pd.concat(raw_frames, ignore_index=True)
        .drop_duplicates(subset=["date", "time"], keep="last")
        .sort_values(["date", "time"])
        .reset_index(drop=True)
    )

    Path(features_output).parent.mkdir(parents=True, exist_ok=True)
    Path(raw_output).parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(features_output, index=False)
    raw_df.to_csv(raw_output, index=False)

    print(f"Merged {len(feature_paths)} feature chunks into {features_output}")
    print(f"Merged {len(raw_paths)} raw chunks into {raw_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge chunked SPY setup CSV outputs.")
    parser.add_argument("--input-dir", default="data/generated")
    parser.add_argument("--features-pattern", default="*_spy_open_setup_features.csv")
    parser.add_argument("--raw-pattern", default="*_spy_open_setup_raw.csv")
    parser.add_argument("--features-output", default="data/generated/spy_open_setup_features.csv")
    parser.add_argument("--raw-output", default="data/generated/spy_open_setup_raw.csv")
    args = parser.parse_args()

    merge_chunk_files(
        input_dir=args.input_dir,
        features_pattern=args.features_pattern,
        raw_pattern=args.raw_pattern,
        features_output=args.features_output,
        raw_output=args.raw_output,
    )


if __name__ == "__main__":
    main()
