"""Entry point for building the SPY opening-window dataset."""
import argparse

from project_config import DEFAULT_DATA_END_DATE, TRAIN_START_DATE
from pipeline.dataset import build_chunk_output_paths, build_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SPY setup datasets for a date range.")
    parser.add_argument("--start", default=TRAIN_START_DATE, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end", default=DEFAULT_DATA_END_DATE, help="End date in YYYY-MM-DD format")
    parser.add_argument("--output-dir", default="data/generated", help="Directory for generated CSVs")
    parser.add_argument("--features-path", default=None, help="Explicit output path for features CSV")
    parser.add_argument("--raw-path", default=None, help="Explicit output path for raw CSV")
    args = parser.parse_args()

    if args.features_path and args.raw_path:
        features_path = args.features_path
        raw_path = args.raw_path
    else:
        features_path, raw_path = build_chunk_output_paths(
            start_date=args.start,
            end_date=args.end,
            output_dir=args.output_dir,
        )

    build_dataset(
        start_date=args.start,
        end_date=args.end,
        features_path=features_path,
        raw_path=raw_path,
    )
