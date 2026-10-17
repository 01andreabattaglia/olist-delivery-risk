"""Upload the raw Kaggle CSVs to s3://olist-raw/olist/ingest_date=YYYY-MM-DD/."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from olist_delivery_risk.storage import get_s3_client, list_keys, upload_file

RAW_DIR = Path("data/raw")
BUCKET = "olist-raw"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().isoformat(), help="ingest date (YYYY-MM-DD)")
    parser.add_argument("--source", type=Path, default=RAW_DIR, help="folder with the CSVs")
    args = parser.parse_args()

    client = get_s3_client()
    prefix = f"olist/ingest_date={args.date}/"
    csvs = sorted(args.source.rglob("*.csv"))
    if not csvs:
        raise SystemExit(f"No CSV found under {args.source}")

    for csv in csvs:
        upload_file(csv, BUCKET, prefix + csv.name, client=client)
        print(f"uploaded: {prefix}{csv.name}")

    keys = list_keys(BUCKET, prefix, client=client)
    print(f"\n{len(keys)} keys in s3://{BUCKET}/{prefix}")
    for key in keys:
        print(f"  {key}")


if __name__ == "__main__":
    main()
