"""Create the project's S3 buckets. Idempotent: safe to run many times."""

from __future__ import annotations

from botocore.client import BaseClient
from botocore.exceptions import ClientError

from src.olist_delivery_risk.storage import get_s3_client

BUCKETS = ["olist-raw", "olist-processed"]


def ensure_bucket(client: BaseClient, bucket: str) -> bool:
    """Create ``bucket`` if missing. Returns True if it was created."""
    try:
        client.create_bucket(Bucket=bucket)
        return True
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
            return False
        raise


def main() -> None:
    client = get_s3_client()
    for bucket in BUCKETS:
        created = ensure_bucket(client, bucket)
        print(f"{'created' if created else 'exists '}: {bucket}")


if __name__ == "__main__":
    main()
