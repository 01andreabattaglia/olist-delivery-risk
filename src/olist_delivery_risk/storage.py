"""S3 helpers for the local SeaweedFS store (and real AWS)."""

from __future__ import annotations

import os
from pathlib import Path

import boto3
from botocore.client import BaseClient, Config
from dotenv import load_dotenv


def get_s3_client() -> BaseClient:
    """Build an S3 client from environment variables.

    ``S3_ENDPOINT_URL`` empty or missing -> real AWS. When it points to a
    custom endpoint (local SeaweedFS), path-style addressing is required.
    """
    load_dotenv()
    endpoint_url = os.getenv("S3_ENDPOINT_URL", "").strip()
    config = Config(s3={"addressing_style": "path"}) if endpoint_url else None
    return boto3.client("s3", endpoint_url=endpoint_url or None, config=config)


def upload_file(
    local_path: str | Path,
    bucket: str,
    key: str,
    client: BaseClient | None = None,
) -> None:
    """Upload a local file to ``s3://bucket/key``."""
    client = client or get_s3_client()
    client.upload_file(str(local_path), bucket, key)


def list_keys(bucket: str, prefix: str = "", client: BaseClient | None = None) -> list[str]:
    """List every key in ``bucket`` under ``prefix``, following pagination."""
    client = client or get_s3_client()
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    return keys
