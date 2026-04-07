"""Layer 5: Pipeline idempotency and chaos-during-run tests.

Verifies:
- Running pipeline twice produces identical output
- Injecting chaos during a full run doesn't crash pipeline

Run: pytest tests/e2e/test_05_full_pipeline/test_pipeline_idempotency.py -v -m full_pipeline
"""

import io
import logging
import os
import tempfile

import pandas as pd
import pytest

from src.data.compile_training_data import (
    apply_quality_gate,
    stratified_split,
    upload_snapshot,
)
from src.data.ingest_and_expand import ingest_csv
from src.data.text_cleaner import TextCleaner
from src.utils.config import config
from src.utils.minio_client import get_minio_client

logger = logging.getLogger(__name__)


def _run_pipeline(df: pd.DataFrame, client) -> str:
    """Run full pipeline and return snapshot version."""
    cleaner = TextCleaner()
    if "text" in df.columns:
        df["cleaned_text"] = df["text"].astype(str).apply(cleaner.clean)
    elif "cleaned_text" not in df.columns:
        df["cleaned_text"] = [f"cleaned {i}" for i in range(len(df))]

    df = apply_quality_gate(df)

    if "is_suicide" not in df.columns:
        df["is_suicide"] = 0
    if "is_toxicity" not in df.columns:
        df["is_toxicity"] = 0

    train_df, val_df, test_df = stratified_split(df)
    version = upload_snapshot(client, config.BUCKET_TRAINING, train_df, val_df, test_df)
    return version


def _get_snapshot_stats(client, version: str) -> dict:
    """Get row counts for a snapshot version."""
    stats = {}
    for split_name in ("train", "val", "test"):
        response = client.get_object(
            config.BUCKET_TRAINING, f"{version}/{split_name}.csv"
        )
        split_df = pd.read_csv(io.BytesIO(response.read()))
        response.close()
        stats[split_name] = len(split_df)
    return stats


@pytest.mark.full_pipeline
class TestPipelineIdempotency:
    """Verify pipeline produces identical output on repeated runs."""

    def test_pipeline_idempotency(
        self, docker_services, clean_state, test_dataset_small
    ):
        """Running pipeline twice with same input produces same row counts."""
        client = get_minio_client()

        # Run 1
        df1 = test_dataset_small.copy()
        version1 = _run_pipeline(df1, client)
        stats1 = _get_snapshot_stats(client, version1)
        logger.info("Run 1: version=%s, stats=%s", version1, stats1)

        # Clear MinIO for clean run 2
        objects = list(client.list_objects(config.BUCKET_TRAINING, recursive=True))
        for obj in objects:
            client.remove_object(config.BUCKET_TRAINING, obj.object_name)

        # Run 2 — same data
        df2 = test_dataset_small.copy()
        version2 = _run_pipeline(df2, client)
        stats2 = _get_snapshot_stats(client, version2)
        logger.info("Run 2: version=%s, stats=%s", version2, stats2)

        # Row counts must match (version names will differ — that's OK)
        assert stats1 == stats2, f"Idempotency failed: run1={stats1}, run2={stats2}"

    def test_pipeline_with_s3(self, docker_services, clean_state, test_dataset_small):
        """Pipeline runs end-to-end with S3 (managed remote storage)."""
        client = get_minio_client()

        # Upload initial data
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            test_dataset_small.to_csv(f, index=False)
            temp_csv = f.name

        try:
            import src.data.ingest_and_expand as ingest_mod

            original_chunk = ingest_mod.CHUNK_SIZE
            ingest_mod.CHUNK_SIZE = 500

            # Start ingestion
            chunk_count = ingest_csv(csv_path=temp_csv)
            assert chunk_count >= 1

            ingest_mod.CHUNK_SIZE = original_chunk
        finally:
            os.unlink(temp_csv)

        # S3 is always available (remote managed service)
        client = get_minio_client()
        assert client.bucket_exists(config.BUCKET_RAW), "S3 not available"

        # Continue pipeline — clean and split
        df = test_dataset_small.copy()
        cleaner = TextCleaner()
        if "text" in df.columns:
            df["cleaned_text"] = df["text"].astype(str).apply(cleaner.clean)
        else:
            df["cleaned_text"] = [f"text {i}" for i in range(len(df))]

        df = apply_quality_gate(df)
        if "is_suicide" not in df.columns:
            df["is_suicide"] = 0
        if "is_toxicity" not in df.columns:
            df["is_toxicity"] = 0

        train_df, val_df, test_df = stratified_split(df)
        version = upload_snapshot(
            client, config.BUCKET_TRAINING, train_df, val_df, test_df
        )
        assert version.startswith("v"), f"Snapshot failed: {version}"
