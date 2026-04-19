"""Layer 5: Full pipeline with 1,000-row dataset.

Runs complete pipeline: ingest → clean → compile → snapshot.
Verifies row counts match at each stage and final snapshot is correct.

Run: pytest tests/e2e/test_05_full_pipeline/test_full_pipeline_small.py -v -m full_pipeline
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


@pytest.mark.full_pipeline
class TestFullPipelineSmall:
    """Full pipeline end-to-end with 1,000 rows."""

    def test_full_pipeline_ingest_to_snapshot(
        self, docker_services, clean_state, test_dataset_small
    ):
        """Complete pipeline: ingest CSV → clean text → quality gate → split → upload."""
        df = test_dataset_small
        original_count = len(df)
        logger.info("Starting full pipeline with %d rows", original_count)

        # Stage 1: Ingest — write to temp CSV, upload to MinIO
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            df.to_csv(f, index=False)
            temp_csv = f.name

        try:
            import src.data.ingest_and_expand as ingest_mod

            original_chunk = ingest_mod.CHUNK_SIZE
            ingest_mod.CHUNK_SIZE = 500  # Force multiple chunks

            chunk_count = ingest_csv(csv_path=temp_csv)
            assert chunk_count >= 2, f"Expected >=2 chunks, got {chunk_count}"
            logger.info("Stage 1 (ingest): %d chunks uploaded", chunk_count)

            ingest_mod.CHUNK_SIZE = original_chunk
        finally:
            os.unlink(temp_csv)

        # Stage 2: Clean text
        cleaner = TextCleaner()
        if "text" in df.columns:
            df["cleaned_text"] = df["text"].astype(str).apply(cleaner.clean)
        elif "cleaned_text" not in df.columns:
            # If no text column, use index as placeholder
            df["cleaned_text"] = [f"cleaned text {i}" for i in range(len(df))]

        cleaned_count = len(df)
        logger.info("Stage 2 (clean): %d rows", cleaned_count)
        assert cleaned_count == original_count, (
            f"Row count changed during cleaning: {original_count} → {cleaned_count}"
        )

        # Stage 3: Quality gate
        df = apply_quality_gate(df)
        quality_count = len(df)
        logger.info("Stage 3 (quality gate): %d rows", quality_count)
        assert quality_count <= original_count, (
            f"Quality gate should not add rows: {original_count} → {quality_count}"
        )

        # Stage 4: Stratified split
        # Ensure required columns exist
        if "is_suicide" not in df.columns:
            df["is_suicide"] = 0
        if "is_toxicity" not in df.columns:
            df["is_toxicity"] = 0

        train_df, val_df, test_df = stratified_split(df)
        total_split = len(train_df) + len(val_df) + len(test_df)
        logger.info(
            "Stage 4 (split): train=%d, val=%d, test=%d",
            len(train_df),
            len(val_df),
            len(test_df),
        )
        assert total_split == quality_count, (
            f"Split total {total_split} != input {quality_count}"
        )

        # Stage 5: Upload snapshot to MinIO
        client = get_minio_client()
        version = upload_snapshot(
            client, config.BUCKET_TRAINING, train_df, val_df, test_df
        )
        logger.info("Stage 5 (snapshot): version=%s", version)
        assert version.startswith("v"), f"Expected versioned name, got: {version}"

        # Verify snapshot contents
        for split_name, expected_count in [
            ("train", len(train_df)),
            ("val", len(val_df)),
            ("test", len(test_df)),
        ]:
            response = client.get_object(
                config.BUCKET_TRAINING, f"{version}/{split_name}.csv"
            )
            split_df = pd.read_csv(io.BytesIO(response.read()))
            response.close()
            assert len(split_df) == expected_count, (
                f"{split_name}: expected {expected_count} rows, got {len(split_df)}"
            )
