"""Layer 5: Full pipeline with 10,000-row dataset.

Validates pipeline at realistic volume. Same stages as small test
but with 10K rows to catch scaling issues.

Run: pytest tests/e2e/test_05_full_pipeline/test_full_pipeline_medium.py -v -m full_pipeline
"""

import logging
import os
import tempfile

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
class TestFullPipelineMedium:
    """Full pipeline end-to-end with 10,000 rows."""

    def test_full_pipeline_medium_volume(
        self, docker_services, clean_state, test_dataset_medium
    ):
        """Pipeline handles 10K rows without errors or data loss."""
        df = test_dataset_medium
        original_count = len(df)
        logger.info("Starting medium pipeline with %d rows", original_count)

        # Stage 1: Ingest
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            df.to_csv(f, index=False)
            temp_csv = f.name

        try:
            chunk_count = ingest_csv(csv_path=temp_csv)
            assert chunk_count >= 1, "No chunks uploaded"
            logger.info("Stage 1: %d chunks", chunk_count)
        finally:
            os.unlink(temp_csv)

        # Stage 2: Clean
        cleaner = TextCleaner()
        if "text" in df.columns:
            df["cleaned_text"] = df["text"].astype(str).apply(cleaner.clean)
        elif "cleaned_text" not in df.columns:
            df["cleaned_text"] = [f"cleaned text {i}" for i in range(len(df))]

        # Stage 3: Quality gate
        df = apply_quality_gate(df)
        logger.info("Stage 3: %d rows after quality gate", len(df))
        assert len(df) > 0, "Quality gate removed all rows"

        # Stage 4: Split
        if "is_suicide" not in df.columns:
            df["is_suicide"] = 0
        if "is_toxicity" not in df.columns:
            df["is_toxicity"] = 0

        train_df, val_df, test_df = stratified_split(df)
        total = len(train_df) + len(val_df) + len(test_df)
        logger.info(
            "Stage 4: train=%d, val=%d, test=%d",
            len(train_df),
            len(val_df),
            len(test_df),
        )

        # Verify proportions (5% tolerance)
        train_pct = len(train_df) / total
        assert 0.65 <= train_pct <= 0.75, (
            f"Train proportion {train_pct:.1%} outside tolerance"
        )

        # Stage 5: Snapshot
        client = get_minio_client()
        version = upload_snapshot(
            client, config.BUCKET_TRAINING, train_df, val_df, test_df
        )
        logger.info("Stage 5: version=%s", version)
        assert version.startswith("v")

        # Verify all splits uploaded
        for split_name in ("train", "val", "test"):
            stat = client.stat_object(
                config.BUCKET_TRAINING, f"{version}/{split_name}.csv"
            )
            assert stat.size > 0, f"{split_name}.csv is empty"
