"""Layer 2: Stratified split and versioned snapshot tests.

Validates 70/15/15 split proportions and MinIO snapshot upload
with correct versioned folder structure.

Run: pytest tests/e2e/test_02_data_flow/test_splitting.py -v -m data_flow
"""

import io
import logging

import pandas as pd
import pytest

from src.data.compile_training_data import stratified_split, upload_snapshot
from src.utils.config import config
from src.utils.minio_client import get_minio_client

logger = logging.getLogger(__name__)


@pytest.mark.data_flow
class TestSplitting:
    """Verify stratified split and versioned MinIO snapshots."""

    def test_stratified_split_proportions(self, docker_services):
        """Split produces 70/15/15 proportions within 5% tolerance."""
        df = pd.DataFrame(
            {
                "cleaned_text": [f"text {i}" for i in range(1000)],
                "is_suicide": [0] * 800 + [1] * 200,
                "is_toxicity": ([0] * 600 + [1] * 200) + ([0] * 150 + [1] * 50),
            }
        )
        train_df, val_df, test_df = stratified_split(df)

        total = len(train_df) + len(val_df) + len(test_df)
        train_pct = len(train_df) / total
        val_pct = len(val_df) / total
        test_pct = len(test_df) / total

        assert 0.65 <= train_pct <= 0.75, (
            f"Train split {train_pct:.1%} outside 65-75% range"
        )
        assert 0.10 <= val_pct <= 0.20, f"Val split {val_pct:.1%} outside 10-20% range"
        assert 0.10 <= test_pct <= 0.20, (
            f"Test split {test_pct:.1%} outside 10-20% range"
        )

    def test_stratified_split_no_label_combo_column(self, docker_services):
        """Internal label_combo column not present in output DataFrames."""
        df = pd.DataFrame(
            {
                "cleaned_text": [f"text {i}" for i in range(100)],
                "is_suicide": [0] * 80 + [1] * 20,
                "is_toxicity": [0] * 70 + [1] * 30,
            }
        )
        train_df, val_df, test_df = stratified_split(df)
        for split_name, split_df in [
            ("train", train_df),
            ("val", val_df),
            ("test", test_df),
        ]:
            assert "label_combo" not in split_df.columns, (
                f"Internal label_combo column leaked into {split_name}"
            )

    def test_versioned_snapshot_structure(self, docker_services, clean_state):
        """Snapshot uploaded to MinIO with train/val/test CSVs in v{timestamp}/."""
        client = get_minio_client()

        train_df = pd.DataFrame(
            {
                "cleaned_text": ["train row 1", "train row 2"],
                "is_suicide": [0, 1],
                "is_toxicity": [0, 0],
                "source": ["test", "test"],
                "message_id": [1, 2],
            }
        )
        val_df = pd.DataFrame(
            {
                "cleaned_text": ["val row 1"],
                "is_suicide": [0],
                "is_toxicity": [1],
                "source": ["test"],
                "message_id": [3],
            }
        )
        test_df = pd.DataFrame(
            {
                "cleaned_text": ["test row 1"],
                "is_suicide": [1],
                "is_toxicity": [0],
                "source": ["test"],
                "message_id": [4],
            }
        )

        version = upload_snapshot(
            client, config.BUCKET_TRAINING, train_df, val_df, test_df
        )
        assert version.startswith("v"), f"Version should start with 'v', got: {version}"

        # Verify all 3 splits exist
        for split_name in ("train", "val", "test"):
            object_name = f"{version}/{split_name}.csv"
            stat = client.stat_object(config.BUCKET_TRAINING, object_name)
            assert stat.size > 0, f"{object_name} is empty"

        # Verify content matches
        response = client.get_object(config.BUCKET_TRAINING, f"{version}/train.csv")
        loaded_train = pd.read_csv(io.BytesIO(response.read()))
        response.close()
        assert len(loaded_train) == 2, f"Expected 2 train rows, got {len(loaded_train)}"
