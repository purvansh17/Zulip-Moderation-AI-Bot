"""Layer 2: CSV ingestion data flow tests.

Validates that ingest_csv() uploads CSV chunks to MinIO at the
expected path with correct row counts.

Run: pytest tests/e2e/test_02_data_flow/test_ingestion.py -v -m data_flow
"""

import io
import logging
import os
import tempfile

import pandas as pd
import pytest

from src.data.ingest_and_expand import ingest_csv
from src.utils.config import config
from src.utils.minio_client import get_minio_client

logger = logging.getLogger(__name__)


@pytest.mark.data_flow
class TestIngestion:
    """Verify CSV ingestion to MinIO."""

    def test_ingest_csv_to_minio(
        self, docker_services, clean_state, test_dataset_small
    ):
        """CSV chunks uploaded to zulip-raw-messages/real/combined_dataset/chunk_NNN.csv."""
        # Write small dataset to temp CSV (simulate combined_dataset.csv)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            test_dataset_small.to_csv(f, index=False)
            temp_csv = f.name

        try:
            # Override chunk size to ensure multiple chunks from 1000 rows
            original_chunk_size = config.CHUNK_SIZE
            # Use 500 rows per chunk → expect 2 chunks from 1000 rows
            import src.data.ingest_and_expand as ingest_mod

            ingest_mod.CHUNK_SIZE = 500

            chunk_count = ingest_csv(csv_path=temp_csv)

            assert chunk_count >= 2, f"Expected >=2 chunks, got {chunk_count}"

            # Verify chunks exist in MinIO
            client = get_minio_client()
            objects = list(
                client.list_objects(
                    config.BUCKET_RAW,
                    prefix="real/combined_dataset/",
                    recursive=True,
                )
            )
            object_names = [obj.object_name for obj in objects]
            assert len(object_names) >= 2, (
                f"Expected >=2 objects at real/combined_dataset/, got {object_names}"
            )
            # Verify naming convention: chunk_NNN.csv
            for name in object_names:
                assert name.startswith("real/combined_dataset/chunk_"), (
                    f"Unexpected object name: {name}"
                )
                assert name.endswith(".csv"), f"Not a CSV: {name}"

            # Verify content: total rows across chunks = original
            total_rows = 0
            for obj in objects:
                response = client.get_object(config.BUCKET_RAW, obj.object_name)
                data = response.read()
                response.close()
                chunk_df = pd.read_csv(io.BytesIO(data))
                total_rows += len(chunk_df)

            assert total_rows == len(test_dataset_small), (
                f"Row count mismatch: {total_rows} vs {len(test_dataset_small)}"
            )
        finally:
            os.unlink(temp_csv)
            ingest_mod.CHUNK_SIZE = original_chunk_size
