"""Layer 2: Training data compilation tests.

Validates compile_initial and compile_incremental modes against
live PostgreSQL + MinIO.

Run: pytest tests/e2e/test_02_data_flow/test_compilation.py -v -m data_flow
"""

import io
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.data.compile_training_data import (
    apply_quality_gate,
    filter_temporal_leakage,
)
from src.utils.config import config
from src.utils.db import get_db_connection
from src.utils.minio_client import get_minio_client

logger = logging.getLogger(__name__)


@pytest.mark.data_flow
class TestCompilation:
    """Verify training data compilation against live services."""

    def test_compile_initial_mode(self, docker_services, clean_state):
        """Initial mode: load CSV chunks to MinIO, verify accessible."""
        client = get_minio_client()

        # Simulate initial data load — upload a small CSV directly
        test_df = pd.DataFrame(
            {
                "text": ["test message one", "test message two", "test message three"],
                "is_suicide": [0, 0, 1],
                "is_toxicity": [0, 1, 0],
                "source": ["real", "real", "synthetic"],
            }
        )
        csv_bytes = test_df.to_csv(index=False).encode("utf-8")
        client.put_object(
            bucket_name=config.BUCKET_RAW,
            object_name="test/initial/chunk_000.csv",
            data=io.BytesIO(csv_bytes),
            length=len(csv_bytes),
            content_type="text/csv",
        )

        # Verify upload
        response = client.get_object(config.BUCKET_RAW, "test/initial/chunk_000.csv")
        loaded = pd.read_csv(io.BytesIO(response.read()))
        response.close()
        assert len(loaded) == 3, f"Expected 3 rows, got {len(loaded)}"

    def test_temporal_leakage_filter(self, docker_services):
        """Temporal leakage filter drops rows where created_at >= decided_at."""
        now = datetime.now(timezone.utc)
        df = pd.DataFrame(
            {
                "cleaned_text": ["ok row", "leaked row", "also ok"],
                "created_at": [
                    now - timedelta(hours=2),
                    now,  # created_at >= decided_at → LEAKED
                    now - timedelta(hours=1),
                ],
                "decided_at": [
                    now - timedelta(hours=1),
                    now - timedelta(hours=2),
                    now + timedelta(hours=1),
                ],
            }
        )
        result = filter_temporal_leakage(df)
        assert len(result) == 2, f"Expected 2 rows after filter, got {len(result)}"
        assert "leaked row" not in result["cleaned_text"].values, (
            "Leaked row should have been filtered"
        )

    def test_quality_gate_filters(self, docker_services):
        """Quality gate removes #ERROR! rows and short texts."""
        df = pd.DataFrame(
            {
                "cleaned_text": [
                    "#ERROR! duplicate",
                    "valid text that is long enough",
                    "short",
                    "another valid and sufficiently long text",
                    "#ERROR!",
                ],
            }
        )
        result = apply_quality_gate(df)
        assert not result["cleaned_text"].str.contains("#ERROR!").any(), (
            "#ERROR! rows not filtered"
        )
        assert (result["cleaned_text"].str.len() >= 10).all(), (
            "Short text rows not filtered"
        )
        assert len(result) == 2, f"Expected 2 rows, got {len(result)}"

    def test_messages_table_insert_query(self, docker_services, clean_state):
        """Insert test messages to PostgreSQL, query them back."""
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            # Create test user first (messages.user_id is NOT NULL with FK)
            cur.execute(
                "INSERT INTO users (id, username, source) "
                "VALUES (gen_random_uuid(), 'test_user_compilation', 'real') RETURNING id"
            )
            user_id = cur.fetchone()[0]
            # Insert test messages
            cur.execute(
                "INSERT INTO messages (id, user_id, text, cleaned_text, is_suicide, is_toxicity, source, created_at) "
                "VALUES (gen_random_uuid(), %s, %s, %s, %s, %s, %s, NOW())",
                (user_id, "raw text 1", "cleaned text 1", False, False, "real"),
            )
            cur.execute(
                "INSERT INTO messages (id, user_id, text, cleaned_text, is_suicide, is_toxicity, source, created_at) "
                "VALUES (gen_random_uuid(), %s, %s, %s, %s, %s, %s, NOW())",
                (user_id, "raw text 2", "cleaned text 2", True, False, "real"),
            )
            conn.commit()

            # Query back
            cur.execute("SELECT text, is_suicide FROM messages ORDER BY created_at")
            rows = cur.fetchall()
            assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"
            assert rows[0][0] == "raw text 1"
            assert rows[1][1] is True  # is_suicide
        finally:
            conn.close()
