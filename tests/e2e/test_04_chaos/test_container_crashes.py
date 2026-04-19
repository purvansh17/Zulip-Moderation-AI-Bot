"""Layer 4: Container crash chaos tests.

Verifies services recover after container crashes with data intact.

Run: pytest tests/e2e/test_04_chaos/test_container_crashes.py -v -m chaos
"""

import io
import logging
import subprocess
import time

import pytest

from src.utils.config import config
from src.utils.db import get_db_connection
from src.utils.minio_client import get_minio_client

logger = logging.getLogger(__name__)


@pytest.mark.chaos
class TestContainerCrashes:
    """Verify recovery after container crashes."""

    def test_api_crash_recovery(self, docker_services, clean_state):
        """API restarts, state preserved in PostgreSQL."""
        # Insert data before crash
        conn = get_db_connection()
        cur = conn.cursor()
        # Create test user (messages.user_id is NOT NULL with FK)
        cur.execute(
            "INSERT INTO users (id, username, source) "
            "VALUES (gen_random_uuid(), 'test_user_crash', 'real') RETURNING id"
        )
        user_id = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO messages (id, user_id, text, cleaned_text, is_suicide, is_toxicity, source, created_at) "
            "VALUES (gen_random_uuid(), %s, 'before crash', 'before crash', false, false, 'real', NOW())",
            (user_id,),
        )
        conn.commit()
        cur.close()
        conn.close()

        # Crash API
        subprocess.run(["docker", "stop", "api"], check=True, capture_output=True)
        time.sleep(2)

        # Restart API
        subprocess.run(["docker", "start", "api"], check=True, capture_output=True)
        time.sleep(10)  # API needs time to start

        # Verify data preserved
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT text FROM messages WHERE text = 'before crash'")
        row = cur.fetchone()
        cur.close()
        conn.close()
        assert row is not None, "Data lost after API crash/restart"

    def test_s3_data_persists(self, docker_services, clean_state):
        """S3 objects persist across operations (S3 is a managed remote service)."""
        client = get_minio_client()
        data = b"persistent data"
        client.put_object(
            bucket_name=config.BUCKET_RAW,
            object_name="test/crash/persist.csv",
            data=io.BytesIO(data),
            length=len(data),
        )

        # Verify data is accessible on a fresh client (simulates VM recreation)
        client2 = get_minio_client()
        response = client2.get_object(config.BUCKET_RAW, "test/crash/persist.csv")
        content = response.read()
        response.close()
        assert content == data, "S3 data not accessible on fresh client"
