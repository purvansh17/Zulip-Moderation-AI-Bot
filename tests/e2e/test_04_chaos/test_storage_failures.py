"""Layer 4: Storage failure chaos tests.

Verifies pipeline handles S3 connectivity issues gracefully.
S3 is a remote managed service — we test client-side resilience.

Run: pytest tests/e2e/test_04_chaos/test_storage_failures.py -v -m chaos
"""

import io
import logging
import time

import pytest

from src.utils.config import config
from src.utils.minio_client import get_minio_client

logger = logging.getLogger(__name__)


@pytest.mark.chaos
class TestStorageFailures:
    """Verify graceful handling of S3 connectivity issues."""

    def test_s3_upload_and_retrieve(self, docker_services, clean_state):
        """Upload succeeds and data is retrievable from S3."""
        client = get_minio_client()
        data = b"test data after upload"
        client.put_object(
            bucket_name=config.BUCKET_RAW,
            object_name="test/chaos/upload_test.csv",
            data=io.BytesIO(data),
            length=len(data),
            content_type="text/csv",
        )

        # Verify
        response = client.get_object(config.BUCKET_RAW, "test/chaos/upload_test.csv")
        content = response.read()
        response.close()
        assert content == data, "Uploaded data does not match"

    def test_s3_connection_with_bad_endpoint(self):
        """Client with bad endpoint raises exception."""
        from minio import Minio

        bad_client = Minio(
            endpoint="nonexistent.invalid:9999",
            access_key="fake",
            secret_key="fake",
            secure=False,
        )
        with pytest.raises(Exception):
            bad_client.bucket_exists("any-bucket")

    def test_s3_data_available_across_clients(self, docker_services, clean_state):
        """Data written by one client is readable by another."""
        # Write with first client
        client1 = get_minio_client()
        data = b"cross-client test"
        client1.put_object(
            bucket_name=config.BUCKET_RAW,
            object_name="test/chaos/cross_client.csv",
            data=io.BytesIO(data),
            length=len(data),
        )

        # Read with second client (simulates fresh connection)
        time.sleep(1)
        client2 = get_minio_client()
        response = client2.get_object(config.BUCKET_RAW, "test/chaos/cross_client.csv")
        content = response.read()
        response.close()
        assert content == data, "Data not accessible across clients"
