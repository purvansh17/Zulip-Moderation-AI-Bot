"""Shared fixtures and helpers for E2E tests.

Provides:
- docker_services: session-scoped, starts docker-compose, waits for health
- clean_state: function-scoped, truncates tables + clears S3 objects
- test_dataset_small/medium: stratified samples from combined_dataset.csv
- kill_container: context manager for chaos injection
- corrupt_data: context manager for data corruption scenarios
"""

import logging
import subprocess
import time
from contextlib import contextmanager

import pytest

from src.utils.config import config
from src.utils.db import get_db_connection
from src.utils.minio_client import get_minio_client

logger = logging.getLogger(__name__)

COMPOSE_FILE = "docker/docker-compose.yaml"
HEALTH_CHECK_TIMEOUT = 60  # seconds
HEALTH_CHECK_INTERVAL = 2  # seconds

# Table names from docker/init_sql/00_create_tables.sql
TABLES_TO_TRUNCATE = ["moderation", "messages", "users", "flags"]


# ---- Docker lifecycle ----------------------------------------------------


def _wait_for_postgres(timeout: int = HEALTH_CHECK_TIMEOUT) -> bool:
    """Poll PostgreSQL until pg_isready succeeds or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(
            [
                "docker",
                "exec",
                "postgres",
                "pg_isready",
                "-h",
                "localhost",
                "-p",
                "5432",
                "-U",
                "user",
                "-d",
                "chatsentry",
            ],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            logger.info("PostgreSQL is ready")
            return True
        time.sleep(HEALTH_CHECK_INTERVAL)
    logger.error("PostgreSQL health check timed out after %ds", timeout)
    return False


def _wait_for_s3(timeout: int = HEALTH_CHECK_TIMEOUT) -> bool:
    """Poll S3 via MinIO client until connection succeeds or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            client = get_minio_client()
            client.list_buckets()
            logger.info("S3 is ready")
            return True
        except Exception:
            pass
        time.sleep(HEALTH_CHECK_INTERVAL)
    logger.error("S3 health check timed out after %ds", timeout)
    return False


@pytest.fixture(scope="session")
def docker_services():
    """Start docker-compose, wait for health checks, tear down after session.

    Skips startup if services are already running (checks pg_isready).
    """
    # Check if already running (via docker exec since pg_isready may not be on host)
    pg_already_running = (
        subprocess.run(
            [
                "docker",
                "exec",
                "postgres",
                "pg_isready",
                "-h",
                "localhost",
                "-p",
                "5432",
            ],
            capture_output=True,
        ).returncode
        == 0
    )

    _started_here = False
    if not pg_already_running:
        logger.info("Starting Docker services via docker-compose...")
        subprocess.run(
            ["docker", "compose", "-f", COMPOSE_FILE, "up", "-d"],
            check=True,
        )
        _started_here = True
    else:
        logger.info("Docker services already running, skipping startup")

    # Wait for health checks
    assert _wait_for_postgres(), "PostgreSQL did not become healthy"
    assert _wait_for_s3(), "S3 did not become healthy"

    yield

    # Teardown only if we started them
    if _started_here:
        logger.info("Tearing down Docker services...")
        subprocess.run(
            ["docker", "compose", "-f", COMPOSE_FILE, "down", "-v"],
            check=True,
        )


# ---- State cleanup -------------------------------------------------------


@pytest.fixture
def clean_state(docker_services):
    """Truncate PostgreSQL tables and clear S3 objects between tests."""
    # PostgreSQL cleanup
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        for table in TABLES_TO_TRUNCATE:
            cur.execute(f"TRUNCATE TABLE {table} CASCADE")
        conn.commit()
        cur.close()
        logger.debug("Truncated PostgreSQL tables")
    finally:
        conn.close()

    # S3 cleanup
    client = get_minio_client()
    for bucket in {config.BUCKET_RAW, config.BUCKET_TRAINING}:
        if client.bucket_exists(bucket):
            objects = list(client.list_objects(bucket, recursive=True))
            if objects:
                for obj in objects:
                    client.remove_object(bucket, obj.object_name)
                logger.debug("Cleared %d objects from %s", len(objects), bucket)

    yield


# ---- Test data fixtures --------------------------------------------------


@pytest.fixture
def test_dataset_small():
    """1,000-row stratified sample from combined_dataset.csv."""
    from tests.e2e.test_data import load_small_dataset

    return load_small_dataset()


@pytest.fixture
def test_dataset_medium():
    """10,000-row stratified sample from combined_dataset.csv."""
    from tests.e2e.test_data import load_medium_dataset

    return load_medium_dataset()


# ---- Chaos injection helpers ---------------------------------------------


@contextmanager
def kill_container(service_name: str):
    """Stop a Docker container, yield, then restart it.

    Usage:
        with kill_container("postgres"):
            # test behavior when postgres is down
    """
    logger.info("Chaos: stopping container %s", service_name)
    subprocess.run(
        ["docker", "stop", service_name],
        check=True,
        capture_output=True,
    )
    try:
        yield
    finally:
        logger.info("Chaos: restarting container %s", service_name)
        subprocess.run(
            ["docker", "start", service_name],
            check=True,
            capture_output=True,
        )
        # Wait for health after restart
        if service_name == "postgres":
            _wait_for_postgres()
        else:
            time.sleep(5)  # Generic wait for other services


@contextmanager
def corrupt_data(data_type: str):
    """Generate corrupted test data for pipeline resilience testing.

    Args:
        data_type: Type of corruption. One of:
            - "csv_encoding": Rows with mixed/broken encoding
            - "null_values": Rows with NULL text or labels
            - "duplicates": Rows with duplicate message IDs

    Yields:
        pd.DataFrame with corrupted data.
    """
    import pandas as pd

    if data_type == "csv_encoding":
        df = pd.DataFrame(
            {
                "text": [
                    "valid text here",
                    "\xff\xfe broken encoding",
                    "another valid text",
                    "\x80\x81 garbage",
                    "normal row",
                ],
                "is_suicide": [0, 0, 0, 0, 0],
                "is_toxicity": [0, 0, 0, 0, 0],
                "source": ["corrupt"] * 5,
            }
        )
    elif data_type == "null_values":
        import numpy as np

        df = pd.DataFrame(
            {
                "text": ["valid text", None, "another valid", np.nan, "normal"],
                "is_suicide": [0, 0, None, 0, 0],
                "is_toxicity": [0, 0, 0, None, 0],
                "source": ["corrupt"] * 5,
            }
        )
    elif data_type == "duplicates":
        df = pd.DataFrame(
            {
                "text": ["text one", "text one", "text two", "text two", "unique text"],
                "is_suicide": [0, 0, 1, 1, 0],
                "is_toxicity": [0, 0, 0, 0, 1],
                "source": ["corrupt"] * 5,
                "message_id": [1, 1, 2, 2, 3],
            }
        )
    else:
        raise ValueError(f"Unknown corruption type: {data_type}")

    yield df
