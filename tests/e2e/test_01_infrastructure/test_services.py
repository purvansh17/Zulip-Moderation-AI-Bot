"""Layer 1: Infrastructure health tests.

Verifies all Docker services are running and healthy before
pipeline tests execute. These are gate checks — if infrastructure
is broken, no point running data flow or chaos tests.

Run: pytest tests/e2e/ -m infrastructure -v
"""

import json
import logging
import subprocess

import pytest

from src.utils.config import config
from src.utils.db import get_db_connection
from src.utils.minio_client import get_minio_client

logger = logging.getLogger(__name__)

# Tables defined in docker/init_sql/00_create_tables.sql
EXPECTED_TABLES = {"users", "messages", "moderation", "flags"}
EXPECTED_BUCKETS = {config.BUCKET_RAW, config.BUCKET_TRAINING}
# Containers from docker/docker-compose.yaml (no MinIO — using Chameleon S3)
EXPECTED_CONTAINERS = {"postgres", "adminer", "api", "ge-viewer"}


@pytest.mark.infrastructure
class TestPostgresHealth:
    """Verify PostgreSQL is reachable and schema is initialized."""

    def test_connection(self, docker_services):
        """PostgreSQL accepts connections via get_db_connection()."""
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        result = cur.fetchone()
        cur.close()
        conn.close()
        assert result == (1,), f"Expected (1,), got {result}"

    def test_tables_exist(self, docker_services):
        """All 4 expected tables exist in the chatsentry database."""
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public'"
        )
        tables = {row[0] for row in cur.fetchall()}
        cur.close()
        conn.close()
        missing = EXPECTED_TABLES - tables
        assert not missing, f"Missing tables: {missing}"


@pytest.mark.infrastructure
class TestS3Health:
    """Verify Chameleon S3 is reachable and buckets are accessible."""

    def test_connection(self, docker_services):
        """S3 client can connect and list buckets."""
        client = get_minio_client()
        buckets = [b.name for b in client.list_buckets()]
        assert len(buckets) >= 2, f"Expected >=2 buckets, got {buckets}"

    def test_buckets_exist(self, docker_services):
        """Both required buckets exist: zulip-raw-messages, zulip-training-data."""
        client = get_minio_client()
        existing = {b.name for b in client.list_buckets()}
        missing = EXPECTED_BUCKETS - existing
        assert not missing, f"Missing buckets: {missing}"


@pytest.mark.infrastructure
class TestApiHealth:
    """Verify FastAPI health endpoint responds."""

    def test_health_endpoint(self, docker_services):
        """GET /health returns 200 with healthy status."""
        import urllib.request

        url = "http://localhost:8000/health"
        req = urllib.request.urlopen(url, timeout=10)
        assert req.getcode() == 200, f"Expected 200, got {req.getcode()}"
        body = json.loads(req.read().decode())
        assert body.get("status") == "ok", f"Unexpected body: {body}"


@pytest.mark.infrastructure
class TestDockerComposeRunning:
    """Verify all expected containers are in running state."""

    def test_all_containers_running(self, docker_services):
        """All expected containers from docker-compose.yaml are running."""
        result = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                "docker/docker-compose.yaml",
                "ps",
                "--format",
                "json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        # docker compose ps --format json outputs one JSON object per line
        running_containers = set()
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                container = json.loads(line)
                if container.get("State") == "running":
                    running_containers.add(container.get("Name", ""))

        missing = EXPECTED_CONTAINERS - running_containers
        assert not missing, f"Containers not running: {missing}"
