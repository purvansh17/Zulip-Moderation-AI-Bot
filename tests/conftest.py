
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def api_client():
    return TestClient(app)


@pytest.fixture
def minio_client():
    from src.utils.minio_client import get_minio_client

    return get_minio_client()


@pytest.fixture
def pg_conn():
    from src.utils.db import get_db_connection

    conn = get_db_connection()
    yield conn
    conn.close()
