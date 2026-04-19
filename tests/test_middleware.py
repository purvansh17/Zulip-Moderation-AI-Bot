"""Integration tests for the TextCleaningMiddleware.

Tests that the middleware intercepts POST /messages and POST /flags,
cleans text fields, persists to PostgreSQL, and returns raw + cleaned text.

Maps to Plan 02-03 tasks 2 and 3.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_persistence():
    """Mock PostgreSQL and MinIO to avoid requiring running services."""
    with (
        patch("src.api.main.get_db_connection") as mock_db,
        patch("src.api.main.get_minio_client") as mock_minio,
    ):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        # Return a fake user UUID for user lookup queries
        mock_cursor.fetchone.return_value = ("00000000-0000-0000-0000-000000000001",)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_db.return_value = mock_conn

        mock_minio.return_value = MagicMock()

        yield {
            "db": mock_db,
            "conn": mock_conn,
            "cursor": mock_cursor,
            "minio": mock_minio,
        }


# ---------------------------------------------------------------------------
# Test 1: POST /messages returns raw_text and cleaned_text
# ---------------------------------------------------------------------------


def test_post_messages_returns_raw_and_cleaned(api_client):
    """Middleware returns both raw_text and cleaned_text in the response."""
    payload = {"text": "**bold** text", "user_id": "user-1", "source": "real"}
    response = api_client.post("/messages", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "raw_text" in data
    assert "cleaned_text" in data
    assert data["raw_text"] == "**bold** text"
    assert "bold" in data["cleaned_text"]


# ---------------------------------------------------------------------------
# Test 2: POST /messages text is persisted to PostgreSQL with cleaned_text
# ---------------------------------------------------------------------------


def test_post_messages_persists_to_db(api_client, mock_persistence):
    """Middleware persists message with cleaned_text to PostgreSQL."""
    payload = {"text": "hello world", "user_id": "user-1", "source": "real"}
    response = api_client.post("/messages", json=payload)
    assert response.status_code == 200

    # Verify INSERT was called with cleaned_text
    cursor = mock_persistence["cursor"]
    assert cursor.execute.called
    call_args = cursor.execute.call_args
    sql = call_args[0][0]
    params = call_args[0][1]
    assert "INSERT INTO messages" in sql
    assert "cleaned_text" in sql
    assert len(params) == 5  # id, user_id, raw_text, cleaned_text, source


# ---------------------------------------------------------------------------
# Test 3: POST /flags cleans reason field
# ---------------------------------------------------------------------------


def test_post_flags_cleans_reason(api_client):
    """Middleware cleans the reason field on POST /flags."""
    payload = {
        "message_id": "msg-1",
        "flagged_by": "user-1",
        "reason": "email me at test@example.com",
    }
    response = api_client.post("/flags", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "reason_cleaned" in data
    assert data["reason_cleaned"] is not None
    assert "test@example.com" not in data["reason_cleaned"]
    assert "[EMAIL]" in data["reason_cleaned"]


# ---------------------------------------------------------------------------
# Test 4: Middleware only intercepts POST /messages and POST /flags
# ---------------------------------------------------------------------------


def test_get_health_not_intercepted(api_client):
    """GET /health passes through middleware without interference."""
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Test 5: Cleaned text has markdown stripped, URLs replaced, PII scrubbed
# ---------------------------------------------------------------------------


def test_cleaning_in_middleware(api_client):
    """Middleware applies TextCleaner: markdown, URLs, PII are all cleaned."""
    payload = {
        "text": "Check **this** https://evil.com and email bad@actor.com",
        "user_id": "user-1",
        "source": "real",
    }
    response = api_client.post("/messages", json=payload)
    assert response.status_code == 200
    data = response.json()

    cleaned = data["cleaned_text"]
    assert "bad@actor.com" not in cleaned
    assert "[EMAIL]" in cleaned
    assert "https://evil.com" not in cleaned
    assert "[URL]" in cleaned
    assert "**" not in cleaned


# ---------------------------------------------------------------------------
# Test 6: Verify TextCleaner is invoked (cleaned differs from raw)
# ---------------------------------------------------------------------------


def test_cleaned_differs_from_raw(api_client):
    """cleaned_text differs from raw_text when input needs cleaning."""
    payload = {
        "text": "**bold** text with https://example.com",
        "user_id": "user-1",
    }
    response = api_client.post("/messages", json=payload)
    data = response.json()
    assert data["raw_text"] != data["cleaned_text"]
    assert "**bold**" in data["raw_text"]
    assert "**" not in data["cleaned_text"]
