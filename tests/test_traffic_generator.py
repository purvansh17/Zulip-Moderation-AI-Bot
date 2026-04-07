"""Tests for synthetic HTTP traffic generator.

Tests cover CSV loading, POST dispatch, and RPS rate control.
"""

import asyncio
import csv
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.synthetic_traffic_generator import (
    _build_flag_payload,
    _build_message_payload,
    load_csv_messages,
    run_traffic_generator,
    send_message,
)


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    """Create a temporary CSV file with known content."""
    csv_path = tmp_path / "test_data.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "is_suicide", "is_toxicity"])
        writer.writeheader()
        writer.writerow({"text": "Hello world", "is_suicide": "0", "is_toxicity": "0"})
        writer.writerow(
            {"text": "This is toxic content", "is_suicide": "0", "is_toxicity": "1"}
        )
        writer.writerow(
            {"text": "Another message", "is_suicide": "0", "is_toxicity": "0"}
        )
        writer.writerow({"text": "  ", "is_suicide": "0", "is_toxicity": "0"})  # empty
    return csv_path


@pytest.mark.asyncio
async def test_load_csv_messages(csv_file: Path) -> None:
    """Test that CSV loading returns a list of non-empty strings from text column."""
    messages = await load_csv_messages(str(csv_file))

    assert isinstance(messages, list)
    # 4 rows but one has only whitespace — should be filtered
    assert len(messages) == 3
    for msg in messages:
        assert isinstance(msg, str)
        assert len(msg) > 0

    assert messages[0] == "Hello world"
    assert messages[1] == "This is toxic content"
    assert messages[2] == "Another message"


@pytest.mark.asyncio
async def test_load_csv_messages_empty_file(tmp_path: Path) -> None:
    """Test loading from an empty CSV returns empty list."""
    csv_path = tmp_path / "empty.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text"])
        writer.writeheader()

    messages = await load_csv_messages(str(csv_path))
    assert messages == []


@pytest.mark.asyncio
async def test_send_message_makes_post_request() -> None:
    """Test that send_message sends a POST with correct payload."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={"status": "accepted", "message_id": "123"}
    )

    # Create a mock context manager for session.post
    mock_cm = AsyncMock()
    mock_cm.__aenter__.return_value = mock_response
    mock_cm.__aexit__.return_value = None

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_cm)

    payload = {"text": "test message", "user_id": "user-123", "source": "real"}
    result = await send_message(mock_session, "http://localhost:8000/messages", payload)

    # Verify POST was called with correct URL and payload
    mock_session.post.assert_called_once_with(
        "http://localhost:8000/messages", json=payload
    )
    assert result == {"status": "accepted", "message_id": "123"}


@pytest.mark.asyncio
async def test_send_message_handles_connection_error() -> None:
    """Test that send_message returns None on connection error."""
    import aiohttp

    mock_session = MagicMock()
    mock_post_cm = AsyncMock()
    mock_post_cm.__aenter__.side_effect = aiohttp.ClientError("Connection refused")
    mock_session.post = MagicMock(return_value=mock_post_cm)

    payload = {"text": "test", "user_id": "user-1", "source": "real"}
    result = await send_message(mock_session, "http://localhost:8000/messages", payload)

    assert result is None


@pytest.mark.asyncio
async def test_send_message_handles_non_200() -> None:
    """Test that send_message returns None on non-200 status."""
    mock_response = AsyncMock()
    mock_response.status = 500
    mock_response.text = AsyncMock(return_value="Internal Server Error")

    mock_cm = AsyncMock()
    mock_cm.__aenter__.return_value = mock_response
    mock_cm.__aexit__.return_value = None

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_cm)

    payload = {"text": "test", "user_id": "user-1", "source": "real"}
    result = await send_message(mock_session, "http://localhost:8000/messages", payload)

    assert result is None


def test_build_message_payload(csv_file: Path) -> None:
    """Test _build_message_payload returns valid payload structure."""
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        messages = [row["text"] for row in reader if row["text"].strip()]

    payload = _build_message_payload(messages)

    assert "text" in payload
    assert "user_id" in payload
    assert "source" in payload
    assert payload["source"] == "real"
    assert payload["text"] in messages
    assert len(payload["user_id"]) > 0  # UUID string


def test_build_flag_payload() -> None:
    """Test _build_flag_payload returns valid flag payload structure."""
    payload = _build_flag_payload()

    assert "message_id" in payload
    assert "flagged_by" in payload
    assert "reason" in payload
    assert len(payload["message_id"]) > 0  # UUID string
    assert len(payload["flagged_by"]) > 0


@pytest.mark.asyncio
async def test_run_traffic_generator_respects_rps() -> None:
    """Test that traffic generator dispatches approximately at target RPS."""
    csv_file_content = (
        "text,is_suicide,is_toxicity\nTest message 1,0,0\nTest message 2,0,0\n"
    )

    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write(csv_file_content)
        csv_path = f.name

    try:
        # Mock aiohttp.ClientSession to track dispatches
        dispatch_times: list[float] = []

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "ok"})

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_response
        mock_cm.__aexit__.return_value = None

        mock_session = MagicMock()
        original_post = mock_session.post

        def tracking_post(*args, **kwargs):
            dispatch_times.append(time.monotonic())
            return mock_cm

        mock_session.post = MagicMock(side_effect=tracking_post)

        # Patch aiohttp.ClientSession to return our mock
        with patch(
            "src.data.synthetic_traffic_generator.aiohttp.ClientSession"
        ) as mock_session_cls:
            mock_session_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            # Patch generate_synthetic_message to return CSV fallback
            with patch(
                "src.data.synthetic_traffic_generator.generate_synthetic_message",
                new_callable=AsyncMock,
                return_value=None,
            ):
                result = await run_traffic_generator(
                    base_url="http://localhost:8000",
                    duration_seconds=2,
                    rps=5,
                    csv_path=csv_path,
                )

        # At 5 RPS for 2 seconds, expect ~10 requests (±50% for timing variance)
        assert result["total"] >= 5, f"Expected >=5 requests, got {result['total']}"
        assert result["total"] <= 20, f"Expected <=20 requests, got {result['total']}"

    finally:
        import os

        os.unlink(csv_path)


@pytest.mark.asyncio
async def test_run_traffic_generator_aborts_on_empty_csv(tmp_path: Path) -> None:
    """Test that generator aborts gracefully when CSV is empty."""
    csv_path = tmp_path / "empty.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text"])
        writer.writeheader()

    result = await run_traffic_generator(
        base_url="http://localhost:8000",
        duration_seconds=1,
        rps=5,
        csv_path=str(csv_path),
    )

    assert result["total"] == 0
