from datetime import datetime, timezone
from unittest.mock import MagicMock, patch


def test_dashboard_shows_messages_without_human_labels(api_client):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        (
            "11111111-1111-1111-1111-111111111111",
            "recent message",
            "recent message",
            datetime(2026, 4, 22, 6, 0, tzinfo=timezone.utc),
        )
    ]
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    with patch("src.api.routes.dashboard.get_db_connection", return_value=mock_conn):
        response = api_client.get("/dashboard")

    assert response.status_code == 200
    assert "recent message" in response.text

    sql = mock_cursor.execute.call_args[0][0]
    assert "mod.action = 'labeled'" in sql
    assert "NOT EXISTS" in sql
