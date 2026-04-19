"""Integration tests for PostgreSQL schema. Requires running Docker services."""



def test_users_table_exists(pg_conn):
    cur = pg_conn.cursor()
    cur.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'users'"
    )
    columns = {row[0] for row in cur.fetchall()}
    assert "id" in columns
    assert "username" in columns
    assert "email" in columns
    assert "source" in columns
    assert "created_at" in columns


def test_messages_table_exists_with_toxicity_columns(pg_conn):
    cur = pg_conn.cursor()
    cur.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'messages'"
    )
    columns = {row[0] for row in cur.fetchall()}
    assert "id" in columns
    assert "user_id" in columns
    assert "text" in columns
    assert "is_toxicity" in columns
    assert "is_suicide" in columns
    assert "source" in columns


def test_messages_has_gin_index(pg_conn):
    cur = pg_conn.cursor()
    cur.execute(
        "SELECT indexname FROM pg_indexes "
        "WHERE tablename = 'messages' AND indexname = 'idx_messages_text_fts'"
    )
    result = cur.fetchone()
    assert result is not None, (
        "GIN full-text search index idx_messages_text_fts not found"
    )


def test_flags_table_exists(pg_conn):
    cur = pg_conn.cursor()
    cur.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'flags'"
    )
    columns = {row[0] for row in cur.fetchall()}
    assert "id" in columns
    assert "message_id" in columns
    assert "flagged_by" in columns
    assert "is_verified" in columns


def test_moderation_table_exists(pg_conn):
    cur = pg_conn.cursor()
    cur.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'moderation'"
    )
    columns = {row[0] for row in cur.fetchall()}
    assert "id" in columns
    assert "message_id" in columns
    assert "action" in columns
    assert "confidence" in columns
    assert "model_version" in columns


def test_source_check_constraint(pg_conn):
    cur = pg_conn.cursor()
    cur.execute(
        "SELECT constraint_name FROM information_schema.check_constraints "
        "WHERE constraint_name LIKE '%source%'"
    )
    result = cur.fetchone()
    assert result is not None, "Source CHECK constraint not found"
