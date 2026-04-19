"""Tests for CSV chunking behavior. Unit tests run without Docker services."""

import os

import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "combined_dataset.csv")
CHUNK_SIZE = 50_000


def test_csv_file_exists():
    assert os.path.exists(CSV_PATH), f"CSV file not found at {CSV_PATH}"


def test_csv_has_expected_columns():
    first_chunk = pd.read_csv(CSV_PATH, nrows=5)
    assert "text" in first_chunk.columns, "Missing 'text' column"
    assert "is_suicide" in first_chunk.columns, "Missing 'is_suicide' column"
    assert "is_toxicity" in first_chunk.columns, "Missing 'is_toxicity' column"


def test_csv_chunking_produces_expected_chunks():
    chunk_count = 0
    for chunk in pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE):
        chunk_count += 1
        assert len(chunk) <= CHUNK_SIZE, f"Chunk {chunk_count} exceeds CHUNK_SIZE"

    # Actual: 391,645 rows / 50,000 = 7.83 -> 8 chunks
    # Note: wc -l reports ~1.58M lines due to embedded newlines in text column
    assert chunk_count == 8, f"Expected 8 chunks, got {chunk_count}"


def test_csv_total_row_count():
    total_rows = sum(len(chunk) for chunk in pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE))
    assert total_rows == 391_645, f"Expected 391,645 rows, got {total_rows}"
