"""Tests for compile_training_data.py — batch pipeline (BATCH-01 through BATCH-05)."""

import pandas as pd
import pytest


@pytest.fixture
def csv_sample_df() -> pd.DataFrame:
    """Sample DataFrame matching combined_dataset.csv schema (text, is_suicide, is_toxicity).

    Includes known data quality issues for quality gate testing:
    - 2 rows with '#ERROR!' text (DATA_ISSUES.md Issue 4)
    - 1 row with text < 10 chars (Issue 5)
    - 1 row with text > 5000 chars (Issue 5)
    - Mix of label combinations (1_0, 0_0, 0_1) — no 1_1 rows (Issue 2)
    """
    long_text = "x" * 5001
    return pd.DataFrame(
        {
            "text": [
                "I feel so sad and hopeless today",
                "You are such an idiot, kill yourself",
                "#ERROR!",
                "#ERROR!",
                "Hi",
                long_text,
                "This is a normal message about cats",
                "I want to end my life",
                "Go away you stupid person",
                "The weather is nice today",
                "I am thinking about suicide",
                "What a terrible awful person you are",
                "Nice work on the project",
                "I feel like nobody would miss me",
                "Shut up you moron",
                "Let's meet for lunch tomorrow",
            ],
            "cleaned_text": [
                "I feel so sad and hopeless today",
                "You are such an idiot, kill yourself",
                "#ERROR!",
                "#ERROR!",
                "Hi",
                long_text,
                "This is a normal message about cats",
                "I want to end my life",
                "Go away you stupid person",
                "The weather is nice today",
                "I am thinking about suicide",
                "What a terrible awful person you are",
                "Nice work on the project",
                "I feel like nobody would miss me",
                "Shut up you moron",
                "Let's meet for lunch tomorrow",
            ],
            "is_suicide": [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            "is_toxicity": [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            "source": ["real"] * 16,
        }
    )


@pytest.fixture
def pg_sample_df() -> pd.DataFrame:
    """Sample DataFrame matching PostgreSQL messages+moderation join schema.

    Represents the output of the incremental query with is_toxicity column.
    """
    return pd.DataFrame(
        {
            "message_id": [f"uuid-{i}" for i in range(8)],
            "cleaned_text": [
                "I feel so sad and hopeless today",
                None,  # NULL cleaned_text — triggers TextCleaner fallback (D-18)
                "This is a normal message",
                "I want to end my life",
                "Go away you stupid person",
                "Nice work on the project",
                None,  # Another NULL
                "Shut up you moron",
            ],
            "text": [
                "I feel so sad and hopeless today",
                "RAW: some unclean text here",
                "This is a normal message",
                "I want to end my life",
                "Go away you stupid person",
                "Nice work on the project",
                "RAW: another unclean text",
                "Shut up you moron",
            ],
            "is_suicide": [1, 0, 0, 1, 0, 0, 0, 0],
            "is_toxicity": [0, 0, 0, 0, 1, 0, 0, 1],
            "source": ["real"] * 8,
            "created_at": pd.to_datetime(
                [
                    "2026-01-01T10:00:00Z",
                    "2026-01-02T10:00:00Z",
                    "2026-01-03T10:00:00Z",
                    "2026-01-04T10:00:00Z",
                    "2026-01-05T10:00:00Z",
                    "2026-01-06T10:00:00Z",
                    "2026-01-07T10:00:00Z",
                    "2026-01-08T10:00:00Z",
                ]
            ),
            "decided_at": pd.to_datetime(
                [
                    "2026-01-01T12:00:00Z",
                    "2026-01-02T12:00:00Z",
                    "2026-01-03T12:00:00Z",
                    "2026-01-04T12:00:00Z",
                    "2026-01-05T12:00:00Z",
                    "2026-01-06T12:00:00Z",
                    "2026-01-07T12:00:00Z",
                    "2026-01-08T12:00:00Z",
                ]
            ),
        }
    )


# ---------------------------------------------------------------------------
# GE validation tests (replaces quality gate, D-01 through D-05)
# ---------------------------------------------------------------------------


def test_ge_validation_replaces_quality_gate(csv_sample_df):
    """GE validation is called instead of apply_quality_gate() (D-01)."""
    from src.data.compile_training_data import select_output_columns
    from src.data.data_quality import validate_training_data

    df = select_output_columns(csv_sample_df)

    success, results = validate_training_data(df)
    assert isinstance(success, bool)
    assert "statistics" in results
    assert "expectation_results" in results


def test_ge_validation_catches_error_rows(csv_sample_df):
    """GE validation catches #ERROR! rows (replaces D-21)."""
    from src.data.compile_training_data import select_output_columns
    from src.data.data_quality import validate_training_data

    df = select_output_columns(csv_sample_df)
    success, results = validate_training_data(df)

    regex_results = [
        r
        for r in results["expectation_results"]
        if "regex" in r.expectation_config.type
    ]
    assert len(regex_results) == 1
    assert not regex_results[0].success


def test_ge_validation_catches_short_texts(csv_sample_df):
    """GE validation catches texts below 10 chars (replaces D-22)."""
    from src.data.compile_training_data import select_output_columns
    from src.data.data_quality import validate_training_data

    df = select_output_columns(csv_sample_df)
    success, results = validate_training_data(df)

    length_results = [
        r
        for r in results["expectation_results"]
        if "lengths" in r.expectation_config.type
    ]
    assert len(length_results) == 1
    assert not length_results[0].success


def test_ge_validation_warn_and_continue(csv_sample_df):
    """GE validation failures don't halt pipeline (D-03)."""
    from src.data.compile_training_data import select_output_columns
    from src.data.data_quality import validate_training_data

    df = select_output_columns(csv_sample_df)
    success, results = validate_training_data(df)
    assert isinstance(success, bool)


# ---------------------------------------------------------------------------
# Quality gate tests (DATA_ISSUES.md Issues 4, 5)
# ---------------------------------------------------------------------------


def test_quality_gate_removes_error_rows():
    """Quality gate removes rows containing '#ERROR!' (DATA_ISSUES.md Issue 4)."""
    from src.data.compile_training_data import apply_quality_gate

    df = pd.DataFrame(
        {
            "cleaned_text": [
                "#ERROR! duplicate",
                "valid text that is long enough",
                "short",
                "another valid and sufficiently long text",
                "#ERROR!",
            ]
        }
    )
    result = apply_quality_gate(df)
    assert not result["cleaned_text"].str.contains("#ERROR!", na=False).any()


def test_quality_gate_filters_short_texts():
    """Quality gate removes texts below 10 chars (DATA_ISSUES.md Issue 5)."""
    from src.data.compile_training_data import apply_quality_gate

    df = pd.DataFrame(
        {
            "cleaned_text": [
                "ok",
                "valid text that is long enough",
                "short",
                "another valid and sufficiently long text",
            ]
        }
    )
    result = apply_quality_gate(df)
    assert (result["cleaned_text"].str.len() >= 10).all()


def test_quality_gate_caps_long_texts():
    """Quality gate caps texts above 5000 chars (DATA_ISSUES.md Issue 5)."""
    from src.data.compile_training_data import apply_quality_gate

    long_text = "x" * 6000
    df = pd.DataFrame(
        {
            "cleaned_text": [
                "valid text that is long enough",
                long_text,
            ]
        }
    )
    result = apply_quality_gate(df)
    assert (result["cleaned_text"].str.len() <= 5000).all()


# ---------------------------------------------------------------------------
# Temporal leakage prevention tests (BATCH-02, D-04 through D-06)
# ---------------------------------------------------------------------------


def test_temporal_leakage_filter_excludes_future_decisions():
    """Only messages with created_at < decided_at are included (D-05)."""
    from src.data.compile_training_data import filter_temporal_leakage

    df = pd.DataFrame(
        {
            "message_id": ["a", "b", "c"],
            "created_at": pd.to_datetime(
                ["2026-01-01T10:00:00Z", "2026-01-02T10:00:00Z", "2026-01-03T10:00:00Z"]
            ),
            "decided_at": pd.to_datetime(
                ["2026-01-01T12:00:00Z", "2026-01-01T08:00:00Z", "2026-01-03T12:00:00Z"]
            ),
        }
    )
    result = filter_temporal_leakage(df)
    assert len(result) == 2
    assert "b" not in result["message_id"].values


# ---------------------------------------------------------------------------
# Output schema tests (BATCH-04, D-15 through D-17)
# ---------------------------------------------------------------------------


def test_output_schema_has_five_columns():
    """Output contains exactly 5 columns: cleaned_text, is_suicide, is_toxicity, source, message_id (D-15)."""
    from src.data.compile_training_data import select_output_columns

    df = pd.DataFrame(
        {
            "message_id": ["a"],
            "cleaned_text": ["hello"],
            "is_suicide": [0],
            "is_toxicity": [0],
            "source": ["real"],
            "created_at": pd.to_datetime(["2026-01-01T10:00:00Z"]),
            "decided_at": pd.to_datetime(["2026-01-01T12:00:00Z"]),
        }
    )
    result = select_output_columns(df)
    expected_cols = {
        "cleaned_text",
        "is_suicide",
        "is_toxicity",
        "source",
        "message_id",
    }
    assert set(result.columns) == expected_cols


# ---------------------------------------------------------------------------
# Stratified split tests (BATCH-05, D-12 through D-14)
# ---------------------------------------------------------------------------


def test_stratified_split_proportions():
    """Stratified split preserves class proportions across train/val/test (D-13, D-14)."""
    from src.data.compile_training_data import stratified_split

    # Create dataset with known class distribution
    n = 1000
    df = pd.DataFrame(
        {
            "cleaned_text": [f"message {i}" for i in range(n)],
            "is_suicide": [1] * 700 + [0] * 300,
            "is_toxicity": [0] * 700 + [1] * 40 + [0] * 260,
            "source": ["real"] * n,
            "message_id": [f"uuid-{i}" for i in range(n)],
        }
    )
    train, val, test = stratified_split(df)

    # Check proportions are approximately 70/15/15
    assert abs(len(train) / n - 0.70) < 0.05
    assert abs(len(val) / n - 0.15) < 0.05
    assert abs(len(test) / n - 0.15) < 0.05

    # Check no label_combo column in output
    assert "label_combo" not in train.columns


def test_stratified_split_filters_empty_class():
    """Stratified split filters out empty stratification classes before splitting."""
    from src.data.compile_training_data import stratified_split

    df = pd.DataFrame(
        {
            "cleaned_text": [f"message {i}" for i in range(100)],
            "is_suicide": [1] * 70 + [0] * 30,
            "is_toxicity": [0] * 70 + [1] * 30,
            "source": ["real"] * 100,
            "message_id": [f"uuid-{i}" for i in range(100)],
        }
    )
    # This dataset has no 1_1 rows — should still split without error
    train, val, test = stratified_split(df)
    assert len(train) + len(val) + len(test) == len(df)


# ---------------------------------------------------------------------------
# Versioned upload test (BATCH-03, D-09 through D-11)
# ---------------------------------------------------------------------------


def test_version_format():
    """Version string follows vYYYYMMDD-HHMMSS format (D-09)."""
    from src.data.compile_training_data import generate_version

    version = generate_version()
    import re

    assert re.match(r"v\d{8}-\d{6}", version)
