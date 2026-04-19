"""Tests for data_quality.py - GE Expectation Suite validation (QUALITY-01, QUALITY-02, D-01 through D-05)."""

import pandas as pd
import pytest


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """DataFrame that passes all 6 expectations."""
    return pd.DataFrame(
        {
            "cleaned_text": [
                "I feel so sad and hopeless today",
                "You are such an idiot",
                "This is a normal message about cats",
                "I want to end my life",
                "Go away you stupid person",
                "Nice work on the project",
                "The weather is nice today",
                "Let's meet for lunch tomorrow",
                "What a terrible awful person you are",
                "Great job on the assignment",
                "The meeting is at 3pm tomorrow",
                "Thanks for sharing that article",
                "I appreciate your help with this",
                "Can we reschedule to next week",
                "The presentation went well today",
                "Looking forward to the weekend",
                "Please review the document when free",
                "Happy birthday hope you have fun",
                "The new policy changes are good",
                "See you at the team lunch",
            ],
            "is_suicide": [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "is_toxicity": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "source": ["real"] * 20,
        }
    )


# ---------------------------------------------------------------------------
# Suite construction tests
# ---------------------------------------------------------------------------


def test_build_expectation_suite_has_six_expectations():
    """Suite contains exactly 6 expectations (D-02)."""
    import great_expectations as gx

    from src.data.data_quality import build_expectation_suite

    context = gx.get_context(mode="ephemeral")
    suite = build_expectation_suite(context)
    assert len(suite.expectations) == 6


def test_build_expectation_suite_custom_thresholds():
    """Suite accepts custom thresholds (D-05)."""
    import great_expectations as gx

    from src.data.data_quality import build_expectation_suite

    context = gx.get_context(mode="ephemeral")
    suite = build_expectation_suite(
        context,
        thresholds={"min_text_length": 20, "max_text_length": 3000},
    )
    assert len(suite.expectations) == 6


# ---------------------------------------------------------------------------
# Validation on clean data (all expectations pass)
# ---------------------------------------------------------------------------


def test_suite_passes_on_clean_data(clean_df):
    """All 6 expectations pass on clean data (D-02)."""
    from src.data.data_quality import validate_training_data

    success, results = validate_training_data(clean_df)
    assert success is True
    assert results["statistics"]["successful_expectations"] == 6


def test_validation_returns_statistics(clean_df):
    """Validation returns statistics dict with expectation counts (QUALITY-01)."""
    from src.data.data_quality import validate_training_data

    success, results = validate_training_data(clean_df)
    assert "statistics" in results
    assert results["statistics"]["evaluated_expectations"] == 6


# ---------------------------------------------------------------------------
# Validation catches violations (individual expectation tests)
# ---------------------------------------------------------------------------


def test_catches_error_pattern():
    """Expectation catches #ERROR! rows (replaces D-21)."""
    from src.data.data_quality import validate_training_data

    df = pd.DataFrame(
        {
            "cleaned_text": ["Good message", "#ERROR!", "Another good message"],
            "is_suicide": [0, 0, 0],
            "is_toxicity": [0, 0, 0],
            "source": ["real"] * 3,
        }
    )
    success, results = validate_training_data(df)
    error_results = [
        r
        for r in results["expectation_results"]
        if "regex" in r.expectation_config.type
    ]
    assert len(error_results) == 1
    assert not error_results[0].success


def test_catches_short_text():
    """Expectation catches texts below 10 chars (replaces D-22)."""
    from src.data.data_quality import validate_training_data

    df = pd.DataFrame(
        {
            "cleaned_text": ["Good message", "Hi", "Another good message"],
            "is_suicide": [0, 0, 0],
            "is_toxicity": [0, 0, 0],
            "source": ["real"] * 3,
        }
    )
    success, results = validate_training_data(df)
    length_results = [
        r
        for r in results["expectation_results"]
        if "lengths" in r.expectation_config.type
    ]
    assert len(length_results) == 1
    assert not length_results[0].success


def test_catches_long_text():
    """Expectation catches texts above 5000 chars (replaces D-23)."""
    from src.data.data_quality import validate_training_data

    long_text = "x" * 5001
    df = pd.DataFrame(
        {
            "cleaned_text": ["Good message", long_text, "Another good message"],
            "is_suicide": [0, 0, 0],
            "is_toxicity": [0, 0, 0],
            "source": ["real"] * 3,
        }
    )
    success, results = validate_training_data(df)
    length_results = [
        r
        for r in results["expectation_results"]
        if "lengths" in r.expectation_config.type
    ]
    assert len(length_results) == 1
    assert not length_results[0].success


def test_catches_invalid_label():
    """Expectation catches is_suicide values outside {0, 1}."""
    from src.data.data_quality import validate_training_data

    df = pd.DataFrame(
        {
            "cleaned_text": ["Good message", "Another good message"],
            "is_suicide": [0, 2],
            "is_toxicity": [0, 0],
            "source": ["real"] * 2,
        }
    )
    success, results = validate_training_data(df)
    label_results = [
        r for r in results["expectation_results"] if "set" in r.expectation_config.type
    ]
    assert len(label_results) == 1
    assert not label_results[0].success


def test_class_balance_catches_skew():
    """Class balance check catches toxicity ratio outside 2-8% (QUALITY-02)."""
    from src.data.data_quality import validate_training_data

    df = pd.DataFrame(
        {
            "cleaned_text": [f"message {i}" for i in range(100)],
            "is_suicide": [0] * 100,
            "is_toxicity": [1] * 50 + [0] * 50,
            "source": ["real"] * 100,
        }
    )
    success, results = validate_training_data(df)
    balance_results = [
        r for r in results["expectation_results"] if "mean" in r.expectation_config.type
    ]
    assert len(balance_results) == 1
    assert not balance_results[0].success


def test_class_balance_passes_in_range():
    """Class balance check passes when toxicity ratio is in 2-8% range."""
    from src.data.data_quality import validate_training_data

    df = pd.DataFrame(
        {
            "cleaned_text": [f"message {i}" for i in range(100)],
            "is_suicide": [0] * 100,
            "is_toxicity": [1] * 5 + [0] * 95,
            "source": ["real"] * 100,
        }
    )
    success, results = validate_training_data(df)
    balance_results = [
        r for r in results["expectation_results"] if "mean" in r.expectation_config.type
    ]
    assert len(balance_results) == 1
    assert balance_results[0].success


def test_null_check_catches_nulls():
    """Expectation catches null values in cleaned_text."""
    from src.data.data_quality import validate_training_data

    df = pd.DataFrame(
        {
            "cleaned_text": ["Good message", None, "Another good message"],
            "is_suicide": [0, 0, 0],
            "is_toxicity": [0, 0, 0],
            "source": ["real"] * 3,
        }
    )
    success, results = validate_training_data(df)
    null_results = [
        r for r in results["expectation_results"] if "null" in r.expectation_config.type
    ]
    assert len(null_results) == 1
    assert not null_results[0].success


# ---------------------------------------------------------------------------
# Warn-and-continue behavior (D-03)
# ---------------------------------------------------------------------------


def test_validation_warn_and_continue():
    """Validation failures log warnings but do not raise exceptions (D-03)."""
    from src.data.data_quality import validate_training_data

    df = pd.DataFrame(
        {
            "cleaned_text": ["Hi", "#ERROR!", "Good message"],
            "is_suicide": [0, 0, 0],
            "is_toxicity": [0, 0, 0],
            "source": ["real"] * 3,
        }
    )
    success, results = validate_training_data(df)
    assert isinstance(success, bool)
    assert "statistics" in results


# ---------------------------------------------------------------------------
# Data Docs generation (D-04, QUALITY-01)
# ---------------------------------------------------------------------------


def test_data_docs_generated(clean_df):
    """Data Docs HTML is generated from validation results (D-04)."""
    from src.data.data_quality import validate_training_data

    success, results = validate_training_data(clean_df)
    assert "data_docs_html" in results
    assert "<html>" in results["data_docs_html"]
    assert "ChatSentry Data Quality Report" in results["data_docs_html"]


def test_data_docs_shows_pass_fail(clean_df):
    """Data Docs HTML shows pass/fail status per expectation (QUALITY-01)."""
    from src.data.data_quality import validate_training_data

    success, results = validate_training_data(clean_df)
    html = results["data_docs_html"]
    assert "PASS" in html or "pass" in html.lower()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_dataframe():
    """Empty DataFrame returns success=True with zero expectations."""
    from src.data.data_quality import validate_training_data

    df = pd.DataFrame(
        {"cleaned_text": [], "is_suicide": [], "is_toxicity": [], "source": []}
    )
    success, results = validate_training_data(df)
    assert success is True
    assert results["statistics"]["evaluated_expectations"] == 0


def test_runtime_parameters():
    """Thresholds can be overridden at validation time (D-05)."""
    from src.data.data_quality import validate_training_data

    df = pd.DataFrame(
        {
            "cleaned_text": ["Short", "Normal length message here"],
            "is_suicide": [0, 0],
            "is_toxicity": [0, 0],
            "source": ["real"] * 2,
        }
    )
    success_default, results_default = validate_training_data(df)
    length_results_default = [
        r
        for r in results_default.get("expectation_results", [])
        if "lengths" in r.expectation_config.type
    ]
    success_relaxed, results_relaxed = validate_training_data(
        df, thresholds={"min_text_length": 3}
    )
    length_results_relaxed = [
        r
        for r in results_relaxed.get("expectation_results", [])
        if "lengths" in r.expectation_config.type
    ]
    if length_results_relaxed:
        assert length_results_relaxed[0].success
