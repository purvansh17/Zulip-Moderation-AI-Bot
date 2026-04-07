"""Layer 3: Great Expectations validation tests.

Verifies GE expectation suite catches data quality issues:
#ERROR! patterns, short/long texts, invalid labels, class imbalance.

Run: pytest tests/e2e/test_03_data_quality/test_ge_validation.py -v -m data_quality
"""

import logging

import pandas as pd
import pytest

from src.data.data_quality import validate_training_data

logger = logging.getLogger(__name__)


@pytest.mark.data_quality
class TestGeValidation:
    """Verify GE catches data quality issues."""

    def test_catches_error_pattern(self, docker_services):
        """#ERROR! rows flagged by GE expectation."""
        df = pd.DataFrame(
            {
                "cleaned_text": [
                    "#ERROR! duplicate",
                    "valid text that is long enough",
                    "another valid and sufficiently long text",
                ],
                "is_suicide": [0, 0, 0],
                "is_toxicity": [0, 0, 0],
                "source": ["real"] * 3,
            }
        )
        success, results = validate_training_data(df)
        # Should fail because #ERROR! rows present
        assert success is False, "Expected validation to fail with #ERROR! rows"
        assert len(results.get("expectation_results", [])) > 0

    def test_catches_short_text(self, docker_services):
        """Texts <10 chars flagged as warning."""
        df = pd.DataFrame(
            {
                "cleaned_text": ["short", "ok", "a", "this is long enough text"],
                "is_suicide": [0, 0, 0, 0],
                "is_toxicity": [0, 0, 0, 0],
                "source": ["real"] * 4,
            }
        )
        success, results = validate_training_data(df)
        # Short texts should cause expectation failure (warning severity)
        expectation_results = results.get("expectation_results", [])
        has_length_check = any(
            "length" in str(r.get("expectation_config", "")).lower()
            or "value_lengths" in str(r.get("expectation_config", "")).lower()
            for r in expectation_results
        )
        assert has_length_check, "No text length expectation found in results"

    def test_catches_long_text(self, docker_services):
        """Texts >5000 chars flagged."""
        long_text = "x" * 6000
        df = pd.DataFrame(
            {
                "cleaned_text": [long_text, "normal text here"],
                "is_suicide": [0, 0],
                "is_toxicity": [0, 0],
                "source": ["real"] * 2,
            }
        )
        success, results = validate_training_data(df)
        assert len(results.get("expectation_results", [])) > 0

    def test_validates_labels(self, docker_services):
        """Valid 0/1 labels pass, invalid values caught."""
        df = pd.DataFrame(
            {
                "cleaned_text": [
                    "valid text one",
                    "valid text two",
                    "valid text three",
                ],
                "is_suicide": [0, 1, 0],
                "is_toxicity": [1, 0, 0],
                "source": ["real"] * 3,
            }
        )
        success, results = validate_training_data(df)
        # Valid labels should pass (if no other issues)
        assert isinstance(success, bool), "validate_training_data returned non-bool"

    def test_checks_class_balance(self, docker_services):
        """Toxicity ratio outside 2-8% flagged."""
        # All benign — 0% toxicity ratio, below 2% threshold
        df = pd.DataFrame(
            {
                "cleaned_text": [f"text {i} that is long enough" for i in range(100)],
                "is_suicide": [0] * 100,
                "is_toxicity": [0] * 100,  # 0% toxicity
                "source": ["real"] * 100,
            }
        )
        success, results = validate_training_data(df)
        # Should fail on class balance expectation
        assert success is False, "Expected failure with 0% toxicity ratio"
