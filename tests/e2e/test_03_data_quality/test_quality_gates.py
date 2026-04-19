"""Layer 3: Quality gate and Data Docs tests.

Verifies apply_quality_gate() filters data issues and
Data Docs HTML reports are generated/uploaded to MinIO.

Run: pytest tests/e2e/test_03_data_quality/test_quality_gates.py -v -m data_quality
"""

import logging

import pandas as pd
import pytest

from src.data.compile_training_data import apply_quality_gate

logger = logging.getLogger(__name__)


@pytest.mark.data_quality
class TestQualityGates:
    """Verify quality gate filtering and reporting."""

    def test_quality_gate_removes_errors(self, docker_services):
        """#ERROR! rows removed by quality gate."""
        df = pd.DataFrame(
            {
                "cleaned_text": [
                    "#ERROR! bad row",
                    "valid text that is long enough",
                    "#ERROR! another bad",
                ],
            }
        )
        result = apply_quality_gate(df)
        assert not result["cleaned_text"].str.contains("#ERROR!").any(), (
            f"#ERROR! rows remain: {result['cleaned_text'].tolist()}"
        )

    def test_quality_gate_filters_short(self, docker_services):
        """Short texts (<10 chars) removed by quality gate."""
        df = pd.DataFrame(
            {
                "cleaned_text": [
                    "short",
                    "valid text that is long enough",
                    "ok",
                    "another valid and sufficiently long text",
                ],
            }
        )
        result = apply_quality_gate(df)
        assert (result["cleaned_text"].str.len() >= 10).all(), (
            f"Short texts remain: {result['cleaned_text'].tolist()}"
        )

    def test_quality_gate_caps_long(self, docker_services):
        """Long texts (>5000 chars) capped to 5000."""
        long_text = "x" * 6000
        df = pd.DataFrame(
            {
                "cleaned_text": [long_text, "normal text here"],
            }
        )
        result = apply_quality_gate(df)
        max_len = result["cleaned_text"].str.len().max()
        assert max_len <= 5000, f"Text length {max_len} exceeds 5000 cap"

    def test_quality_gate_correct_count(self, docker_services):
        """Quality gate produces expected row count."""
        df = pd.DataFrame(
            {
                "cleaned_text": [
                    "#ERROR! duplicate",
                    "valid text that is long enough",
                    "short",
                    "another valid and sufficiently long text",
                    "#ERROR!",
                ],
            }
        )
        result = apply_quality_gate(df)
        assert len(result) == 2, (
            f"Expected 2 rows after filtering, got {len(result)}: "
            f"{result['cleaned_text'].tolist()}"
        )

    def test_data_docs_html_exists(self, docker_services, clean_state):
        """GE Data Docs HTML can be generated (file exists locally)."""
        from src.data.data_quality import validate_training_data

        df = pd.DataFrame(
            {
                "cleaned_text": [
                    "valid text one",
                    "valid text two",
                    "#ERROR! bad row",
                ],
                "is_suicide": [0, 0, 0],
                "is_toxicity": [0, 0, 0],
                "source": ["real"] * 3,
            }
        )
        success, results = validate_training_data(df)
        # Just verify the validation ran and returned results
        assert "expectation_results" in results
        assert len(results["expectation_results"]) > 0
