"""Layer 4: Data corruption chaos tests.

Verifies pipeline handles malformed, null, and duplicate data.

Run: pytest tests/e2e/test_04_chaos/test_data_corruption.py -v -m chaos
"""

import logging

import pandas as pd
import pytest

from src.data.compile_training_data import apply_quality_gate
from tests.e2e.conftest import corrupt_data

logger = logging.getLogger(__name__)


@pytest.mark.chaos
class TestDataCorruption:
    """Verify graceful handling of corrupted data."""

    def test_null_values_handled(self, docker_services):
        """NULL text/labels don't crash quality gate."""
        import numpy as np

        df = pd.DataFrame(
            {
                "cleaned_text": ["valid text here", None, np.nan, "another valid text"],
            }
        )
        # Should not raise exception
        result = apply_quality_gate(df)
        assert isinstance(result, pd.DataFrame), "Expected DataFrame output"
        # Nulls should be filtered or handled
        assert result["cleaned_text"].notna().all(), "NULL values remain in output"

    def test_duplicate_rows_handled(self, docker_services):
        """Duplicate rows pass through quality gate (dedup is not its job)."""
        df = pd.DataFrame(
            {
                "cleaned_text": [
                    "duplicate text",
                    "duplicate text",
                    "unique text that is long enough",
                ],
            }
        )
        result = apply_quality_gate(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 2, "Expected at least 2 rows (duplicates preserved)"

    def test_empty_dataframe_handled(self, docker_services):
        """Empty DataFrame doesn't crash quality gate."""
        df = pd.DataFrame({"cleaned_text": []})
        result = apply_quality_gate(df)
        assert len(result) == 0, "Expected empty result from empty input"

    def test_corrupt_data_context_manager(self, docker_services):
        """corrupt_data context manager produces valid DataFrames."""
        with corrupt_data("null_values") as df:
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 5, f"Expected 5 rows, got {len(df)}"

        with corrupt_data("duplicates") as df:
            assert isinstance(df, pd.DataFrame)
            assert "message_id" in df.columns, "Duplicates should have message_id"
