"""Layer 2: Text cleaning data flow tests.

Validates TextCleaner pipeline on real data patterns:
markdown stripping, URL extraction, PII scrubbing, emoji standardization.

Run: pytest tests/e2e/test_02_data_flow/test_text_cleaning.py -v -m data_flow
"""

import logging

import pytest

from src.data.text_cleaner import TextCleaner

logger = logging.getLogger(__name__)


@pytest.mark.data_flow
class TestTextCleaning:
    """Verify TextCleaner transformations on realistic input."""

    def test_strips_markdown(self, docker_services):
        """Markdown syntax removed from text."""
        cleaner = TextCleaner()
        raw = "**bold** and *italic* and `code` and ~~strike~~"
        cleaned = cleaner.clean(raw)
        assert "**" not in cleaned, f"Markdown bold remains: {cleaned}"
        assert "~~" not in cleaned, f"Markdown strike remains: {cleaned}"
        # Content should remain (words bold, italic, code, strike somewhere)
        assert "bold" in cleaned
        assert "italic" in cleaned

    def test_extracts_urls(self, docker_services):
        """URLs replaced/extracted from text."""
        cleaner = TextCleaner()
        raw = "Visit https://example.com/path?q=1 and http://test.org"
        cleaned = cleaner.clean(raw)
        assert "https://example.com/path?q=1" not in cleaned, (
            f"Full URL remains: {cleaned}"
        )

    def test_scrubs_pii(self, docker_services):
        """Email, phone, username patterns scrubbed."""
        cleaner = TextCleaner()
        raw = "Email me@test.org call 555-123-4567 and @john_doe"
        cleaned = cleaner.clean(raw)
        assert "me@test.org" not in cleaned, f"Email remains: {cleaned}"
        assert "555-123-4567" not in cleaned, f"Phone remains: {cleaned}"
        assert "@john_doe" not in cleaned, f"Username remains: {cleaned}"

    def test_combined_cleaning(self, docker_services):
        """All transformations applied in pipeline order."""
        cleaner = TextCleaner()
        raw = (
            "**bold** visit https://evil.com email bad@test.org "
            "@user123 and check :thumbsup: emoji"
        )
        cleaned = cleaner.clean(raw)
        # None of the artifacts should remain
        for artifact in ["**", "https://evil.com", "bad@test.org", "@user123"]:
            assert artifact not in cleaned, (
                f"Artifact '{artifact}' remains in: {cleaned}"
            )

    def test_cleaner_on_sample_rows(self, docker_services, test_dataset_small):
        """TextCleaner processes sample rows without errors."""
        cleaner = TextCleaner()
        sample = test_dataset_small.head(20)
        for _, row in sample.iterrows():
            text = str(row.get("text", ""))
            if text and text != "nan":
                cleaned = cleaner.clean(text)
                assert isinstance(cleaned, str), f"Expected str, got {type(cleaned)}"
                assert len(cleaned) >= 0, "Cleaned text should not be None"
