"""Unit tests for the TextCleaner pipeline module.

Each test maps to a specific ONLINE requirement:
    - ONLINE-01: Markdown / HTML stripping
    - ONLINE-02: Emoji standardization
    - ONLINE-03: URL extraction
    - ONLINE-04: PII scrubbing
    - ONLINE-05: Unicode normalization
    - ONLINE-06: Full pipeline orchestration
"""

from src.data.text_cleaner import (
    TextCleaner,
    extract_urls,
    fix_unicode,
    scrub_pii,
    standardize_emojis,
    strip_markdown,
)

# ---------------------------------------------------------------------------
# Individual step tests
# ---------------------------------------------------------------------------


def test_fix_unicode():
    """ONLINE-05: ftfy fixes mojibake encoding issues."""
    result = fix_unicode("cafÃ©")
    assert "café" in result


def test_strip_markdown_html():
    """ONLINE-01: HTML tags are stripped from text."""
    result = strip_markdown("<b>bold</b> text")
    assert result == "bold text"


def test_strip_markdown_syntax():
    """ONLINE-01: Markdown syntax markers are removed."""
    result = strip_markdown("**bold** text")
    assert result == "bold text"


def test_extract_urls():
    """ONLINE-03: Single URL is replaced with [URL] placeholder."""
    result = extract_urls("visit https://example.com now")
    assert result == "visit [URL] now"


def test_extract_urls_multiple():
    """ONLINE-03: Multiple URLs are each replaced with [URL]."""
    result = extract_urls("https://a.com and https://b.com")
    assert result == "[URL] and [URL]"


def test_standardize_emojis():
    """ONLINE-02: Unicode emojis are converted to :shortcode: format."""
    result = standardize_emojis("I am happy 😂")
    assert ":face_with_tears_of_joy:" in result


def test_standardize_emojis_no_emoji():
    """ONLINE-02: Plain text without emojis is unchanged."""
    result = standardize_emojis("plain text")
    assert result == "plain text"


def test_scrub_pii_email():
    """ONLINE-04: Email addresses are replaced with [EMAIL]."""
    result = scrub_pii("email me at test@example.com")
    assert result == "email me at [EMAIL]"


def test_scrub_pii_phone():
    """ONLINE-04: Phone numbers are replaced with [PHONE]."""
    result = scrub_pii("call 555-123-4567")
    assert result == "call [PHONE]"


def test_scrub_pii_username():
    """ONLINE-04: @username mentions are replaced with [USER]."""
    result = scrub_pii("hello @john_doe")
    assert result == "hello [USER]"


# ---------------------------------------------------------------------------
# Pipeline integration tests
# ---------------------------------------------------------------------------


def test_text_cleaner_pipeline_order():
    """ONLINE-06: Full pipeline cleans all element types in correct order."""
    tc = TextCleaner()
    raw = "Hey @alice email a@b.com https://x.com cafÃ© 😂 **bold**"
    cleaned = tc.clean(raw)

    assert "@alice" not in cleaned
    assert "[USER]" in cleaned
    assert "a@b.com" not in cleaned
    assert "[EMAIL]" in cleaned
    assert "https://x.com" not in cleaned
    assert "[URL]" in cleaned
    assert "café" in cleaned
    assert ":face_with_tears_of_joy:" in cleaned
    assert "**" not in cleaned


def test_text_cleaner_custom_steps():
    """ONLINE-06: Custom steps list restricts pipeline to only those steps."""
    tc = TextCleaner(steps=[extract_urls, scrub_pii])
    raw = "visit https://x.com and email a@b.com"
    cleaned = tc.clean(raw)

    assert "[URL]" in cleaned
    assert "[EMAIL]" in cleaned
    # Emoji should NOT be processed (custom steps exclude standardize_emojis)
    raw_emoji = "happy 😂"
    cleaned_emoji = tc.clean(raw_emoji)
    assert "😂" in cleaned_emoji


def test_text_cleaner_empty_input():
    """Edge case: empty string returns empty string."""
    tc = TextCleaner()
    assert tc.clean("") == ""


def test_text_cleaner_no_side_effects():
    """Input string is not mutated by the cleaning pipeline."""
    tc = TextCleaner()
    original = "hello @user test@email.com https://example.com"
    _ = tc.clean(original)
    assert original == "hello @user test@email.com https://example.com"
