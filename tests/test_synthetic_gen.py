"""Tests for synthetic data generation. Unit tests run without HuggingFace API."""
from src.data.prompts import PROMPTS_BY_LABEL, LABEL_DISTRIBUTION, GenerationPrompt
from src.data.synthetic_generator import _parse_generated_text, TARGET_TOTAL


def test_prompts_cover_all_labels():
    assert "toxic" in PROMPTS_BY_LABEL
    assert "suicide" in PROMPTS_BY_LABEL
    assert "benign" in PROMPTS_BY_LABEL


def test_prompt_labels_have_correct_flags():
    toxic = PROMPTS_BY_LABEL["toxic"]
    assert toxic.is_toxicity is True
    assert toxic.is_suicide is False

    suicide = PROMPTS_BY_LABEL["suicide"]
    assert suicide.is_suicide is True
    assert suicide.is_toxicity is False

    benign = PROMPTS_BY_LABEL["benign"]
    assert benign.is_suicide is False
    assert benign.is_toxicity is False


def test_label_distribution_sums_to_one():
    total = sum(LABEL_DISTRIBUTION.values())
    assert abs(total - 1.0) < 0.01, f"Label distribution sums to {total}, expected ~1.0"


def test_label_distribution_rebalances_minority_classes():
    # Toxic should be > 10% (original ratio) to oversample
    assert LABEL_DISTRIBUTION["toxic"] >= 0.25, "Toxic class should be oversampled"
    # Suicide should be > 22% (original ratio) to oversample
    assert LABEL_DISTRIBUTION["suicide"] >= 0.25, "Suicide class should be oversampled"


def test_parse_generated_text_numbered_list():
    raw = "1. Hello world\n2. This is a test\n3. Another message"
    result = _parse_generated_text(raw)
    assert len(result) == 3
    assert result[0] == "Hello world"
    assert result[1] == "This is a test"
    assert result[2] == "Another message"


def test_parse_generated_text_empty_lines():
    raw = "1. First\n\n2. Third\n\n"
    result = _parse_generated_text(raw)
    assert len(result) == 2
    assert result[0] == "First"
    assert result[1] == "Third"


def test_parse_generated_text_no_numbering():
    raw = "Just a plain message\nAnother plain message"
    result = _parse_generated_text(raw)
    assert len(result) == 2
    assert result[0] == "Just a plain message"


def test_target_total_is_reasonable():
    assert 5_000 <= TARGET_TOTAL <= 10_000
