"""Synthetic data generator using HuggingFace Inference API.

Per D-09: Uses Mistral-7B-Instruct-v0.2 via HuggingFace Inference API.
Per D-10: Target ~10K synthetic rows.
Per D-11: Oversamples minority classes (toxic/suicide).
Per D-12: Multi-turn thread prompts for realistic conversations.
Per D-13: Labels assigned from prompt, not post-hoc classification.
Per D-14: source = 'synthetic_hf'.
Per D-15: Upload to zulip-raw-messages/synthetic/.
"""
import csv
import io
import logging
import time
from typing import Optional

from huggingface_hub import InferenceClient

from src.data.prompts import (
    LABEL_DISTRIBUTION,
    PROMPTS_BY_LABEL,
    GenerationPrompt,
    LabelType,
)
from src.utils.config import config
from src.utils.minio_client import get_minio_client

logger = logging.getLogger(__name__)

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
PROVIDER = "featherless-ai"
MAX_RETRIES = 3
RETRY_DELAYS = [5, 10, 20]  # Exponential backoff (Pitfall 4)
MESSAGES_PER_CALL = 10  # Each API call generates 10 messages
TARGET_TOTAL = 10_000  # D-10


def _call_hf_api(
    client: InferenceClient,
    prompt: GenerationPrompt,
    max_retries: int = MAX_RETRIES,
) -> Optional[str]:
    """Call HuggingFace API with exponential backoff retry (Pitfall 4).

    Args:
        client: InferenceClient instance.
        prompt: GenerationPrompt with system and user prompts.
        max_retries: Maximum retry attempts.

    Returns:
        Generated text or None if all retries failed.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat_completion(
                messages=[
                    {"role": "system", "content": prompt.system_prompt},
                    {"role": "user", "content": prompt.user_prompt},
                ],
                model=MODEL_ID,
                max_tokens=512,
                temperature=0.8,
            )
            return response.choices[0].message.content
        except Exception as e:
            delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
            if "429" in str(e):
                logger.warning(
                    "Rate limited (attempt %d/%d), retrying in %ds",
                    attempt + 1,
                    max_retries,
                    delay,
                )
            else:
                logger.warning(
                    "API error (attempt %d/%d): %s, retrying in %ds",
                    attempt + 1,
                    max_retries,
                    e,
                    delay,
                )
            time.sleep(delay)

    logger.error(
        "All %d retries exhausted for label: %s", max_retries, prompt.label_type
    )
    return None


def _parse_generated_text(text: str) -> list[str]:
    """Parse numbered list from generated text.

    Args:
        text: Raw text from API (e.g., "1. message one\n2. message two\n...").

    Returns:
        List of message strings.
    """
    messages = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Remove numbering prefix (e.g., "1. ", "2. ", etc.)
        if line[0].isdigit():
            dot_pos = line.find(".")
            if dot_pos > 0:
                line = line[dot_pos + 1 :].strip()
        if line:
            messages.append(line)
    return messages


def generate_synthetic_data(
    target_total: int = TARGET_TOTAL,
    bucket: str = config.BUCKET_RAW,
) -> dict[LabelType, int]:
    """Generate synthetic data and upload to MinIO.

    Args:
        target_total: Total number of synthetic rows to generate.
        bucket: MinIO bucket name.

    Returns:
        Dict mapping label type to count of rows generated.
    """
    client_hf = InferenceClient(
        provider=PROVIDER,
        api_key=config.HF_TOKEN,
    )
    minio = get_minio_client()

    # Ensure bucket exists
    if not minio.bucket_exists(bucket):
        minio.make_bucket(bucket)

    counts: dict[LabelType, int] = {"toxic": 0, "suicide": 0, "benign": 0}
    all_rows: list[dict] = []

    for label_type, proportion in LABEL_DISTRIBUTION.items():
        target_count = int(target_total * proportion)
        prompt = PROMPTS_BY_LABEL[label_type]
        generated = 0

        logger.info("Generating %d %s messages...", target_count, label_type)

        while generated < target_count:
            raw_text = _call_hf_api(client_hf, prompt)
            if raw_text is None:
                logger.error("Skipping batch for %s — API failed", label_type)
                break

            messages = _parse_generated_text(raw_text)
            for msg in messages:
                if generated >= target_count:
                    break
                all_rows.append(
                    {
                        "text": msg,
                        "is_suicide": prompt.is_suicide,
                        "is_toxicity": prompt.is_toxicity,
                        "source": "synthetic_hf",
                    }
                )
                generated += 1

            counts[label_type] = generated
            logger.info("  %s: %d / %d generated", label_type, generated, target_count)

    # Upload to MinIO as a single CSV
    output = io.StringIO()
    fieldnames = ["text", "is_suicide", "is_toxicity", "source"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)

    csv_bytes = output.getvalue().encode("utf-8")
    object_name = "synthetic/synthetic_data.csv"
    minio.put_object(
        bucket_name=bucket,
        object_name=object_name,
        data=io.BytesIO(csv_bytes),
        length=len(csv_bytes),
        content_type="text/csv",
    )

    logger.info(
        "Uploaded %d synthetic rows to %s/%s", len(all_rows), bucket, object_name
    )
    return counts
