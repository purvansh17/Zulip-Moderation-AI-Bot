"""Layer 4: Resource exhaustion chaos tests.

Verifies pipeline handles large datasets without OOM.

Run: pytest tests/e2e/test_04_chaos/test_resource_exhaustion.py -v -m chaos
"""

import logging
import tracemalloc

import pandas as pd
import pytest

from src.data.text_cleaner import TextCleaner
from src.utils.config import config

logger = logging.getLogger(__name__)

# Memory limit: 512MB (reasonable for 4 vCPU / 16GB VM)
MEMORY_LIMIT_MB = 512


@pytest.mark.chaos
class TestResourceExhaustion:
    """Verify pipeline stays within memory limits."""

    def test_large_dataset_memory(self, docker_services):
        """Processing 100K rows doesn't OOM (chunking works)."""
        tracemalloc.start()

        # Generate 100K rows in memory
        large_df = pd.DataFrame(
            {
                "text": [
                    f"message number {i} with some content to clean"
                    for i in range(100_000)
                ],
                "is_suicide": [0] * 100_000,
                "is_toxicity": [0] * 100_000,
            }
        )

        # Process through TextCleaner in chunks (simulate pipeline behavior)
        cleaner = TextCleaner()
        chunk_size = config.CHUNK_SIZE  # 50,000
        for start in range(0, len(large_df), chunk_size):
            chunk = large_df.iloc[start : start + chunk_size]
            chunk["cleaned"] = chunk["text"].apply(cleaner.clean)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        logger.info("Peak memory: %.1f MB (limit: %d MB)", peak_mb, MEMORY_LIMIT_MB)
        assert peak_mb < MEMORY_LIMIT_MB, (
            f"Peak memory {peak_mb:.1f} MB exceeds {MEMORY_LIMIT_MB} MB limit"
        )

    def test_text_cleaner_batch_memory(self, docker_services):
        """TextCleaner on 10K rows stays within memory bounds."""
        tracemalloc.start()

        cleaner = TextCleaner()
        texts = [
            "**bold** text https://example.com email@test.org @user"
            for _ in range(10_000)
        ]
        cleaned = [cleaner.clean(t) for t in texts]

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        assert peak_mb < MEMORY_LIMIT_MB, (
            f"TextCleaner peak memory {peak_mb:.1f} MB exceeds limit"
        )
        assert len(cleaned) == 10_000
