"""Test data loading and stratified sampling for E2E tests."""

import logging
import os

import pandas as pd

from src.utils.config import config

logger = logging.getLogger(__name__)

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "combined_dataset.csv")


def load_stratified_sample(
    csv_path: str = CSV_PATH,
    n_rows: int = 1_000,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Load a stratified sample from combined_dataset.csv.

    Maintains label distribution of is_suicide and is_toxicity by
    sampling proportionally from each label combination group.

    Args:
        csv_path: Path to combined_dataset.csv.
        n_rows: Target sample size (may be slightly less due to rounding).
        random_state: Seed for reproducibility.

    Returns:
        DataFrame with approximately n_rows, stratified by labels.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="warn")

    # Create combined label for stratification
    df["_strat_key"] = (
        df["is_suicide"].astype(str) + "_" + df["is_toxicity"].astype(str)
    )

    # Calculate per-group sample size
    group_sizes = df["_strat_key"].value_counts(normalize=True)
    samples = []
    for group, proportion in group_sizes.items():
        group_df = df[df["_strat_key"] == group]
        group_n = max(1, int(n_rows * proportion))
        group_n = min(group_n, len(group_df))
        samples.append(group_df.sample(n=group_n, random_state=random_state))

    result = pd.concat(samples, ignore_index=True)
    result.drop(columns=["_strat_key"], inplace=True)

    logger.info(
        "Loaded stratified sample: %d rows (target %d)",
        len(result),
        n_rows,
    )
    return result


def load_small_dataset(random_state: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Load 1,000-row stratified sample for fast iteration."""
    return load_stratified_sample(n_rows=1_000, random_state=random_state)


def load_medium_dataset(random_state: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Load 10,000-row stratified sample for realistic volume."""
    return load_stratified_sample(n_rows=10_000, random_state=random_state)
