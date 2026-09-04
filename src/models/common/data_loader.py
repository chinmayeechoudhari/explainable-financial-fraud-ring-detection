"""Utilities for loading transaction-level modeling splits safely.

The loader keeps raw account identifiers out of tabular baselines and makes the
feature contract explicit. It is deliberately model-agnostic: encoding and
scaling are handled by the estimator pipeline rather than mutating the source
CSV files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "modeling_splits"
TARGET_COLUMN = "Is Laundering"

NUMERIC_FEATURES = [
    "Amount Received",
    "Amount Paid",
    "Amount Difference",
    "Amount Ratio",
    "Year",
    "Month",
    "Day",
    "Hour",
    "DayOfWeek",
    "Transaction Time Category",
    "Is Weekend",
    "Log Amount Received",
    "Log Amount Paid",
    "Same Bank Transaction",
    "Same Currency",
    "Temporal Seconds",
]

CATEGORICAL_FEATURES = [
    "Receiving Currency",
    "Payment Currency",
    "Payment Format",
]

# Raw account/bank identifiers are retained for prediction output but are not
# treated as ordered numeric inputs by the first tabular baseline.
IDENTIFIER_COLUMNS = [
    "From Bank",
    "From Account",
    "To Bank",
    "To Account",
]

FORBIDDEN_COLUMNS = {
    "laundering_count",
}


def split_path(split: str, data_dir: Path | str = DEFAULT_DATA_DIR) -> Path:
    """Return the expected modeling-split path."""
    if split not in {"train", "validation", "test", "future"}:
        raise ValueError(f"Unknown split: {split}")
    return Path(data_dir) / f"{split}.csv"


def validate_feature_contract(columns: Iterable[str]) -> None:
    """Fail fast if a dataset violates the frozen modeling contract."""
    columns = set(columns)
    required = set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES) | {TARGET_COLUMN}
    missing = sorted(required - columns)
    forbidden = sorted(FORBIDDEN_COLUMNS & columns)

    if missing:
        raise ValueError(f"Missing required modeling columns: {missing}")
    if forbidden:
        raise ValueError(f"Forbidden target-derived columns detected: {forbidden}")


def load_split(
    split: str,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load one cleaned modeling split and validate its schema."""
    path = split_path(split, data_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Modeling split not found: {path}. Run the data pipeline first."
        )

    df = pd.read_csv(path, usecols=columns)
    validate_feature_contract(df.columns)

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(f"{split}: target contains missing values")

    labels = set(df[TARGET_COLUMN].dropna().unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"{split}: unexpected target values: {sorted(labels)}")

    return df


def feature_columns() -> list[str]:
    """Return tabular feature columns in deterministic order."""
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES
