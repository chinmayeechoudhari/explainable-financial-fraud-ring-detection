"""Train-only feature scaling for temporal neural models."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.temporal.tgat_dataset import ALL_FEATURES


@dataclass
class TemporalFeatureScaler:
    """Standardize temporal TGAT features using training data only."""

    mean: np.ndarray
    scale: np.ndarray
    clip: float = 10.0

    @classmethod
    def fit(cls, frame: pd.DataFrame, clip: float = 10.0) -> "TemporalFeatureScaler":
        # Use pandas column-wise reductions instead of materializing the full
        # training matrix as float64. This matters for multi-million-row data.
        values = frame[ALL_FEATURES]
        mean = values.mean(axis=0, skipna=True).to_numpy(dtype=np.float64)
        scale = values.std(axis=0, ddof=0, skipna=True).to_numpy(dtype=np.float64)
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        return cls(mean=mean, scale=scale, clip=clip)

    def transform_array(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        transformed = (values - self.mean) / self.scale
        return np.clip(transformed, -self.clip, self.clip).astype(np.float32)

    def transform_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result.loc[:, ALL_FEATURES] = self.transform_array(
            result[ALL_FEATURES].to_numpy(dtype=np.float64)
        )
        return result
