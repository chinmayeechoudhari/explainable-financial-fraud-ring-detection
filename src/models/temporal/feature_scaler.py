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
        values = frame[ALL_FEATURES].to_numpy(dtype=np.float64)
        mean = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
        return cls(mean=mean, scale=scale, clip=clip)

    def transform_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        values = result[ALL_FEATURES].to_numpy(dtype=np.float64)
        values = (values - self.mean) / self.scale
        values = np.clip(values, -self.clip, self.clip)
        result.loc[:, ALL_FEATURES] = values.astype(np.float32)
        return result

    def transform_array(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        transformed = (values - self.mean) / self.scale
        return np.clip(transformed, -self.clip, self.clip).astype(np.float32)
