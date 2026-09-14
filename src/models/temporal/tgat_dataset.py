"""Build compact chronological TGAT batches from temporal GNN datasets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .temporal_neighbor_store import TemporalEvent, TemporalNeighborStore

CURRENT_FEATURES = [
    "Amount Received", "Amount Paid", "Amount Difference", "Amount Ratio",
    "Same Bank Transaction", "Cross Bank Transaction", "Transaction Time Category",
    "Is Weekend", "Log Amount Received", "Log Amount Paid", "Same Currency",
]
TEMPORAL_FEATURES = [
    "sender_in_count", "sender_out_count", "sender_total_count", "sender_in_amount",
    "sender_out_amount", "sender_avg_in_amount", "sender_avg_out_amount", "sender_time_since_last",
    "receiver_in_count", "receiver_out_count", "receiver_total_count", "receiver_in_amount",
    "receiver_out_amount", "receiver_avg_in_amount", "receiver_time_since_last",
]
ALL_FEATURES = CURRENT_FEATURES + TEMPORAL_FEATURES
TARGET = "Is Laundering"


@dataclass
class TemporalBatch:
    transaction_features: torch.Tensor
    sender_features: torch.Tensor
    sender_delta_seconds: torch.Tensor
    receiver_features: torch.Tensor
    receiver_delta_seconds: torch.Tensor
    labels: torch.Tensor
    timestamps: torch.Tensor


class EventFeatureStore:
    """Stores compact event feature vectors for temporal neighbor lookup."""

    def __init__(self, max_history: int = 64) -> None:
        self.store = TemporalNeighborStore(max_history_per_node=max_history)
        self.account_to_id: dict[str, int] = {}

    def get_id(self, account: str) -> int:
        if account not in self.account_to_id:
            self.account_to_id[account] = len(self.account_to_id)
        return self.account_to_id[account]

    def add_row(self, row, timestamp: int, feature_indices: tuple[int, ...]) -> None:
        """Insert one itertuples row without constructing pandas objects."""
        source = self.get_id(str(row[feature_indices[0]]))
        destination = self.get_id(str(row[feature_indices[1]]))
        self.store.add_event(
            timestamp,
            source,
            destination,
            (row[index] for index in feature_indices[2:]),
        )

    def history(self, node_id: int, timestamp: int, limit: int) -> list[TemporalEvent]:
        """Return recent historical events."""
        return self.store.recent_events(node_id, timestamp, limit)


def _pad_history(
    events: Iterable[TemporalEvent],
    feature_array: np.ndarray,
    delta_array: np.ndarray,
    query_timestamp: int,
) -> None:
    """Write a historical neighborhood directly into preallocated arrays."""
    for index, event in enumerate(events):
        if index >= len(delta_array):
            break
        feature_array[index] = event.features
        delta = int(query_timestamp) - int(event.timestamp)
        if delta <= 0:
            raise ValueError("Temporal sampler returned a non-historical event")
        delta_array[index] = float(delta)


def build_batch(frame: pd.DataFrame, store: EventFeatureStore, neighbor_k: int = 10, feature_scaler=None) -> TemporalBatch:
    """Build a causal batch with preallocated NumPy output buffers."""
    columns = list(frame.columns)
    idx = {name: pos for pos, name in enumerate(columns)}
    sender_index = idx["From Account"]
    receiver_index = idx["To Account"]
    timestamp_index = idx["_timestamp"]
    row_count = len(frame)
    feature_dim = len(ALL_FEATURES)

    if feature_scaler is None:
        transaction_matrix = frame[ALL_FEATURES].to_numpy(dtype=np.float32, copy=False)
    else:
        transaction_matrix = feature_scaler.transform_array(
            frame[ALL_FEATURES].to_numpy(dtype=np.float64)
        )

    sender_features = np.zeros((row_count, neighbor_k + 1, feature_dim), dtype=np.float32)
    receiver_features = np.zeros((row_count, neighbor_k + 1, feature_dim), dtype=np.float32)
    sender_delta = np.zeros((row_count, neighbor_k), dtype=np.float32)
    receiver_delta = np.zeros((row_count, neighbor_k), dtype=np.float32)
    labels = frame[TARGET].to_numpy(dtype=np.float32, copy=False)
    timestamps = frame["_timestamp"].to_numpy(dtype=np.int64, copy=False)

    for row_index, row in enumerate(frame.itertuples(index=False, name=None)):
        timestamp = int(row[timestamp_index])
        sender = store.get_id(str(row[sender_index]))
        receiver = store.get_id(str(row[receiver_index]))
        _pad_history(
            store.history(sender, timestamp, neighbor_k),
            sender_features[row_index, 1:],
            sender_delta[row_index],
            timestamp,
        )
        _pad_history(
            store.history(receiver, timestamp, neighbor_k),
            receiver_features[row_index, 1:],
            receiver_delta[row_index],
            timestamp,
        )

    return TemporalBatch(
        transaction_features=torch.from_numpy(transaction_matrix),
        sender_features=torch.from_numpy(sender_features),
        sender_delta_seconds=torch.from_numpy(sender_delta),
        receiver_features=torch.from_numpy(receiver_features),
        receiver_delta_seconds=torch.from_numpy(receiver_delta),
        labels=torch.from_numpy(labels),
        timestamps=torch.from_numpy(timestamps),
    )


class TGATSmokeDataset(Dataset):
    """Small deterministic dataset wrapper used by tests."""
    def __init__(self, frame: pd.DataFrame, neighbor_k: int = 10) -> None:
        self.frame = frame.reset_index(drop=True)
        self.neighbor_k = neighbor_k
    def __len__(self) -> int:
        return len(self.frame)
    def __getitem__(self, index: int) -> dict[str, np.ndarray | int]:
        row = self.frame.iloc[index]
        return {"transaction": row[ALL_FEATURES].to_numpy(dtype=np.float32), "timestamp": int(row["_timestamp"]), "label": int(row[TARGET])}
