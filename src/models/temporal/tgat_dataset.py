"""Build compact chronological TGAT batches from temporal GNN datasets.

This module converts each transaction row into a bounded temporal neighborhood.
Historical events are selected strictly before the query timestamp. For a full
run, use the precomputed temporal_gnn train/validation/test/future files.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .temporal_neighbor_store import TemporalNeighborStore


CURRENT_FEATURES = [
    "Amount Received", "Amount Paid", "Amount Difference", "Amount Ratio",
    "Same Bank Transaction", "Cross Bank Transaction", "Transaction Time Category",
    "Is Weekend", "Log Amount Received", "Log Amount Paid", "Same Currency",
]
TEMPORAL_FEATURES = [
    "sender_in_count", "sender_out_count", "sender_total_count", "sender_in_amount",
    "sender_out_amount", "sender_avg_in_amount", "sender_avg_out_amount", "sender_time_since_last",
    "receiver_in_count", "receiver_out_count", "receiver_total_count", "receiver_in_amount",
    "receiver_out_amount", "receiver_avg_in_amount", "receiver_avg_out_amount", "receiver_time_since_last",
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
        self.id_to_features: dict[int, np.ndarray] = {}

    def get_id(self, account: str) -> int:
        if account not in self.account_to_id:
            self.account_to_id[account] = len(self.account_to_id)
        return self.account_to_id[account]

    def add_row(self, row: pd.Series, timestamp: int) -> None:
        source = self.get_id(str(row["From Account"]))
        destination = self.get_id(str(row["To Account"]))
        features = row[ALL_FEATURES].to_numpy(dtype=np.float32)
        self.id_to_features[source] = features
        self.id_to_features[destination] = features
        self.store.add_event(timestamp, source, destination, features)

    def history(self, node_id: int, timestamp: int, limit: int) -> list[tuple[int, np.ndarray]]:
        events = self.store.recent_events(node_id, timestamp, limit)
        return [(event.timestamp, np.asarray(event.features, dtype=np.float32)) for event in events]


def _pad_history(events: Iterable[tuple[int, np.ndarray]], k: int, feature_dim: int, query_timestamp: int) -> tuple[np.ndarray, np.ndarray]:
    feature_array = np.zeros((k, feature_dim), dtype=np.float32)
    delta_array = np.zeros(k, dtype=np.float32)
    for index, (timestamp, features) in enumerate(events):
        if index >= k:
            break
        feature_array[index] = features
        delta = int(query_timestamp) - int(timestamp)
        if delta <= 0:
            raise ValueError("Temporal sampler returned a non-historical event")
        delta_array[index] = float(delta)
    return feature_array, delta_array


def build_batch(
    frame: pd.DataFrame,
    store: EventFeatureStore,
    neighbor_k: int = 10,
) -> TemporalBatch:
    transaction_features = []
    sender_features = []
    sender_delta = []
    receiver_features = []
    receiver_delta = []
    labels = []
    timestamps = []

    for _, row in frame.iterrows():
        timestamp = int(row["_timestamp"])
        sender = store.get_id(str(row["From Account"]))
        receiver = store.get_id(str(row["To Account"]))

        sender_history = store.history(sender, timestamp, neighbor_k)
        receiver_history = store.history(receiver, timestamp, neighbor_k)

        sender_events, sender_times = _pad_history(
            sender_history, neighbor_k, len(ALL_FEATURES), timestamp
        )
        receiver_events, receiver_times = _pad_history(
            receiver_history, neighbor_k, len(ALL_FEATURES), timestamp
        )

        transaction_features.append(row[CURRENT_FEATURES + TEMPORAL_FEATURES].to_numpy(dtype=np.float32))
        sender_features.append(np.vstack([np.zeros(len(ALL_FEATURES), dtype=np.float32), sender_events]))
        receiver_features.append(np.vstack([np.zeros(len(ALL_FEATURES), dtype=np.float32), receiver_events]))
        sender_delta.append(sender_times)
        receiver_delta.append(receiver_times)
        labels.append(int(row[TARGET]))
        timestamps.append(timestamp)

    return TemporalBatch(
        transaction_features=torch.tensor(np.stack(transaction_features), dtype=torch.float32),
        sender_features=torch.tensor(np.stack(sender_features), dtype=torch.float32),
        sender_delta_seconds=torch.tensor(np.stack(sender_delta), dtype=torch.float32),
        receiver_features=torch.tensor(np.stack(receiver_features), dtype=torch.float32),
        receiver_delta_seconds=torch.tensor(np.stack(receiver_delta), dtype=torch.float32),
        labels=torch.tensor(labels, dtype=torch.float32),
        timestamps=torch.tensor(timestamps, dtype=torch.long),
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
        return {
            "transaction": row[ALL_FEATURES].to_numpy(dtype=np.float32),
            "timestamp": int(row["_timestamp"]),
            "label": int(row[TARGET]),
        }
