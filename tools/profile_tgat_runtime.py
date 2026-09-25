"""Profile TGAT batch construction, history lookup, and model compute.

This is a diagnostic only. It does not modify model code or experiment results.
Run from the repository root:
    python tools/profile_tgat_runtime.py --rows 20000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from src.models.temporal.feature_scaler import TemporalFeatureScaler
from src.models.temporal.tgat_core import TGATTransactionModel
from src.models.temporal.tgat_dataset import ALL_FEATURES, EventFeatureStore, build_batch
from src.models.temporal.train_tgat import _add_chunk_to_store, _row_layout, read_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--neighbor-k", type=int, default=10)
    args = parser.parse_args()

    path = Path("data/processed/temporal_gnn/train.csv")
    print(f"Reading first {args.rows:,} training rows...")
    frame = read_frame(path, args.rows)

    print("\nRelevant dtypes:")
    for name in ["From Account", "To Account", "Timestamp", "_timestamp", *ALL_FEATURES]:
        if name in frame.columns:
            print(f"  {name}: {frame[name].dtype}")

    print("\nFitting train-only scaler on diagnostic sample...")
    scaler = TemporalFeatureScaler.fit(frame)
    frame = scaler.transform_frame(frame)

    store = EventFeatureStore(max_history=args.neighbor_k)
    source_idx, destination_idx, timestamp_idx, feature_indices = _row_layout(frame, ALL_FEATURES)

    model = TGATTransactionModel(
        transaction_dim=len(ALL_FEATURES),
        event_dim=len(ALL_FEATURES),
        hidden_dim=32,
        time_dim=16,
        num_heads=2,
        dropout=0.2,
        num_layers=2,
        use_time_encoding=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    build_seconds = 0.0
    history_seconds = 0.0
    forward_seconds = 0.0
    backward_seconds = 0.0
    batches = 0
    rows = 0
    positive_queries = 0
    negative_queries = 0

    start = 0
    total_start = time.perf_counter()

    while start < len(frame):
        end = min(start + args.batch_size, len(frame))
        if end < len(frame):
            timestamp = frame.iloc[end - 1]["_timestamp"]
            while end < len(frame) and frame.iloc[end]["_timestamp"] == timestamp:
                end += 1
        chunk = frame.iloc[start:end]

        t0 = time.perf_counter()
        batch = build_batch(
            chunk,
            store,
            args.neighbor_k,
            feature_scaler=None,
            active_features=ALL_FEATURES,
            use_neighbors=True,
        )
        build_seconds += time.perf_counter() - t0

        t0 = time.perf_counter()
        _add_chunk_to_store(
            chunk,
            store,
            source_idx,
            destination_idx,
            timestamp_idx,
            feature_indices,
        )
        history_seconds += time.perf_counter() - t0

        labels = batch.labels
        positive_idx = torch.nonzero(labels == 1, as_tuple=False).flatten()
        negative_idx = torch.nonzero(labels == 0, as_tuple=False).flatten()
        keep_negative = min(len(negative_idx), max(1, len(positive_idx) * 20))
        if keep_negative < len(negative_idx):
            negative_idx = negative_idx[:keep_negative]
        keep = torch.cat([positive_idx, negative_idx])
        if len(keep):
            transaction = batch.transaction_features[keep]
            sender = batch.sender_features[keep]
            sender_delta = batch.sender_delta_seconds[keep]
            receiver = batch.receiver_features[keep]
            receiver_delta = batch.receiver_delta_seconds[keep]
            y = labels[keep]

            optimizer.zero_grad(set_to_none=True)
            t0 = time.perf_counter()
            logits, _ = model(transaction, sender, sender_delta, receiver, receiver_delta)
            forward_seconds += time.perf_counter() - t0

            loss = criterion(logits, y)
            t0 = time.perf_counter()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            backward_seconds += time.perf_counter() - t0

            positive_queries += int((y == 1).sum())
            negative_queries += int((y == 0).sum())

        batches += 1
        rows += len(chunk)
        start = end

    total_seconds = time.perf_counter() - total_start

    print("\n=== TGAT RUNTIME PROFILE ===")
    print(f"rows:               {rows:,}")
    print(f"batches:            {batches:,}")
    print(f"queries trained:    {positive_queries + negative_queries:,}")
    print(f"positive queries:   {positive_queries:,}")
    print(f"negative queries:   {negative_queries:,}")
    print(f"batch construction: {build_seconds:.2f}s")
    print(f"history insertion:  {history_seconds:.2f}s")
    print(f"forward pass:       {forward_seconds:.2f}s")
    print(f"backward+optimizer: {backward_seconds:.2f}s")
    print(f"total:              {total_seconds:.2f}s")
    print(f"rows/sec:           {rows / max(total_seconds, 1e-9):.2f}")
    print(f"batches/sec:        {batches / max(total_seconds, 1e-9):.2f}")

    print("\n=== RELATIVE COST ===")
    parts = {
        "batch construction": build_seconds,
        "history insertion": history_seconds,
        "forward pass": forward_seconds,
        "backward+optimizer": backward_seconds,
    }
    for name, seconds in parts.items():
        print(f"{name:20s}: {100.0 * seconds / max(total_seconds, 1e-9):6.2f}%")
    print("\nDiagnostic completed; no model/result files were written.")


if __name__ == "__main__":
    main()
