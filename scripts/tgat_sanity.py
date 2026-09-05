"""Controlled TGAT learning sanity experiment.

Run from the repository root with ``python -m scripts.tgat_sanity``. This is a
sanity check, not a benchmark; official datasets and the main trainer remain
unchanged.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.models.temporal.tgat_core import TGATTransactionModel
from src.models.temporal.tgat_dataset import ALL_FEATURES, EventFeatureStore, build_batch

DATA_DIR = Path("data/processed/temporal_gnn")


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def read_frame(path: Path, max_rows: int) -> pd.DataFrame:
    frame = pd.read_csv(path, nrows=max_rows)
    frame["_timestamp"] = pd.to_datetime(frame["Timestamp"], errors="raise").astype("int64") // 1_000_000_000
    return frame.sort_values("_timestamp", kind="stable").reset_index(drop=True)


def select_queries(frame: pd.DataFrame, positives: int, negatives: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pos, neg = frame[frame["Is Laundering"] == 1], frame[frame["Is Laundering"] == 0]
    if len(pos) < positives or len(neg) < negatives:
        raise RuntimeError(f"Not enough rows: positives={len(pos)}, negatives={len(neg)}")
    selected = pd.concat([
        pos.iloc[rng.choice(len(pos), positives, replace=False)],
        neg.iloc[rng.choice(len(neg), negatives, replace=False)],
    ], ignore_index=True)
    return selected.sort_values("_timestamp", kind="stable").reset_index(drop=True)


def make_batches(frame, store, batch_size, neighbor_k):
    start = 0
    while start < len(frame):
        end = min(start + batch_size, len(frame))
        if end < len(frame):
            ts = frame.iloc[end - 1]["_timestamp"]
            while end < len(frame) and frame.iloc[end]["_timestamp"] == ts:
                end += 1
        chunk = frame.iloc[start:end].copy()
        yield build_batch(chunk, store, neighbor_k)
        for _, row in chunk.iterrows():
            store.add_row(row, int(row["_timestamp"]))
        start = end


def run_epoch(model, frame, store, optimizer, criterion, batch_size, neighbor_k, device):
    model.train(); losses = []
    for batch in make_batches(frame, store, batch_size, neighbor_k):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(batch.transaction_features.to(device), batch.sender_features.to(device),
                           batch.sender_delta_seconds.to(device), batch.receiver_features.to(device),
                           batch.receiver_delta_seconds.to(device))
        loss = criterion(logits, batch.labels.to(device)); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def predict(model, frame, store, batch_size, neighbor_k, device):
    model.eval(); scores, labels = [], []
    with torch.no_grad():
        for batch in make_batches(frame, store, batch_size, neighbor_k):
            logits, _ = model(batch.transaction_features.to(device), batch.sender_features.to(device),
                               batch.sender_delta_seconds.to(device), batch.receiver_features.to(device),
                               batch.receiver_delta_seconds.to(device))
            scores.extend(torch.sigmoid(logits).cpu().numpy()); labels.extend(batch.labels.numpy())
    return np.asarray(labels), np.asarray(scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--source-rows", type=int, default=2_000_000)
    parser.add_argument("--train-positives", type=int, default=100)
    parser.add_argument("--train-negatives", type=int, default=2000)
    parser.add_argument("--eval-positives", type=int, default=100)
    parser.add_argument("--eval-negatives", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--neighbor-k", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_source = read_frame(args.data_dir / "train.csv", args.source_rows)
    val_source = read_frame(args.data_dir / "validation.csv", args.source_rows)
    train = select_queries(train_source, args.train_positives, args.train_negatives, args.seed)
    validation = select_queries(val_source, args.eval_positives, args.eval_negatives, args.seed + 1)
    if list(train[ALL_FEATURES].columns) != ALL_FEATURES: raise RuntimeError("TGAT feature contract mismatch")

    model = TGATTransactionModel(transaction_dim=len(ALL_FEATURES), event_dim=len(ALL_FEATURES),
                                  hidden_dim=32, time_dim=16, num_heads=2, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    print(f"Device: {device}")
    print(f"Source rows: train={len(train_source)}, validation={len(val_source)}")
    print(f"Train queries: {len(train)} ({int(train['Is Laundering'].sum())} positive, {int((train['Is Laundering']==0).sum())} negative)")
    print(f"Validation queries: {len(validation)} ({int(validation['Is Laundering'].sum())} positive, {int((validation['Is Laundering']==0).sum())} negative)")

    best_pr, best_roc, best_state = -np.inf, -np.inf, None; start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        store = EventFeatureStore(max_history=args.neighbor_k)
        loss = run_epoch(model, train, store, optimizer, criterion, args.batch_size, args.neighbor_k, device)
        labels, scores = predict(model, validation, store, args.batch_size, args.neighbor_k, device)
        pr, roc = average_precision_score(labels, scores), roc_auc_score(labels, scores)
        print(f"Epoch {epoch}/{args.epochs}: loss={loss:.6f}, val_pr_auc={pr:.6f}, val_roc_auc={roc:.6f}")
        if pr > best_pr:
            best_pr, best_roc = pr, roc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None: raise RuntimeError("No sanity checkpoint produced")
    model.load_state_dict(best_state)
    eval_store = EventFeatureStore(max_history=args.neighbor_k)
    # Reconstruct the same chronological training history before validation.
    _, _ = predict(model, train, eval_store, args.batch_size, args.neighbor_k, device)
    labels, scores = predict(model, validation, eval_store, args.batch_size, args.neighbor_k, device)
    pos_scores, neg_scores = scores[labels == 1], scores[labels == 0]
    print(f"Best validation PR-AUC: {average_precision_score(labels, scores):.6f}")
    print(f"Best validation ROC-AUC: {roc_auc_score(labels, scores):.6f}")
    print(f"Positive score mean: {pos_scores.mean():.6f}")
    print(f"Negative score mean: {neg_scores.mean():.6f}")
    print(f"Sanity runtime: {time.time() - start_time:.1f}s")
    print("TGAT LEARNING SANITY TEST: COMPLETED")


if __name__ == "__main__": main()
