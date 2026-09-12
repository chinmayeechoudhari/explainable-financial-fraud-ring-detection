"""Train and evaluate the leakage-safe TGAT transaction classifier.

The stream is chronological. For every timestamp group, neighborhoods are
constructed from the state before that timestamp; only after the examples are
created are those transactions inserted into temporal history. Validation and
test continue the history from earlier splits and never update model weights.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from src.models.common.evaluation import classification_metrics
from src.models.temporal.feature_scaler import TemporalFeatureScaler
from src.models.temporal.tgat_core import TGATTransactionModel
from src.models.temporal.tgat_dataset import ALL_FEATURES, CURRENT_FEATURES, TARGET, EventFeatureStore, build_batch


DEFAULT_DATA = Path("data/processed/temporal_gnn")
DEFAULT_RESULTS = Path("results/tgat")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_frame(path: Path, rows: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, nrows=rows)
    frame["_timestamp"] = pd.to_datetime(frame["Timestamp"], errors="raise").astype("int64") // 1_000_000_000
    frame = frame.sort_values("_timestamp", kind="stable").reset_index(drop=True)
    return frame


def make_batches(frame: pd.DataFrame, store: EventFeatureStore, batch_size: int, neighbor_k: int, feature_scaler=None):
    """Yield causal batches and update history only after each timestamp-safe batch."""
    start = 0
    columns = list(frame.columns)
    idx = {name: pos for pos, name in enumerate(columns)}
    feature_indices = tuple(idx[name] for name in ALL_FEATURES)
    source_idx = idx["From Account"]
    destination_idx = idx["To Account"]
    timestamp_idx = idx["_timestamp"]
    while start < len(frame):
        end = min(start + batch_size, len(frame))
        if end < len(frame):
            timestamp = frame.iloc[end - 1]["_timestamp"]
            while end < len(frame) and frame.iloc[end]["_timestamp"] == timestamp:
                end += 1
        chunk = frame.iloc[start:end]
        yield build_batch(chunk, store, neighbor_k, feature_scaler=feature_scaler)
        for row in chunk.itertuples(index=False, name=None):
            store.add_row(row, int(row[timestamp_idx]), feature_indices=(source_idx, destination_idx, *feature_indices))
        start = end


def train_epoch(model, frame, store, optimizer, criterion, batch_size, neighbor_k, negative_ratio, device, feature_scaler):
    model.train()
    total_loss = 0.0
    used = 0
    positives = 0
    negatives = 0
    total_batches = 0
    for batch in make_batches(frame, store, batch_size, neighbor_k, feature_scaler):
        total_batches += 1
        labels = batch.labels
        positive_idx = torch.nonzero(labels == 1, as_tuple=False).flatten()
        negative_idx = torch.nonzero(labels == 0, as_tuple=False).flatten()
        keep_negative = min(len(negative_idx), max(1, len(positive_idx) * negative_ratio))
        if keep_negative < len(negative_idx):
            perm = torch.randperm(len(negative_idx))[:keep_negative]
            negative_idx = negative_idx[perm]
        keep = torch.cat([positive_idx, negative_idx])
        if len(keep) == 0:
            continue

        transaction = batch.transaction_features[keep].to(device)
        sender = batch.sender_features[keep].to(device)
        sender_delta = batch.sender_delta_seconds[keep].to(device)
        receiver = batch.receiver_features[keep].to(device)
        receiver_delta = batch.receiver_delta_seconds[keep].to(device)
        y = labels[keep].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(transaction, sender, sender_delta, receiver, receiver_delta)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += float(loss.item()) * len(keep)
        used += len(keep)
        positives += int((y == 1).sum())
        negatives += int((y == 0).sum())

        if total_batches % 100 == 0:
            print(f"  processed {total_batches} batches | queries={used:,} | pos={positives:,} | neg={negatives:,}", flush=True)
    return total_loss / max(used, 1), positives, negatives, total_batches


def predict(model, frame, store, batch_size, neighbor_k, device, feature_scaler=None):
    model.eval()
    scores = []
    labels = []
    timestamps = []
    with torch.no_grad():
        for batch in make_batches(frame, store, batch_size, neighbor_k, feature_scaler):
            output = model(
                batch.transaction_features.to(device),
                batch.sender_features.to(device),
                batch.sender_delta_seconds.to(device),
                batch.receiver_features.to(device),
                batch.receiver_delta_seconds.to(device),
            )
            logits, _ = output
            scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(batch.labels.numpy().tolist())
            timestamps.extend(batch.timestamps.numpy().tolist())
    return np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float64), np.asarray(timestamps, dtype=np.int64)


def evaluate(labels, scores):
    return classification_metrics(labels, scores, threshold=0.5, k_values=(100, 500, 1000))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--neighbor-k", type=int, default=10)
    parser.add_argument("--negative-ratio", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-validation-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows = 5000 if args.smoke else args.max_train_rows
    val_rows = 2000 if args.smoke else args.max_validation_rows
    test_rows = 2000 if args.smoke else args.max_test_rows

    train = read_frame(args.data_dir / "train.csv", train_rows)
    validation = read_frame(args.data_dir / "validation.csv", val_rows)
    test = read_frame(args.data_dir / "test.csv", test_rows)
    if len(train) == 0 or len(validation) == 0 or len(test) == 0:
        raise RuntimeError("TGAT input split is empty")
    if list(train[ALL_FEATURES].columns) != ALL_FEATURES:
        raise RuntimeError("TGAT feature contract mismatch")
    if not set(train[TARGET].dropna().unique()).issubset({0, 1}):
        raise RuntimeError("Invalid target labels")

    # Fit statistics on training data only. Validation/test are transformed
    # with exactly the same scaler; no future information enters the scaler.
    print("Fitting train-only feature scaler...", flush=True)
    scaler = TemporalFeatureScaler.fit(train)
    train = scaler.transform_frame(train)
    validation = scaler.transform_frame(validation)
    test = scaler.transform_frame(test)
    print("Feature scaling: train-only mean/std, clipped to [-10, 10]", flush=True)

    model = TGATTransactionModel(
        transaction_dim=len(ALL_FEATURES),
        event_dim=len(ALL_FEATURES),
        hidden_dim=32,
        time_dim=16,
        num_heads=2,
        dropout=0.2,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update({
        "device": str(device),
        "feature_count": len(ALL_FEATURES),
        "current_feature_count": len(CURRENT_FEATURES),
        "temporal_feature_count": len(ALL_FEATURES) - len(CURRENT_FEATURES),
        "feature_scaling": "train_only_mean_std_clip_10",
    })
    (args.results_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    best_val = -np.inf
    best_state = None
    log = []
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        store = EventFeatureStore(max_history=args.neighbor_k)
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{args.epochs} - training...", flush=True)
        loss, positives, negatives, batches = train_epoch(
            model, train, store, optimizer, criterion,
            args.batch_size, args.neighbor_k, args.negative_ratio, device, scaler,
        )
        print(f"Epoch {epoch}/{args.epochs} - validating...", flush=True)
        val_labels, val_scores, _ = predict(
            model, validation, store, args.batch_size, args.neighbor_k, device, scaler
        )
        val_pr = average_precision_score(val_labels, val_scores)
        row = {
            "epoch": epoch,
            "loss": loss,
            "train_positive_queries": positives,
            "train_negative_queries": negatives,
            "train_batches": batches,
            "validation_pr_auc": float(val_pr),
            "seconds": time.time() - epoch_start,
        }
        log.append(row)
        print(
            f"Epoch {epoch}/{args.epochs}: loss={loss:.6f}, "
            f"val_pr_auc={val_pr:.6f}, batches={batches}, time={row['seconds']:.1f}s",
            flush=True,
        )
        if val_pr > best_val:
            best_val = val_pr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No TGAT checkpoint was produced")
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), args.results_dir / "model.pt")
    (args.results_dir / "training_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    print("\nRebuilding train history for final validation/test evaluation...", flush=True)
    evaluation_store = EventFeatureStore(max_history=args.neighbor_k)
    _, _, _ = predict(model, train, evaluation_store, args.batch_size, args.neighbor_k, device, scaler)
    print("Evaluating validation...", flush=True)
    val_labels, val_scores, val_ts = predict(
        model, validation, evaluation_store, args.batch_size, args.neighbor_k, device, scaler
    )
    print("Evaluating test...", flush=True)
    test_labels, test_scores, test_ts = predict(
        model, test, evaluation_store, args.batch_size, args.neighbor_k, device, scaler
    )

    metrics = {
        "validation": evaluate(val_labels, val_scores),
        "test": evaluate(test_labels, test_scores),
        "training_seconds": time.time() - start_time,
        "feature_count": len(ALL_FEATURES),
        "neighbor_k": args.neighbor_k,
        "negative_sampling_ratio": args.negative_ratio,
        "device": str(device),
        "feature_scaling": "train_only_mean_std_clip_10",
    }
    (args.results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"timestamp": val_ts, "label": val_labels, "score": val_scores}).to_csv(
        args.results_dir / "validation_predictions.csv", index=False
    )
    pd.DataFrame({"timestamp": test_ts, "label": test_labels, "score": test_scores}).to_csv(
        args.results_dir / "test_predictions.csv", index=False
    )
    print("TGAT TRAINING PIPELINE: COMPLETED")


if __name__ == "__main__":
    main()
