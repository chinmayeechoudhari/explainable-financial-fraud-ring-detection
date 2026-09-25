"""Frozen TGAT evaluation on the later FUTURE split.

No retraining or future-label tuning occurs. History is advanced chronologically
through train -> validation -> test before future transactions are scored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.common.evaluation import classification_metrics
from src.models.temporal.feature_scaler import TemporalFeatureScaler
from src.models.temporal.tgat_core import TGATTransactionModel
from src.models.temporal.tgat_dataset import ALL_FEATURES, CURRENT_FEATURES, EventFeatureStore
from src.models.temporal.train_tgat import populate_history, predict, read_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/temporal_gnn"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/tgat/model.pt"))
    parser.add_argument("--threshold-file", type=Path, default=Path("results/threshold_analysis/frozen_threshold.json"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/generalization"))
    parser.add_argument("--neighbor-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = read_frame(args.data_dir / "train.csv")
    validation = read_frame(args.data_dir / "validation.csv")
    test = read_frame(args.data_dir / "test.csv")
    future = read_frame(args.data_dir / "future.csv")

    shift_columns = [
        "Amount Received", "Amount Paid", "Amount Difference", "Amount Ratio",
        "Same Bank Transaction", "Cross Bank Transaction", "Is Weekend",
    ]
    train_shift_stats = train[shift_columns].agg(["mean", "std"])
    future_shift_stats = future[shift_columns].agg(["mean", "std"])

    scaler = TemporalFeatureScaler.fit(train)
    train = scaler.transform_frame(train)
    validation = scaler.transform_frame(validation)
    test = scaler.transform_frame(test)
    future = scaler.transform_frame(future)

    model = TGATTransactionModel(
        transaction_dim=len(ALL_FEATURES), event_dim=len(ALL_FEATURES),
        hidden_dim=32, time_dim=16, num_heads=2, dropout=0.2,
        num_layers=2, use_time_encoding=True,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    store = EventFeatureStore(max_history=args.neighbor_k)
    populate_history(train, store, args.batch_size, list(ALL_FEATURES))
    # Advance state through earlier evaluation periods without using their labels
    # for any model or threshold decision.
    _, _, _ = predict(model, validation, store, args.batch_size, args.neighbor_k, device, list(ALL_FEATURES), True)
    _, _, _ = predict(model, test, store, args.batch_size, args.neighbor_k, device, list(ALL_FEATURES), True)
    future_labels, future_scores, future_ts = predict(
        model, future, store, args.batch_size, args.neighbor_k, device, list(ALL_FEATURES), True
    )

    frozen = json.loads(args.threshold_file.read_text(encoding="utf-8"))
    threshold = float(frozen["threshold"])
    metrics = classification_metrics(
        future_labels, future_scores, threshold=threshold,
        k_values=(50, 100, 250, 500, 1000),
    )
    metrics["rows"] = int(len(future_labels))
    metrics["positives"] = int(future_labels.sum())
    metrics["prevalence"] = float(future_labels.mean())
    metrics["threshold_source"] = str(args.threshold_file)
    metrics["model_source"] = str(args.checkpoint)

    shift_rows = []
    for column in shift_columns:
        train_mean = float(train_shift_stats.loc["mean", column])
        train_std = float(train_shift_stats.loc["std", column])
        future_mean = float(future_shift_stats.loc["mean", column])
        future_std = float(future_shift_stats.loc["std", column])
        pooled_scale = max(abs(train_std), 1e-12)
        shift_rows.append({
            "feature": column,
            "train_mean": train_mean,
            "future_mean": future_mean,
            "train_std": train_std,
            "future_std": future_std,
            "mean_shift_in_train_sd": (future_mean - train_mean) / pooled_scale,
        })
    shift_table = pd.DataFrame(shift_rows)

    out = args.results_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "timestamp": future_ts,
        "label": future_labels,
        "score": future_scores,
        "prediction": (future_scores >= threshold).astype(np.int8),
    }).to_csv(out / "future_predictions.csv", index=False)
    shift_table.to_csv(out / "distribution_shift.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "config.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "threshold_file": str(args.threshold_file),
        "threshold": threshold,
        "scaler": "fit on train only",
        "history_order": ["train", "validation", "test", "future"],
        "future_rows": len(future),
        "future_positives": int(future_labels.sum()),
        "future_prevalence": float(future_labels.mean()),
        "device": str(device),
    }, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "Frozen-model future evaluation. No retraining and no threshold tuning on future labels. "
        "The future split is small (n=223), so metrics are reported as a stress test with uncertainty.\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
