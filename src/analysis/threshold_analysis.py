"""Validation-only threshold selection for the frozen TGAT checkpoint.

Reads prediction artifacts only. Test labels are never inspected until after the
validation threshold is frozen.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score

from src.models.common.evaluation import classification_metrics, precision_recall_at_k


def threshold_grid(scores: np.ndarray, points: int = 1000) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, points)
    grid = np.unique(np.quantile(scores, quantiles))
    return np.concatenate(([0.0], grid, [1.0]))


def sweep(labels: np.ndarray, scores: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        m = classification_metrics(labels, scores, threshold=float(threshold), k_values=())
        rows.append({
            "threshold": float(threshold),
            "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
            "false_positive_rate": m["false_positive_rate"],
            "tp": m["true_positives"], "fp": m["false_positives"],
            "tn": m["true_negatives"], "fn": m["false_negatives"],
        })
    return pd.DataFrame(rows)


def ranking(labels: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in (50, 100, 250, 500, 1000, 2500, 5000):
        p, r = precision_recall_at_k(labels, scores, k)
        rows.append({"k": k, "precision_at_k": p, "recall_at_k": r})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path("results/tgat/validation_predictions.csv"))
    parser.add_argument("--test-predictions", type=Path, default=Path("results/tgat/test_predictions.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/threshold_analysis"))
    parser.add_argument("--criterion", choices=["max_f1"], default="max_f1")
    args = parser.parse_args()

    out = args.results_dir
    (out / "tgat" / "plots").mkdir(parents=True, exist_ok=True)

    val = pd.read_csv(args.predictions)
    if len(val) != 891571 or int(val["label"].sum()) != 367:
        raise RuntimeError(f"Validation artifact mismatch: rows={len(val)}, positives={int(val['label'].sum())}")

    y = val["label"].to_numpy(dtype=np.int8)
    s = val["score"].to_numpy(dtype=float)
    if not np.isfinite(s).all():
        raise RuntimeError("Validation scores contain NaN/Inf")

    thresholds = threshold_grid(s)
    table = sweep(y, s, thresholds)
    selected = table.loc[table["f1"].idxmax()]
    threshold = float(selected["threshold"])

    frozen = {
        "threshold": threshold,
        "criterion": args.criterion,
        "selected_on": "validation",
        "validation_metrics": selected.to_dict(),
        "date": date.today().isoformat(),
    }
    (out / "frozen_threshold.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    table.to_csv(out / "tgat" / "validation_sweep.csv", index=False)
    ranking(y, s).to_csv(out / "tgat" / "ranking_metrics_validation.csv", index=False)

    # Test is read only after the threshold has been frozen and persisted.
    test = pd.read_csv(args.test_predictions)
    yt = test["label"].to_numpy(dtype=np.int8)
    st = test["score"].to_numpy(dtype=float)
    test_metrics = classification_metrics(yt, st, threshold=threshold, k_values=(50,100,250,500,1000,2500,5000))
    (out / "tgat" / "test_at_frozen_threshold.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    ranking(yt, st).to_csv(out / "tgat" / "ranking_metrics_test.csv", index=False)

    precision, recall, _ = precision_recall_curve(y, s)
    fpr, tpr, _ = roc_curve(y, s)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("TGAT Validation Precision-Recall")
    plt.savefig(out / "tgat" / "plots" / "pr_curve.png", bbox_inches="tight"); plt.close()

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("TGAT Validation ROC")
    plt.savefig(out / "tgat" / "plots" / "roc_curve.png", bbox_inches="tight"); plt.close()

    plt.figure()
    plt.plot(table["threshold"], table["precision"], label="precision")
    plt.plot(table["threshold"], table["recall"], label="recall")
    plt.plot(table["threshold"], table["f1"], label="F1")
    plt.axvline(threshold, linestyle="--", label="frozen threshold")
    plt.xlabel("Threshold"); plt.ylabel("Metric"); plt.legend()
    plt.savefig(out / "tgat" / "plots" / "threshold_vs_metrics.png", bbox_inches="tight"); plt.close()

    plt.figure()
    plt.hist(s[y == 0], bins=80, alpha=0.7, label="legitimate")
    plt.hist(s[y == 1], bins=80, alpha=0.7, label="laundering")
    plt.yscale("log"); plt.xlabel("Score"); plt.ylabel("Count")
    plt.legend()
    plt.savefig(out / "tgat" / "plots" / "score_distribution.png", bbox_inches="tight"); plt.close()

    rk = ranking(y, s)
    plt.figure()
    plt.plot(rk["k"], rk["precision_at_k"], marker="o")
    plt.xlabel("K"); plt.ylabel("Precision@K"); plt.title("TGAT Validation Precision@K")
    plt.savefig(out / "tgat" / "plots" / "precision_at_k.png", bbox_inches="tight"); plt.close()

    config = {
        "criterion": args.criterion,
        "selection_split": "validation",
        "test_evaluation": "single evaluation after threshold freeze",
        "validation_rows": len(val),
        "validation_positives": int(y.sum()),
        "validation_pr_auc": float(average_precision_score(y, s)),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps({
        "validation_selected_threshold": threshold,
        "validation_metrics": selected.to_dict(),
        "test_at_frozen_threshold": test_metrics,
    }, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "Threshold selected on validation only using max F1. The test set is evaluated once after the threshold is frozen.\n",
        encoding="utf-8",
    )
    print(json.dumps({"threshold": threshold, "validation": selected.to_dict(), "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
