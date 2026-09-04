"""Transaction-level XGBoost baseline.

This baseline uses only the frozen A->B transaction feature contract. It is
intentionally tabular: no account graph structure, temporal neighborhoods,
or target-derived graph statistics are used.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.common.data_loader import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    feature_columns,
    load_split,
)
from src.models.common.evaluation import classification_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "baselines" / "xgboost.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "baselines" / "xgboost"
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "modeling_splits"


# Float32 keeps the full training matrix substantially smaller while retaining
# enough precision for this tabular baseline. Category codes are kept as the
# integer-coded features produced by Member A; they are NOT treated as ordered
# semantic values in the project interpretation.
FEATURE_DTYPES = {column: "float32" for column in NUMERIC_FEATURES}
FEATURE_DTYPES.update({column: "float32" for column in CATEGORICAL_FEATURES})
FEATURE_DTYPES[TARGET_COLUMN] = "int8"

IDENTIFIER_COLUMNS = [
    "Timestamp",
    "From Bank",
    "From Account",
    "To Bank",
    "To Account",
]


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def train_frame(train_path: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Load TRAIN features and compute exact TRAIN-only class balance."""
    columns = feature_columns() + [TARGET_COLUMN]
    frame = pd.read_csv(train_path, usecols=columns, dtype=FEATURE_DTYPES)

    if frame.empty:
        raise ValueError("Training split is empty")
    if frame[columns].isna().any().any():
        raise ValueError("Training features contain missing values")

    X = frame[feature_columns()].to_numpy(dtype=np.float32, copy=False)
    y = frame[TARGET_COLUMN].to_numpy(dtype=np.int8, copy=False)
    if not np.isfinite(X).all():
        raise ValueError("Training features contain NaN or infinite values")

    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError(f"Training split must contain both classes: {positives=}, {negatives=}")

    scale_pos_weight = negatives / positives
    info = {
        "train_rows": int(len(y)),
        "train_positive_count": positives,
        "train_negative_count": negatives,
        "scale_pos_weight": float(scale_pos_weight),
        "feature_count": len(feature_columns()),
    }
    return frame, y, info


def build_classifier(config: dict, scale_pos_weight: float) -> xgb.XGBClassifier:
    """Create a fixed, reproducible CPU histogram XGBoost baseline."""
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        max_bin=int(config["max_bin"]),
        n_estimators=int(config["n_estimators"]),
        learning_rate=float(config["learning_rate"]),
        max_depth=int(config["max_depth"]),
        min_child_weight=float(config["min_child_weight"]),
        subsample=float(config["subsample"]),
        colsample_bytree=float(config["colsample_bytree"]),
        reg_alpha=float(config["reg_alpha"]),
        reg_lambda=float(config["reg_lambda"]),
        gamma=float(config["gamma"]),
        scale_pos_weight=float(scale_pos_weight),
        max_delta_step=float(config["max_delta_step"]),
        n_jobs=int(config["n_jobs"]),
        random_state=int(config["random_state"]),
        verbosity=1,
    )


def evaluate_split(
    split: str,
    classifier: xgb.XGBClassifier,
    config: dict,
) -> tuple[dict, pd.DataFrame]:
    path = DATA_DIR / f"{split}.csv"
    read_columns = IDENTIFIER_COLUMNS + feature_columns() + [TARGET_COLUMN]
    output_parts: list[pd.DataFrame] = []
    y_all: list[np.ndarray] = []
    score_all: list[np.ndarray] = []

    for chunk in pd.read_csv(
        path,
        usecols=read_columns,
        dtype={**FEATURE_DTYPES, **{column: "string" for column in IDENTIFIER_COLUMNS}},
        chunksize=int(config["chunk_size"]),
    ):
        y = chunk[TARGET_COLUMN].to_numpy(dtype=np.int8, copy=False)
        X = chunk[feature_columns()].to_numpy(dtype=np.float32, copy=False)
        if not np.isfinite(X).all():
            raise ValueError(f"{split}: features contain NaN or infinite values")

        scores = classifier.predict_proba(X)[:, 1].astype(np.float32)
        predictions = (scores >= float(config["threshold"])).astype(np.int8)
        output_parts.append(
            pd.DataFrame(
                {
                    **{column: chunk[column].values for column in IDENTIFIER_COLUMNS},
                    TARGET_COLUMN: y,
                    "score": scores,
                    "prediction": predictions,
                }
            )
        )
        y_all.append(y)
        score_all.append(scores)

    y_true = np.concatenate(y_all)
    scores = np.concatenate(score_all)
    metrics = classification_metrics(
        y_true,
        scores,
        threshold=float(config["threshold"]),
        k_values=tuple(config["k_values"]),
    )
    return metrics, pd.concat(output_parts, ignore_index=True)


def save_feature_importance(classifier: xgb.XGBClassifier) -> None:
    importance = pd.DataFrame(
        {
            "feature": feature_columns(),
            "gain": classifier.feature_importances_,
        }
    ).sort_values("gain", ascending=False)
    importance.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Fit a tiny XGBoost model on a small TRAIN sample; do not run the full baseline.",
    )
    args = parser.parse_args()

    config = load_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATA_DIR / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing training split: {train_path}")

    # Header/schema validation still goes through the shared contract.
    load_split("train", columns=[TARGET_COLUMN])

    print("Loading TRAIN features...")
    frame, y, training_info = train_frame(train_path)
    print(f"Training rows: {len(frame):,}")
    print(f"Training positives: {training_info['train_positive_count']:,}")
    print(f"Training negatives: {training_info['train_negative_count']:,}")
    print(f"Scale_pos_weight: {training_info['scale_pos_weight']:.6f}")
    print(f"Feature count: {training_info['feature_count']}")

    if args.smoke_test:
        sample_size = min(int(config["smoke_rows"]), len(frame))
        X_sample = frame[feature_columns()].iloc[:sample_size].to_numpy(dtype=np.float32, copy=True)
        y_sample = y[:sample_size]
        # Guarantee both classes for a deterministic smoke test by expanding
        # until both labels are represented.
        if len(np.unique(y_sample)) < 2:
            positive_index = np.flatnonzero(y == 1)
            if len(positive_index) == 0:
                raise ValueError("No positive class available for smoke test")
            end = max(sample_size, int(positive_index[0]) + 1)
            X_sample = frame[feature_columns()].iloc[:end].to_numpy(dtype=np.float32, copy=True)
            y_sample = y[:end]

        smoke_config = dict(config)
        smoke_config.update({"n_estimators": 5, "n_jobs": 1})
        positives = int((y_sample == 1).sum())
        negatives = int((y_sample == 0).sum())
        smoke_weight = negatives / positives if positives else 1.0
        classifier = build_classifier(smoke_config, smoke_weight)
        classifier.fit(X_sample, y_sample)
        scores = classifier.predict_proba(X_sample)[:, 1]
        print(f"Smoke-test rows: {len(y_sample):,}")
        print(f"Training matrix shape: {X_sample.shape}")
        print(f"Prediction range: [{scores.min():.6f}, {scores.max():.6f}]")
        print("SMOKE TEST: PASSED")
        return

    X = frame[feature_columns()].to_numpy(dtype=np.float32, copy=True)
    del frame

    classifier = build_classifier(config, training_info["scale_pos_weight"])
    print("Training XGBoost...")
    start = time.perf_counter()
    classifier.fit(X, y, verbose=False)
    training_seconds = time.perf_counter() - start
    del X, y
    print(f"Training completed in {training_seconds:.1f}s")

    metrics: dict[str, dict] = {}
    predictions: dict[str, pd.DataFrame] = {}
    for split in ("validation", "test"):
        print(f"Evaluating {split.upper()}...")
        split_metrics, split_predictions = evaluate_split(split, classifier, config)
        metrics[split] = split_metrics
        predictions[split] = split_predictions
        print(json.dumps(split_metrics, indent=2))

    model_path = RESULTS_DIR / "model.json"
    classifier.save_model(model_path)
    save_feature_importance(classifier)

    for split, result in predictions.items():
        result.to_csv(RESULTS_DIR / f"predictions_{split}.csv", index=False)

    with (RESULTS_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": "xgboost_tabular_baseline",
                "xgboost_version": xgb.__version__,
                "training": {**training_info, "training_seconds": training_seconds},
                "metrics": metrics,
            },
            handle,
            indent=2,
            allow_nan=True,
        )
    with (RESULTS_DIR / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {RESULTS_DIR / 'metrics.json'}")
    print("XGBOOST BASELINE: COMPLETED")


if __name__ == "__main__":
    main()
