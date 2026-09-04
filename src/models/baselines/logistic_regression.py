"""Scalable transaction-level logistic regression baseline.

This uses scikit-learn's SGDClassifier with ``loss='log_loss'``. The learned
model is a regularized linear logistic model, while the SGD optimizer permits
out-of-core ``partial_fit`` over the 5.7M-row training split.

No random train/test split is performed. The validation and test periods remain
strictly later than training, and the test period is never used for fitting or
threshold selection.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.common.data_loader import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    load_split,
)
from src.models.common.evaluation import classification_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "baselines" / "logistic_regression.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "baselines" / "logistic_regression"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_encoder(train_path: Path, chunk_size: int) -> OneHotEncoder:
    """Discover categorical values from training data only."""
    categories = {column: set() for column in CATEGORICAL_FEATURES}

    for chunk in pd.read_csv(train_path, usecols=CATEGORICAL_FEATURES, chunksize=chunk_size):
        for column in CATEGORICAL_FEATURES:
            categories[column].update(chunk[column].dropna().tolist())

    ordered = [sorted(values, key=lambda value: str(value)) for values in categories.values()]
    if any(len(values) == 0 for values in ordered):
        raise ValueError("At least one categorical feature has no training categories.")

    # Explicit training-only categories prevent validation/test values from
    # changing the fitted feature space. One representative row is sufficient
    # because the category sets are supplied explicitly.
    fit_row = pd.DataFrame(
        [{column: values[0] for column, values in zip(CATEGORICAL_FEATURES, ordered)}]
    )
    encoder = OneHotEncoder(
        categories=ordered,
        handle_unknown="ignore",
        sparse_output=True,
        dtype=np.float64,
    )
    encoder.fit(fit_row)
    return encoder


def fit_scaler(train_path: Path, chunk_size: int) -> StandardScaler:
    """Fit numeric scaling statistics using TRAIN only, in chunks."""
    scaler = StandardScaler()
    for chunk in pd.read_csv(train_path, usecols=NUMERIC_FEATURES, chunksize=chunk_size):
        scaler.partial_fit(chunk.astype(np.float64))
    return scaler


def transform_chunk(
    chunk: pd.DataFrame,
    scaler: StandardScaler,
    encoder: OneHotEncoder,
):
    numeric = scaler.transform(chunk[NUMERIC_FEATURES].astype(np.float64))
    categorical = encoder.transform(chunk[CATEGORICAL_FEATURES])
    return hstack([numeric, categorical], format="csr")


def train_model(
    train_path: Path,
    scaler: StandardScaler,
    encoder: OneHotEncoder,
    config: dict,
) -> tuple[SGDClassifier, dict]:
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=float(config["alpha"]),
        learning_rate=config["learning_rate"],
        average=bool(config["average"]),
        random_state=int(config["random_state"]),
        fit_intercept=True,
        shuffle=True,
    )

    counts = {0: 0, 1: 0}
    for chunk in pd.read_csv(train_path, usecols=[TARGET_COLUMN], chunksize=config["chunk_size"]):
        counts[0] += int((chunk[TARGET_COLUMN] == 0).sum())
        counts[1] += int((chunk[TARGET_COLUMN] == 1).sum())

    total = counts[0] + counts[1]
    if counts[0] == 0 or counts[1] == 0:
        raise ValueError(f"Training split must contain both classes: {counts}")

    class_weights = {
        0: total / (2.0 * counts[0]),
        1: total / (2.0 * counts[1]),
    }

    print(f"Training positives: {counts[1]:,}")
    print(f"Training negatives: {counts[0]:,}")
    print(f"Balanced class weights: {class_weights}")

    start = time.perf_counter()

    for epoch in range(int(config["epochs"])):
        epoch_start = time.perf_counter()
        rows = 0

        # The split itself remains chronological. SGD shuffles samples only
        # inside each chunk for optimization; no row crosses a split boundary.
        for chunk in pd.read_csv(train_path, chunksize=config["chunk_size"]):
            y = chunk[TARGET_COLUMN].to_numpy(dtype=np.int8)
            X = transform_chunk(chunk, scaler, encoder)
            sample_weight = np.where(y == 1, class_weights[1], class_weights[0])

            classifier.partial_fit(
                X,
                y,
                classes=np.array([0, 1], dtype=np.int8),
                sample_weight=sample_weight,
            )
            rows += len(chunk)

        elapsed = time.perf_counter() - epoch_start
        print(f"Epoch {epoch + 1}/{config['epochs']}: {rows:,} rows in {elapsed:.1f}s")

    return classifier, {
        "training_seconds": time.perf_counter() - start,
        "train_rows": total,
        "train_positive_count": counts[1],
        "train_negative_count": counts[0],
        "class_weights": class_weights,
        "feature_count": int(classifier.n_features_in_),
    }


def evaluate_split(
    split: str,
    classifier: SGDClassifier,
    scaler: StandardScaler,
    encoder: OneHotEncoder,
    config: dict,
) -> tuple[dict, pd.DataFrame]:
    path = PROJECT_ROOT / "data" / "processed" / "modeling_splits" / f"{split}.csv"
    output_parts = []
    y_all = []
    score_all = []

    output_columns = ["Timestamp", "From Bank", "From Account", "To Bank", "To Account", TARGET_COLUMN]
    for chunk in pd.read_csv(
        path,
        usecols=output_columns + NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        chunksize=config["chunk_size"],
    ):
        y = chunk[TARGET_COLUMN].to_numpy(dtype=np.int8)
        X = transform_chunk(chunk, scaler, encoder)
        scores = classifier.predict_proba(X)[:, 1]
        predictions = (scores >= float(config["threshold"])).astype(np.int8)

        output_parts.append(
            pd.DataFrame({
                "Timestamp": chunk["Timestamp"].values,
                "From Bank": chunk["From Bank"].values,
                "From Account": chunk["From Account"].values,
                "To Bank": chunk["To Bank"].values,
                "To Account": chunk["To Account"].values,
                TARGET_COLUMN: y,
                "score": scores,
                "prediction": predictions,
            })
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Fit preprocessing on TRAIN and transform one small chunk; do not train.",
    )
    args = parser.parse_args()

    config = load_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "plots").mkdir(exist_ok=True)

    train_path = PROJECT_ROOT / "data" / "processed" / "modeling_splits" / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing training split: {train_path}")

    # Validate the complete schema without loading the full file into memory.
    load_split("train", columns=[TARGET_COLUMN])

    print("Fitting categorical encoder from TRAIN only...")
    encoder = build_encoder(train_path, int(config["chunk_size"]))
    print("Categorical encoder fitted.")

    print("Fitting numeric scaler from TRAIN only...")
    scaler = fit_scaler(train_path, int(config["chunk_size"]))
    print("Numeric scaler fitted.")

    if args.smoke_test:
        sample = next(pd.read_csv(train_path, chunksize=min(1000, int(config["chunk_size"]))))
        X = transform_chunk(sample, scaler, encoder)
        print(f"Smoke-test rows: {len(sample):,}")
        print(f"Transformed shape: {X.shape}")
        print("SMOKE TEST: PASSED")
        return

    classifier, training_info = train_model(train_path, scaler, encoder, config)

    metrics = {}
    predictions = {}
    for split in ("validation", "test"):
        print(f"Evaluating {split.upper()}...")
        split_metrics, split_predictions = evaluate_split(
            split, classifier, scaler, encoder, config
        )
        metrics[split] = split_metrics
        predictions[split] = split_predictions
        print(json.dumps(split_metrics, indent=2))

    model_path = RESULTS_DIR / "model.joblib"
    joblib.dump(
        {
            "classifier": classifier,
            "scaler": scaler,
            "encoder": encoder,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "target": TARGET_COLUMN,
            "config": config,
        },
        model_path,
    )

    for split, frame in predictions.items():
        frame.to_csv(RESULTS_DIR / f"predictions_{split}.csv", index=False)

    with (RESULTS_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"model": "scalable_logistic_regression", "training": training_info, "metrics": metrics},
            handle,
            indent=2,
            allow_nan=True,
        )

    with (RESULTS_DIR / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {RESULTS_DIR / 'metrics.json'}")
    print("LOGISTIC REGRESSION BASELINE: COMPLETED")


if __name__ == "__main__":
    main()
