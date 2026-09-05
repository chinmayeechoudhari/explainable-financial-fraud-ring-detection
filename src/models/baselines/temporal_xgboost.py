"""
Temporal XGBoost baseline.

Uses:
    - 11 current transaction features
    - 16 temporal state features
    - 27 total model features

The model is trained only on the chronological training split.
Validation is used for evaluation/model monitoring.
Test is used only for final evaluation.
"""

import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.common.evaluation import classification_metrics


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

CURRENT_TRANSACTION_FEATURES = [
    "Amount Received",
    "Amount Paid",
    "Amount Difference",
    "Amount Ratio",
    "Same Bank Transaction",
    "Cross Bank Transaction",
    "Transaction Time Category",
    "Is Weekend",
    "Log Amount Received",
    "Log Amount Paid",
    "Same Currency",
]


TEMPORAL_FEATURE_COLUMNS = [
    "sender_in_count",
    "sender_out_count",
    "sender_total_count",
    "sender_in_amount",
    "sender_out_amount",
    "sender_avg_in_amount",
    "sender_avg_out_amount",
    "sender_time_since_last",
    "receiver_in_count",
    "receiver_out_count",
    "receiver_total_count",
    "receiver_in_amount",
    "receiver_out_amount",
    "receiver_avg_in_amount",
    "receiver_avg_out_amount",
    "receiver_time_since_last",
]


FEATURE_COLUMNS = (
    CURRENT_TRANSACTION_FEATURES
    + TEMPORAL_FEATURE_COLUMNS
)


TARGET_COLUMN = "Is Laundering"


# ============================================================
# PATHS
# ============================================================

DATA_DIR = (
    "data/processed/temporal_gnn"
)

RESULTS_DIR = (
    "results/baselines/temporal_xgboost"
)


# ============================================================
# DATA LOADING
# ============================================================

def load_split(split_name):
    """
    Load one temporal modeling split.

    Only model features and the target are loaded.
    Account IDs are intentionally excluded from X.
    """

    path = os.path.join(
        DATA_DIR,
        f"{split_name}.csv",
    )

    print(
        f"\nLoading {split_name.upper()}..."
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Temporal dataset not found: {path}"
        )

    df = pd.read_csv(
        path,
        usecols=[
            *FEATURE_COLUMNS,
            TARGET_COLUMN,
        ],
        low_memory=False,
    )

    X = df[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=np.float32
    )

    y = df[
        TARGET_COLUMN
    ].to_numpy(
        dtype=np.int8
    )

    return X, y


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_features(
    X,
    y,
    split_name,
):
    """
    Validate model input before training/evaluation.
    """

    if np.isnan(X).any():
        raise ValueError(
            f"{split_name}: NaN detected"
        )

    if np.isinf(X).any():
        raise ValueError(
            f"{split_name}: Inf detected"
        )

    if not np.isin(
        y,
        [0, 1],
    ).all():
        raise ValueError(
            f"{split_name}: invalid labels"
        )

    print(
        f"{split_name.upper()} rows: "
        f"{len(y):,}"
    )

    print(
        f"{split_name.upper()} positives: "
        f"{int(y.sum()):,}"
    )

    print(
        f"{split_name.upper()} negatives: "
        f"{int((y == 0).sum()):,}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    print("=" * 70)
    print(
        "TEMPORAL XGBOOST BASELINE"
    )
    print("=" * 70)

    print(
        f"\nCurrent transaction features: "
        f"{len(CURRENT_TRANSACTION_FEATURES)}"
    )

    print(
        f"Temporal state features: "
        f"{len(TEMPORAL_FEATURE_COLUMNS)}"
    )

    print(
        f"Total model features: "
        f"{len(FEATURE_COLUMNS)}"
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    X_train, y_train = load_split(
        "train"
    )

    X_val, y_val = load_split(
        "validation"
    )

    X_test, y_test = load_split(
        "test"
    )

    # ========================================================
    # VALIDATE DATA
    # ========================================================

    validate_features(
        X_train,
        y_train,
        "train",
    )

    validate_features(
        X_val,
        y_val,
        "validation",
    )

    validate_features(
        X_test,
        y_test,
        "test",
    )

    # ========================================================
    # CLASS IMBALANCE
    # ========================================================

    train_positive = int(
        y_train.sum()
    )

    train_negative = int(
        (y_train == 0).sum()
    )

    scale_pos_weight = (
        train_negative
        / train_positive
    )

    print(
        f"\nScale pos weight: "
        f"{scale_pos_weight:.6f}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print(
        "\nTraining temporal XGBoost..."
    )

    start_time = time.time()

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_val, y_val),
        ],
        verbose=False,
    )

    training_seconds = (
        time.time()
        - start_time
    )

    print(
        f"Training completed in "
        f"{training_seconds:.2f} seconds"
    )

    # ========================================================
    # PREDICTION SCORES
    # ========================================================

    print(
        "\nGenerating validation predictions..."
    )

    validation_scores = (
        model.predict_proba(
            X_val
        )[:, 1]
    )

    print(
        "Generating test predictions..."
    )

    test_scores = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    # ========================================================
    # EVALUATION
    # ========================================================

    validation_metrics = (
        classification_metrics(
            y_true=y_val,
            scores=validation_scores,
            threshold=0.5,
        )
    )

    test_metrics = (
        classification_metrics(
            y_true=y_test,
            scores=test_scores,
            threshold=0.5,
        )
    )

    # ========================================================
    # ADD EXPERIMENT METADATA
    # ========================================================

    validation_metrics[
        "training_seconds"
    ] = training_seconds

    test_metrics[
        "training_seconds"
    ] = training_seconds

    validation_metrics[
        "feature_count"
    ] = len(FEATURE_COLUMNS)

    test_metrics[
        "feature_count"
    ] = len(FEATURE_COLUMNS)

    validation_metrics[
        "current_transaction_feature_count"
    ] = len(
        CURRENT_TRANSACTION_FEATURES
    )

    test_metrics[
        "current_transaction_feature_count"
    ] = len(
        CURRENT_TRANSACTION_FEATURES
    )

    validation_metrics[
        "temporal_feature_count"
    ] = len(
        TEMPORAL_FEATURE_COLUMNS
    )

    test_metrics[
        "temporal_feature_count"
    ] = len(
        TEMPORAL_FEATURE_COLUMNS
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = os.path.join(
        RESULTS_DIR,
        "model.json",
    )

    model.save_model(
        model_path
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    validation_predictions = (
        (validation_scores >= 0.5)
        .astype(np.int8)
    )

    test_predictions = (
        (test_scores >= 0.5)
        .astype(np.int8)
    )

    validation_predictions_path = os.path.join(
        RESULTS_DIR,
        "validation_predictions.csv",
    )

    test_predictions_path = os.path.join(
        RESULTS_DIR,
        "test_predictions.csv",
    )

    pd.DataFrame(
        {
            "score": validation_scores,
            "prediction": validation_predictions,
            "label": y_val,
        }
    ).to_csv(
        validation_predictions_path,
        index=False,
    )

    pd.DataFrame(
        {
            "score": test_scores,
            "prediction": test_predictions,
            "label": y_test,
        }
    ).to_csv(
        test_predictions_path,
        index=False,
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics = {
        "model": (
            "Temporal XGBoost"
        ),

        "feature_count": len(
            FEATURE_COLUMNS
        ),

        "current_transaction_feature_count": len(
            CURRENT_TRANSACTION_FEATURES
        ),

        "temporal_feature_count": len(
            TEMPORAL_FEATURE_COLUMNS
        ),

        "train_rows": len(
            y_train
        ),

        "train_positive": train_positive,

        "train_negative": train_negative,

        "scale_pos_weight": (
            scale_pos_weight
        ),

        "training_seconds": (
            training_seconds
        ),

        "hyperparameters": {
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.08,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "random_state": 42,
        },

        "validation": (
            validation_metrics
        ),

        "test": (
            test_metrics
        ),
    }

    metrics_path = os.path.join(
        RESULTS_DIR,
        "metrics.json",
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )

    # ========================================================
    # PRINT VALIDATION RESULTS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "VALIDATION RESULTS"
    )

    print(
        "=" * 70
    )

    for key, value in (
        validation_metrics.items()
    ):

        print(
            f"{key}: {value}"
        )

    # ========================================================
    # PRINT TEST RESULTS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "TEST RESULTS"
    )

    print(
        "=" * 70
    )

    for key, value in (
        test_metrics.items()
    ):

        print(
            f"{key}: {value}"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "TEMPORAL XGBOOST COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        f"Model: {model_path}"
    )

    print(
        f"Metrics: {metrics_path}"
    )

    print(
        f"Validation predictions: "
        f"{validation_predictions_path}"
    )

    print(
        f"Test predictions: "
        f"{test_predictions_path}"
    )


if __name__ == "__main__":
    main()