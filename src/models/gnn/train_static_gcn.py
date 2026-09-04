"""Train and evaluate the Static GCN baseline.

The graph is constructed from TRAIN transactions only.

Validation and TEST transactions are used only for prediction/evaluation.
They never modify the graph or node features.

For accounts not observed in TRAIN, a deterministic zero embedding is used.
This preserves the complete validation/test population instead of silently
dropping transactions.
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
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import Adam

from src.models.gnn.static_gcn import StaticGCN


PROJECT_ROOT = Path(__file__).resolve().parents[3]

GRAPH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "static_gnn"
    / "train_graph.pt"
)

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_splits"
    / "train.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_splits"
    / "validation.csv"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_splits"
    / "test.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "baselines"
    / "static_gcn"
)


def set_seed(seed: int) -> None:
    """Make training as reproducible as practical."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_node_key(bank, account) -> str:
    """Create the same node key used by the static graph builder."""

    return f"{bank}_{account}"


def build_node_mapping(
    train_path: Path,
) -> dict[str, int]:
    """Reconstruct the TRAIN graph's account-to-node-ID mapping."""

    mapping: dict[str, int] = {}

    for chunk in pd.read_csv(
        train_path,
        usecols=[
            "From Bank",
            "From Account",
            "To Bank",
            "To Account",
        ],
        chunksize=100000,
    ):
        src_keys = (
            chunk["From Bank"].astype(str)
            + "_"
            + chunk["From Account"].astype(str)
        )

        dst_keys = (
            chunk["To Bank"].astype(str)
            + "_"
            + chunk["To Account"].astype(str)
        )

        keys = pd.concat(
            [
                src_keys,
                dst_keys,
            ],
            ignore_index=True,
        )

        for key in pd.unique(keys):
            key = str(key)

            if key not in mapping:
                mapping[key] = len(mapping)

    return mapping


def load_transaction_data(
    path: Path,
    mapping: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Load transaction node IDs and labels.

    Unknown accounts are represented by -1.

    They are NOT removed from the evaluation set.
    """

    sender_ids: list[int] = []
    receiver_ids: list[int] = []
    labels: list[int] = []

    unknown_count = 0

    required = [
        "From Bank",
        "From Account",
        "To Bank",
        "To Account",
        "Is Laundering",
    ]

    for chunk in pd.read_csv(
        path,
        usecols=required,
        chunksize=100000,
    ):
        src_keys = (
            chunk["From Bank"].astype(str)
            + "_"
            + chunk["From Account"].astype(str)
        )

        dst_keys = (
            chunk["To Bank"].astype(str)
            + "_"
            + chunk["To Account"].astype(str)
        )

        for src_key, dst_key, label in zip(
            src_keys,
            dst_keys,
            chunk["Is Laundering"],
        ):
            src_key = str(src_key)
            dst_key = str(dst_key)

            src_id = mapping.get(
                src_key,
                -1,
            )

            dst_id = mapping.get(
                dst_key,
                -1,
            )

            if src_id == -1:
                unknown_count += 1

            if dst_id == -1:
                unknown_count += 1

            sender_ids.append(src_id)
            receiver_ids.append(dst_id)
            labels.append(int(label))

    return (
        torch.tensor(
            sender_ids,
            dtype=torch.long,
        ),
        torch.tensor(
            receiver_ids,
            dtype=torch.long,
        ),
        torch.tensor(
            labels,
            dtype=torch.float32,
        ),
        unknown_count,
    )


def safe_embedding_lookup(
    embeddings: torch.Tensor,
    node_ids: torch.Tensor,
) -> torch.Tensor:
    """Return node embeddings while supporting unknown ID = -1.

    Unknown accounts receive an all-zero deterministic embedding.
    """

    embedding_dim = embeddings.shape[1]

    result = torch.zeros(
        (
            len(node_ids),
            embedding_dim,
        ),
        dtype=embeddings.dtype,
        device=embeddings.device,
    )

    known_mask = node_ids >= 0

    if known_mask.any():
        result[known_mask] = embeddings[
            node_ids[known_mask]
        ]

    return result


def predict_transactions(
    model: StaticGCN,
    embeddings: torch.Tensor,
    sender_ids: torch.Tensor,
    receiver_ids: torch.Tensor,
) -> torch.Tensor:
    """Predict transactions while supporting unknown accounts."""

    sender_embeddings = safe_embedding_lookup(
        embeddings,
        sender_ids,
    )

    receiver_embeddings = safe_embedding_lookup(
        embeddings,
        receiver_ids,
    )

    pair_embeddings = torch.cat(
        [
            sender_embeddings,
            receiver_embeddings,
        ],
        dim=1,
    )

    logits = model.classifier(
        pair_embeddings
    ).squeeze(-1)

    return logits


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Calculate fraud-detection metrics."""

    predictions = (
        probabilities >= threshold
    ).astype(np.int8)

    metrics = {
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                labels,
                probabilities,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                labels,
                probabilities,
            )
        ),
        "false_positive_rate": float(
            (
                ((predictions == 1) & (labels == 0)).sum()
                / max(
                    (labels == 0).sum(),
                    1,
                )
            )
        ),
        "threshold": threshold,
        "positive_count": int(
            labels.sum()
        ),
        "negative_count": int(
            len(labels) - labels.sum()
        ),
    }

    return metrics


def precision_recall_at_k(
    labels: np.ndarray,
    probabilities: np.ndarray,
    k: int,
) -> tuple[float, float]:
    """Calculate Precision@K and Recall@K."""

    if len(labels) == 0:
        return 0.0, 0.0

    k = min(
        k,
        len(labels),
    )

    top_indices = np.argsort(
        -probabilities
    )[:k]

    top_labels = labels[
        top_indices
    ]

    precision = float(
        top_labels.mean()
    )

    total_positive = labels.sum()

    recall = (
        float(
            top_labels.sum()
            / total_positive
        )
        if total_positive > 0
        else 0.0
    )

    return precision, recall


def evaluate(
    model: StaticGCN,
    graph,
    sender_ids: torch.Tensor,
    receiver_ids: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[dict, np.ndarray]:
    """Evaluate one split."""

    model.eval()

    with torch.no_grad():
        embeddings = model.encode(
            graph.x,
            graph.edge_index,
        )

        logits = predict_transactions(
            model,
            embeddings,
            sender_ids,
            receiver_ids,
        )

        probabilities = torch.sigmoid(
            logits
        )

    probabilities_np = (
        probabilities.cpu().numpy()
    )

    labels_np = (
        labels.cpu()
        .numpy()
        .astype(np.int8)
    )

    metrics = calculate_metrics(
        labels_np,
        probabilities_np,
    )

    for k in [
        100,
        500,
        1000,
    ]:
        precision_k, recall_k = (
            precision_recall_at_k(
                labels_np,
                probabilities_np,
                k,
            )
        )

        metrics[
            f"precision_at_{k}"
        ] = precision_k

        metrics[
            f"recall_at_{k}"
        ] = recall_k

    return (
        metrics,
        probabilities_np,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Load graph.
    # ---------------------------------------------------------

    print(
        "Loading static TRAIN graph..."
    )

    graph = torch.load(
        GRAPH_PATH,
        map_location="cpu",
        weights_only=False,
    )

    print(
        f"Nodes: {graph.num_nodes:,}"
    )

    print(
        f"Edges: "
        f"{graph.edge_index.shape[1]:,}"
    )

    print(
        f"Node features: "
        f"{tuple(graph.x.shape)}"
    )

    # ---------------------------------------------------------
    # Reconstruct node mapping.
    # ---------------------------------------------------------

    print()
    print(
        "Building TRAIN node mapping..."
    )

    node_mapping = build_node_mapping(
        TRAIN_PATH
    )

    print(
        f"Mapped nodes: "
        f"{len(node_mapping):,}"
    )

    # ---------------------------------------------------------
    # Load transaction supervision.
    # ---------------------------------------------------------

    print()
    print(
        "Loading transaction supervision..."
    )

    (
        train_sender,
        train_receiver,
        train_labels,
        train_unknown,
    ) = load_transaction_data(
        TRAIN_PATH,
        node_mapping,
    )

    (
        val_sender,
        val_receiver,
        val_labels,
        val_unknown,
    ) = load_transaction_data(
        VALIDATION_PATH,
        node_mapping,
    )

    (
        test_sender,
        test_receiver,
        test_labels,
        test_unknown,
    ) = load_transaction_data(
        TEST_PATH,
        node_mapping,
    )

    print(
        f"TRAIN transactions: "
        f"{len(train_labels):,}"
    )

    print(
        f"VALIDATION transactions: "
        f"{len(val_labels):,}"
    )

    print(
        f"TEST transactions: "
        f"{len(test_labels):,}"
    )

    print(
        f"TRAIN positives: "
        f"{int(train_labels.sum()):,}"
    )

    print(
        f"Unknown TRAIN account references - "
        f"TRAIN: {train_unknown:,}, "
        f"VALIDATION: {val_unknown:,}, "
        f"TEST: {test_unknown:,}"
    )

    # ---------------------------------------------------------
    # Model.
    # ---------------------------------------------------------

    model = StaticGCN(
        input_dim=graph.x.shape[1],
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    )

    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
    )

    positive_count = float(
        train_labels.sum()
    )

    negative_count = float(
        len(train_labels)
        - positive_count
    )

    pos_weight = torch.tensor(
        negative_count
        / max(
            positive_count,
            1.0,
        ),
        dtype=torch.float32,
    )

    print()
    print(
        f"Positive class weight: "
        f"{pos_weight.item():.6f}"
    )

    # ---------------------------------------------------------
    # Training.
    # ---------------------------------------------------------

    print()
    print(
        "Starting Static GCN training..."
    )

    start_time = time.time()

    best_val_pr_auc = -np.inf
    best_state = None

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        epoch_start = time.time()

        model.train()

        optimizer.zero_grad()

        embeddings = model.encode(
            graph.x,
            graph.edge_index,
        )

        logits = predict_transactions(
            model,
            embeddings,
            train_sender,
            train_receiver,
        )

        loss = F.binary_cross_entropy_with_logits(
            logits,
            train_labels,
            pos_weight=pos_weight,
        )

        loss.backward()

        optimizer.step()

        train_seconds = (
            time.time()
            - epoch_start
        )

        val_metrics, _ = evaluate(
            model,
            graph,
            val_sender,
            val_receiver,
            val_labels,
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Loss: {loss.item():.6f} | "
            f"Val PR-AUC: "
            f"{val_metrics['pr_auc']:.6f} | "
            f"Time: {train_seconds:.1f}s"
        )

        if (
            val_metrics["pr_auc"]
            > best_val_pr_auc
        ):
            best_val_pr_auc = (
                val_metrics["pr_auc"]
            )

            best_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

    total_seconds = (
        time.time()
        - start_time
    )

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    # ---------------------------------------------------------
    # Final evaluation.
    # ---------------------------------------------------------

    print()
    print(
        "Evaluating best Static GCN..."
    )

    val_metrics, val_probabilities = evaluate(
        model,
        graph,
        val_sender,
        val_receiver,
        val_labels,
    )

    test_metrics, test_probabilities = evaluate(
        model,
        graph,
        test_sender,
        test_receiver,
        test_labels,
    )

    val_metrics[
        "training_seconds"
    ] = total_seconds

    test_metrics[
        "training_seconds"
    ] = total_seconds

    val_metrics[
        "feature_count"
    ] = int(
        graph.x.shape[1]
    )

    test_metrics[
        "feature_count"
    ] = int(
        graph.x.shape[1]
    )

    print()
    print(
        "VALIDATION RESULTS"
    )

    print(
        json.dumps(
            val_metrics,
            indent=2,
        )
    )

    print()
    print(
        "TEST RESULTS"
    )

    print(
        json.dumps(
            test_metrics,
            indent=2,
        )
    )

    # ---------------------------------------------------------
    # Save artifacts.
    # ---------------------------------------------------------

    checkpoint_path = (
        RESULTS_DIR
        / "model.pt"
    )

    torch.save(
        model.state_dict(),
        checkpoint_path,
    )

    with (
        RESULTS_DIR
        / "metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "model": "StaticGCN",
                "config": vars(args),
                "unknown_account_policy": (
                    "zero_embedding"
                ),
                "validation": val_metrics,
                "test": test_metrics,
            },
            handle,
            indent=2,
        )

    pd.DataFrame(
        {
            "label": val_labels.numpy(),
            "probability": val_probabilities,
        }
    ).to_csv(
        RESULTS_DIR
        / "validation_predictions.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "label": test_labels.numpy(),
            "probability": test_probabilities,
        }
    ).to_csv(
        RESULTS_DIR
        / "test_predictions.csv",
        index=False,
    )

    print()
    print(
        f"Saved checkpoint: "
        f"{checkpoint_path}"
    )

    print(
        f"Saved results: "
        f"{RESULTS_DIR}"
    )

    print(
        "STATIC GCN TRAINING: COMPLETED"
    )


if __name__ == "__main__":
    main()