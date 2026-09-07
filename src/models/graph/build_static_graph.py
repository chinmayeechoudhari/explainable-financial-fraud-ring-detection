"""Build a static, training-only graph for the first GNN baseline.

This module deliberately does not modify Member A's exploratory graph.
It builds an independent PyTorch Geometric representation from the cleaned
TRAIN transactions only.

No validation/test/future rows are used for graph structure or node features.

Node features:
    1. in_degree
    2. out_degree
    3. total_degree
    4. transaction_count
    5. log_incoming_amount
    6. log_outgoing_amount

Target leakage protection:
    - "Is Laundering" is never read.
    - No target-derived graph statistics are created.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_splits"
    / "train.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "static_gnn"
)

REQUIRED_COLUMNS = [
    "From Bank",
    "From Account",
    "To Bank",
    "To Account",
    "Amount Received",
    "Amount Paid",
]


def node_key(bank: pd.Series, account: pd.Series) -> pd.Series:
    """Create a stable account identifier from bank + account."""

    return (
        bank.astype(str)
        + "_"
        + account.astype(str)
    )


def build_graph(
    train_path: Path,
    chunk_size: int,
) -> tuple[Data, dict]:
    """Construct a sparse directed static graph using TRAIN only."""

    node_to_id: dict[str, int] = {}

    # Dynamic NumPy arrays.
    # These are expanded whenever a new node is discovered.
    in_degree = np.zeros(0, dtype=np.int64)
    out_degree = np.zeros(0, dtype=np.int64)
    total_tx = np.zeros(0, dtype=np.int64)

    incoming_amount = np.zeros(
        0,
        dtype=np.float64,
    )

    outgoing_amount = np.zeros(
        0,
        dtype=np.float64,
    )

    # Unique directed account relationships.
    pair_set: set[tuple[int, int]] = set()

    for chunk in pd.read_csv(
        train_path,
        usecols=REQUIRED_COLUMNS,
        chunksize=chunk_size,
    ):
        src_keys = node_key(
            chunk["From Bank"],
            chunk["From Account"],
        )

        dst_keys = node_key(
            chunk["To Bank"],
            chunk["To Account"],
        )

        # ---------------------------------------------------------
        # Assign IDs to newly observed nodes.
        # ---------------------------------------------------------

        unique_keys = pd.unique(
            pd.concat(
                [
                    src_keys,
                    dst_keys,
                ],
                ignore_index=True,
            )
        )

        new_keys = [
            str(key)
            for key in unique_keys
            if str(key) not in node_to_id
        ]

        if new_keys:
            start_id = len(node_to_id)

            for offset, key in enumerate(new_keys):
                node_to_id[key] = start_id + offset

            new_size = len(node_to_id)

            in_degree = np.pad(
                in_degree,
                (0, len(new_keys)),
                mode="constant",
            )

            out_degree = np.pad(
                out_degree,
                (0, len(new_keys)),
                mode="constant",
            )

            total_tx = np.pad(
                total_tx,
                (0, len(new_keys)),
                mode="constant",
            )

            incoming_amount = np.pad(
                incoming_amount,
                (0, len(new_keys)),
                mode="constant",
            )

            outgoing_amount = np.pad(
                outgoing_amount,
                (0, len(new_keys)),
                mode="constant",
            )

            assert len(in_degree) == new_size

        # ---------------------------------------------------------
        # Convert account keys to integer node IDs.
        # ---------------------------------------------------------

        src_ids = src_keys.map(
            node_to_id
        ).to_numpy(
            dtype=np.int64
        )

        dst_ids = dst_keys.map(
            node_to_id
        ).to_numpy(
            dtype=np.int64
        )

        amount_in = chunk[
            "Amount Received"
        ].to_numpy(
            dtype=np.float64
        )

        amount_out = chunk[
            "Amount Paid"
        ].to_numpy(
            dtype=np.float64
        )

        # ---------------------------------------------------------
        # Update node statistics.
        #
        # np.bincount is considerably faster than processing
        # millions of transactions with Python dictionaries.
        # ---------------------------------------------------------

        in_counts = np.bincount(
            dst_ids,
            minlength=len(node_to_id),
        )

        out_counts = np.bincount(
            src_ids,
            minlength=len(node_to_id),
        )

        total_counts = (
            in_counts
            + out_counts
        )

        incoming_sums = np.bincount(
            dst_ids,
            weights=amount_in,
            minlength=len(node_to_id),
        )

        outgoing_sums = np.bincount(
            src_ids,
            weights=amount_out,
            minlength=len(node_to_id),
        )

        in_degree += in_counts
        out_degree += out_counts
        total_tx += total_counts

        incoming_amount += incoming_sums
        outgoing_amount += outgoing_sums

        # ---------------------------------------------------------
        # Add unique directed account relationships.
        #
        # Repeated transactions between the same accounts become
        # one graph edge, matching the intended static graph
        # definition.
        # ---------------------------------------------------------

        pair_set.update(
            zip(
                src_ids.tolist(),
                dst_ids.tolist(),
            )
        )

    # -------------------------------------------------------------
    # Validate graph.
    # -------------------------------------------------------------

    if not pair_set:
        raise RuntimeError(
            "Training graph contains no edges."
        )

    num_nodes = len(node_to_id)

    if num_nodes == 0:
        raise RuntimeError(
            "Training graph contains no nodes."
        )

    # -------------------------------------------------------------
    # Construct node feature matrix.
    # -------------------------------------------------------------

    x = np.column_stack(
        [
            in_degree.astype(np.float32),
            out_degree.astype(np.float32),
            (
                in_degree + out_degree
            ).astype(np.float32),
            total_tx.astype(np.float32),
            np.log1p(
                incoming_amount
            ).astype(np.float32),
            np.log1p(
                outgoing_amount
            ).astype(np.float32),
        ]
    )

    # -------------------------------------------------------------
    # Normalize using TRAIN graph statistics only.
    # -------------------------------------------------------------

    mean = x.mean(
        axis=0,
        dtype=np.float64,
    )

    std = x.std(
        axis=0,
        dtype=np.float64,
    )

    # Prevent division by zero for constant features.
    std[std == 0] = 1.0

    x = (
        x - mean
    ) / std

    x = x.astype(
        np.float32
    )

    # -------------------------------------------------------------
    # Construct sparse directed edge index.
    # -------------------------------------------------------------

    edges = np.asarray(
        sorted(pair_set),
        dtype=np.int64,
    )

    edge_index = torch.tensor(
        edges.T,
        dtype=torch.long,
    )

    node_features = torch.tensor(
        x,
        dtype=torch.float32,
    )

    data = Data(
        x=node_features,
        edge_index=edge_index,
        num_nodes=num_nodes,
    )

    # -------------------------------------------------------------
    # Metadata.
    # -------------------------------------------------------------

    metadata = {
        "num_nodes": num_nodes,
        "num_edges": int(len(edges)),
        "node_feature_count": 6,
        "node_feature_names": [
            "in_degree",
            "out_degree",
            "total_degree",
            "transaction_count",
            "log_incoming_amount",
            "log_outgoing_amount",
        ],
        "train_path": str(train_path),
        "train_only": True,
        "target_used_in_graph_construction": False,
        "target_excluded_from_node_features": True,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "graph_type": "static_directed",
        "repeated_transactions_aggregated_to_unique_edges": True,
    }

    return data, metadata


def run_smoke_test() -> None:
    """Run a bounded smoke test on the first 5,000 TRAIN rows."""

    sample_path = (
        OUTPUT_DIR
        / "_smoke_train.csv"
    )

    smoke_columns = [
        *REQUIRED_COLUMNS,
        "Is Laundering",
    ]

    sample = pd.read_csv(
        TRAIN_PATH,
        usecols=smoke_columns,
        nrows=5000,
    )

    sample.to_csv(
        sample_path,
        index=False,
    )

    try:
        data, metadata = build_graph(
            sample_path,
            chunk_size=5000,
        )
    finally:
        sample_path.unlink(
            missing_ok=True
        )

    print(
        f"Smoke-test nodes: "
        f"{data.num_nodes}"
    )

    print(
        f"Smoke-test edges: "
        f"{data.edge_index.shape[1]}"
    )

    print(
        f"Node feature shape: "
        f"{tuple(data.x.shape)}"
    )

    print(
        f"Edge index shape: "
        f"{tuple(data.edge_index.shape)}"
    )

    print(
        f"Contains NaN: "
        f"{bool(torch.isnan(data.x).any())}"
    )

    print(
        f"Contains Inf: "
        f"{bool(torch.isinf(data.x).any())}"
    )

    print(
        f"Train-only metadata: "
        f"{metadata['train_only']}"
    )

    print(
        f"Target used in construction: "
        f"{metadata['target_used_in_graph_construction']}"
    )

    print(
        f"Target excluded from node features: "
        f"{metadata['target_excluded_from_node_features']}"
    )

    print(
        "STATIC GRAPH SMOKE TEST: PASSED"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a static TRAIN-only graph "
            "for the GCN baseline."
        )
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a bounded 5,000-row smoke test.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="CSV chunk size for graph construction.",
    )

    args = parser.parse_args()

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Missing TRAIN split: {TRAIN_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.smoke_test:
        run_smoke_test()
        return

    # -------------------------------------------------------------
    # Full TRAIN graph.
    # -------------------------------------------------------------

    print(
        "Building static TRAIN graph..."
    )

    print(
        f"TRAIN file: {TRAIN_PATH}"
    )

    print(
        f"Chunk size: {args.chunk_size:,}"
    )

    data, metadata = build_graph(
        TRAIN_PATH,
        args.chunk_size,
    )

    graph_path = (
        OUTPUT_DIR
        / "train_graph.pt"
    )

    metadata_path = (
        OUTPUT_DIR
        / "metadata.json"
    )

    torch.save(
        data,
        graph_path,
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
        )

    print()
    print(
        f"Saved graph: {graph_path}"
    )

    print(
        f"Saved metadata: {metadata_path}"
    )

    print(
        f"Nodes: {data.num_nodes:,}"
    )

    print(
        f"Edges: "
        f"{data.edge_index.shape[1]:,}"
    )

    print(
        f"Node features: "
        f"{tuple(data.x.shape)}"
    )

    print(
        f"Edge index: "
        f"{tuple(data.edge_index.shape)}"
    )

    print(
        f"Contains NaN: "
        f"{bool(torch.isnan(data.x).any())}"
    )

    print(
        f"Contains Inf: "
        f"{bool(torch.isinf(data.x).any())}"
    )

    print(
        "Target used in construction: "
        f"{metadata['target_used_in_graph_construction']}"
    )

    print(
        "STATIC TRAIN GRAPH: COMPLETED"
    )


if __name__ == "__main__":
    main()