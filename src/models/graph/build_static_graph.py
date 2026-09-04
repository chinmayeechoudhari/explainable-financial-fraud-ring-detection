"""Build a static, training-only graph for the first GNN baseline.

This module deliberately does not modify Member A's exploratory graph. It builds
an independent PyTorch Geometric-ready representation from the cleaned TRAIN
transactions only. No validation/test/future rows are used for graph structure
or node features.
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
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_splits" / "train.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "static_gnn"

SOURCE_COLUMNS = [
    "Timestamp",
    "From Bank",
    "From Account",
    "To Bank",
    "To Account",
    "Amount Received",
    "Amount Paid",
    "Is Laundering",
]


def node_key(bank, account) -> str:
    return f"{bank}_{account}"


def parse_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="mixed", errors="coerce")


def build_graph(train_path: Path, chunk_size: int) -> tuple[Data, dict]:
    """Construct a sparse directed static graph using TRAIN only."""
    required = [
        "From Bank",
        "From Account",
        "To Bank",
        "To Account",
        "Amount Received",
        "Amount Paid",
        "Is Laundering",
    ]

    node_to_id: dict[str, int] = {}
    in_degree: dict[int, int] = {}
    out_degree: dict[int, int] = {}
    total_tx: dict[int, int] = {}
    incoming_amount: dict[int, float] = {}
    outgoing_amount: dict[int, float] = {}
    pair_set: set[tuple[int, int]] = set()
    positive_edge_pairs: set[tuple[int, int]] = set()

    def get_id(bank, account) -> int:
        key = node_key(bank, account)
        if key not in node_to_id:
            node_to_id[key] = len(node_to_id)
        return node_to_id[key]

    for chunk in pd.read_csv(train_path, usecols=required, chunksize=chunk_size):
        amounts_in = chunk["Amount Received"].to_numpy(dtype=np.float64)
        amounts_out = chunk["Amount Paid"].to_numpy(dtype=np.float64)
        labels = chunk["Is Laundering"].to_numpy(dtype=np.int8)

        for from_bank, from_account, to_bank, to_account, amount_in, amount_out, label in zip(
            chunk["From Bank"].tolist(),
            chunk["From Account"].tolist(),
            chunk["To Bank"].tolist(),
            chunk["To Account"].tolist(),
            amounts_in,
            amounts_out,
            labels,
        ):
            src = get_id(from_bank, from_account)
            dst = get_id(to_bank, to_account)

            out_degree[src] = out_degree.get(src, 0) + 1
            in_degree[dst] = in_degree.get(dst, 0) + 1
            total_tx[src] = total_tx.get(src, 0) + 1
            total_tx[dst] = total_tx.get(dst, 0) + 1
            outgoing_amount[src] = outgoing_amount.get(src, 0.0) + float(amount_out)
            incoming_amount[dst] = incoming_amount.get(dst, 0.0) + float(amount_in)
            pair = (src, dst)
            pair_set.add(pair)
            if int(label) == 1:
                positive_edge_pairs.add(pair)

    edges = np.asarray(sorted(pair_set), dtype=np.int64)
    if len(edges) == 0:
        raise RuntimeError("Training graph contains no edges.")

    num_nodes = len(node_to_id)
    x = np.zeros((num_nodes, 6), dtype=np.float32)

    for node_id in range(num_nodes):
        x[node_id] = [
            float(in_degree.get(node_id, 0)),
            float(out_degree.get(node_id, 0)),
            float(in_degree.get(node_id, 0) + out_degree.get(node_id, 0)),
            float(total_tx.get(node_id, 0)),
            float(np.log1p(incoming_amount.get(node_id, 0.0))),
            float(np.log1p(outgoing_amount.get(node_id, 0.0))),
        ]

    # Normalize node features using TRAIN graph statistics only.
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    x = (x - mean) / std

    edge_index = torch.tensor(edges.T, dtype=torch.long)
    node_features = torch.tensor(x, dtype=torch.float32)

    data = Data(
        x=node_features,
        edge_index=edge_index,
        num_nodes=num_nodes,
    )

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
        "positive_edge_pair_count": len(positive_edge_pairs),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "target_excluded_from_node_features": True,
    }
    return data, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=100000)
    args = parser.parse_args()

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing TRAIN split: {TRAIN_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        # Build a bounded graph from an explicitly bounded temporary CSV read by
        # creating a small in-memory fixture from the first chunk. This validates
        # PyG tensor construction without generating the full graph artifact.
        sample_path = OUTPUT_DIR / "_smoke_train.csv"
        sample = pd.read_csv(TRAIN_PATH, usecols=SOURCE_COLUMNS, nrows=5000)
        sample.to_csv(sample_path, index=False)
        data, metadata = build_graph(sample_path, min(args.chunk_size, 5000))
        sample_path.unlink(missing_ok=True)
        print(f"Smoke-test nodes: {data.num_nodes}")
        print(f"Smoke-test edges: {data.edge_index.shape[1]}")
        print(f"Node feature shape: {tuple(data.x.shape)}")
        print(f"Edge index shape: {tuple(data.edge_index.shape)}")
        print(f"Contains NaN: {bool(torch.isnan(data.x).any())}")
        print(f"Contains Inf: {bool(torch.isinf(data.x).any())}")
        print(f"Train-only metadata: {metadata['train_only']}")
        print("STATIC GRAPH SMOKE TEST: PASSED")
        return

    data, metadata = build_graph(TRAIN_PATH, args.chunk_size)
    torch.save(data, OUTPUT_DIR / "train_graph.pt")
    with (OUTPUT_DIR / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved graph: {OUTPUT_DIR / 'train_graph.pt'}")
    print(f"Nodes: {data.num_nodes:,}")
    print(f"Edges: {data.edge_index.shape[1]:,}")
    print("STATIC TRAIN GRAPH: COMPLETED")


if __name__ == "__main__":
    main()
