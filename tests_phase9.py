from __future__ import annotations

import torch

from src.models.temporal.tgat_core import TGATTransactionModel


def _inputs(transaction_dim: int, event_dim: int, batch: int = 3, k: int = 4):
    return (
        torch.randn(batch, transaction_dim),
        torch.randn(batch, k + 1, event_dim),
        torch.randint(1, 3600, (batch, k)).float(),
        torch.randn(batch, k + 1, event_dim),
        torch.randint(1, 3600, (batch, k)).float(),
    )


def test_no_time_encoding_preserves_neighbor_mask():
    model = TGATTransactionModel(
        transaction_dim=11, event_dim=11, hidden_dim=32,
        time_dim=16, num_heads=2, dropout=0.0, use_time_encoding=False
    )
    args = _inputs(11, 11)
    logits, attention = model(*args, return_attention=True)
    assert logits.shape == (3,)
    assert attention is not None
    assert len(attention["sender"]) == 2


def test_single_layer_variant():
    model = TGATTransactionModel(
        transaction_dim=11, event_dim=11, hidden_dim=32,
        time_dim=16, num_heads=2, dropout=0.0, num_layers=1
    )
    args = _inputs(11, 11)
    logits, attention = model(*args, return_attention=True)
    assert logits.shape == (3,)
    assert attention is not None
    assert len(attention["sender"]) == 2
    assert attention["sender"][1] is None


def test_no_neighbor_batch_has_zero_valid_deltas():
    from src.models.temporal.tgat_dataset import ALL_FEATURES, CURRENT_FEATURES, EventFeatureStore, build_batch
    import pandas as pd

    row = {
        "From Account": "A", "To Account": "B", "_timestamp": 100,
        "Is Laundering": 0,
    }
    for name in ALL_FEATURES:
        row[name] = 0.0
    frame = pd.DataFrame([row])
    batch = build_batch(
        frame, EventFeatureStore(max_history=10), 10,
        active_features=list(CURRENT_FEATURES), use_neighbors=False
    )
    assert batch.transaction_features.shape == (1, len(CURRENT_FEATURES))
    assert float(batch.sender_delta_seconds.sum()) == 0.0
    assert float(batch.receiver_delta_seconds.sum()) == 0.0


def test_same_timestamp_micro_batches_share_pre_state():
    import pandas as pd
    from src.models.temporal.tgat_dataset import ALL_FEATURES, EventFeatureStore
    from src.models.temporal.train_tgat import make_batches

    rows = []
    for i in range(5):
        row = {
            "From Account": "A",
            "To Account": "B",
            "_timestamp": 100,
            "Is Laundering": 0,
        }
        for name in ALL_FEATURES:
            row[name] = 0.0
        rows.append(row)

    frame = pd.DataFrame(rows)
    store = EventFeatureStore(max_history=10)
    batches = make_batches(frame, store, batch_size=2, neighbor_k=2)

    first = next(batches)
    second = next(batches)
    third = next(batches)

    assert float(first.sender_delta_seconds.sum()) == 0.0
    assert float(second.sender_delta_seconds.sum()) == 0.0
    assert float(third.sender_delta_seconds.sum()) == 0.0

    # The generator has not resumed past the timestamp group yet, so no
    # transaction at T=100 may have entered history.
    assert store.store.recent_events(store.get_id("A"), 101, 10) == []

    # Resume the generator so the complete timestamp-100 group is inserted
    # into history before evaluating a later timestamp.
    try:
        next(batches)
    except StopIteration:
        pass

    with_new_timestamp = pd.DataFrame([{
        **{
            "From Account": "A",
            "To Account": "C",
            "_timestamp": 101,
            "Is Laundering": 0,
        },
        **{name: 0.0 for name in ALL_FEATURES},
    }])
    next_batches = make_batches(
        with_new_timestamp,
        store,
        batch_size=2,
        neighbor_k=2,
    )
    next_batch = next(next_batches)

    assert float(next_batch.sender_delta_seconds[0].sum()) > 0.0
