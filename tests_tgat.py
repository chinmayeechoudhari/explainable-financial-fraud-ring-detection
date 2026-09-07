from __future__ import annotations

import torch

from src.models.temporal.tgat_core import TGATTransactionModel


def main() -> None:
    torch.manual_seed(42)
    batch_size = 4
    neighbors = 5
    transaction_dim = 27
    event_dim = 27

    model = TGATTransactionModel(
        transaction_dim=transaction_dim,
        event_dim=event_dim,
        hidden_dim=32,
        time_dim=16,
        num_heads=2,
        dropout=0.0,
    )
    model.eval()

    transaction_features = torch.randn(batch_size, transaction_dim)
    sender_features = torch.randn(batch_size, neighbors + 1, event_dim)
    receiver_features = torch.randn(batch_size, neighbors + 1, event_dim)
    sender_delta = torch.randint(1, 3600, (batch_size, neighbors)).float()
    receiver_delta = torch.randint(1, 3600, (batch_size, neighbors)).float()

    with torch.no_grad():
        logits, attention = model(
            transaction_features,
            sender_features,
            sender_delta,
            receiver_features,
            receiver_delta,
            return_attention=True,
        )

    assert logits.shape == (batch_size,)
    assert torch.isfinite(logits).all()
    assert attention is not None
    assert len(attention["sender"]) == 2
    assert len(attention["receiver"]) == 2

    for side in ("sender", "receiver"):
        for layer_attention in attention[side]:
            assert layer_attention.shape == (batch_size, neighbors)
            assert torch.isfinite(layer_attention).all()
            assert torch.allclose(
                layer_attention.sum(dim=-1),
                torch.ones(batch_size),
                atol=1e-5,
            )

    print("TGAT output shape: PASSED")
    print("TGAT finite logits: PASSED")
    print("TGAT attention shape: PASSED")
    print("TGAT attention normalization: PASSED")
    print("TGAT SMOKE TEST: PASSED")


if __name__ == "__main__":
    main()
