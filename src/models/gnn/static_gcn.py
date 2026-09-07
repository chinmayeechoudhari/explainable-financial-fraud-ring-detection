"""Static GCN encoder and transaction-level classifier.

The graph is account-level:
    node = Bank + Account
    edge = directed account relationship

The prediction target remains transaction-level:
    Is Laundering ∈ {0, 1}

The GCN produces account embeddings, which are combined for the
sender and receiver of each transaction and passed to an MLP classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class StaticGCN(nn.Module):
    """Two-layer GCN with a transaction-level prediction head."""

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 32,
        embedding_dim: int = 16,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.gcn1 = GCNConv(
            input_dim,
            hidden_dim,
        )

        self.gcn2 = GCNConv(
            hidden_dim,
            embedding_dim,
        )

        self.dropout = dropout

        self.classifier = nn.Sequential(
            nn.Linear(
                embedding_dim * 2,
                16,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                16,
                1,
            ),
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Generate an embedding for every account node."""

        x = self.gcn1(
            x,
            edge_index,
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.gcn2(
            x,
            edge_index,
        )

        return x

    def predict_transactions(
        self,
        node_embeddings: torch.Tensor,
        sender_ids: torch.Tensor,
        receiver_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Predict laundering probability logits for transactions."""

        sender_embeddings = node_embeddings[
            sender_ids
        ]

        receiver_embeddings = node_embeddings[
            receiver_ids
        ]

        pair_embeddings = torch.cat(
            [
                sender_embeddings,
                receiver_embeddings,
            ],
            dim=1,
        )

        logits = self.classifier(
            pair_embeddings
        ).squeeze(-1)

        return logits

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        sender_ids: torch.Tensor,
        receiver_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run GCN encoding followed by transaction prediction."""

        embeddings = self.encode(
            x,
            edge_index,
        )

        return self.predict_transactions(
            embeddings,
            sender_ids,
            receiver_ids,
        )