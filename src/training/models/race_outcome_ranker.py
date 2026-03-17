from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn


class RaceOutcomeRanker(nn.Module):
    """Shared MLP backbone with separate ranking and outcome heads."""

    def __init__(
        self,
        in_dim: int,
        hidden: Iterable[int],
        dropout: float = 0.10,
        num_status_classes: int = 0,
    ):
        super().__init__()
        hidden_dims = [int(h) for h in (list(hidden) if hidden is not None else [])]

        layers: List[nn.Module] = []
        prev = int(in_dim)
        for width in hidden_dims:
            layers.extend([nn.Linear(prev, width), nn.ReLU(), nn.Dropout(float(dropout))])
            prev = width

        self.backbone = nn.Sequential(*layers)
        self.rank_head = nn.Linear(prev, 1)
        self.status_head = nn.Linear(prev, int(num_status_classes)) if int(num_status_classes) > 0 else None

        self.in_dim = int(in_dim)
        self.hidden = hidden_dims
        self.dropout = float(dropout)
        self.num_status_classes = int(num_status_classes)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        rank_scores = self.rank_head(h).squeeze(-1)
        if self.status_head is None:
            return rank_scores
        status_logits = self.status_head(h)
        return rank_scores, status_logits


def make_model(
    in_dim: int,
    hidden: Iterable[int] = (128, 64),
    dropout: float = 0.10,
    num_status_classes: int = 0,
) -> RaceOutcomeRanker:
    return RaceOutcomeRanker(
        in_dim=in_dim,
        hidden=hidden,
        dropout=dropout,
        num_status_classes=num_status_classes,
    )


__all__ = ["RaceOutcomeRanker", "make_model"]
