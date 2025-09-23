# FILE: src/training/models/mlp_ranker.py
from __future__ import annotations

from typing import Iterable, List
import torch
import torch.nn as nn


class Ranker(nn.Module):
    """
    MLP → scalar score. Используется как "скорер" для ранжирования пилотов внутри гонки.

    Args:
        in_dim:   размерность входного признакового вектора
        hidden:   список скрытых слоёв, например [256, 128]
        dropout:  dropout после каждого скрытого слоя
    """
    def __init__(self, in_dim: int, hidden: Iterable[int] = (256, 128), dropout: float = 0.10):
        super().__init__()
        hidden = list(hidden) if hidden is not None else []
        dims: List[int] = [in_dim] + hidden + [1]

        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))  # финальный скор

        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=0.01, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: [N, F] → scores: [N]
        """
        s = self.net(X).squeeze(-1)
        return s


def make_model(in_dim: int, hidden: Iterable[int] = (256, 128), dropout: float = 0.10) -> Ranker:
    """
    Фабричная функция для совместимости с конфигом/скриптами.
    """
    return Ranker(in_dim=in_dim, hidden=list(hidden) if hidden is not None else [], dropout=dropout)


__all__ = ["Ranker", "make_model"]
