# FILE: src/training/losses.py
from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn


def _ensure_1d(x: torch.Tensor) -> torch.Tensor:
    # reshape, а не view: работает корректно и для не-contiguous
    if x.dim() != 1:
        x = x.reshape(-1)
    return x


@torch.jit.script
def _suffix_logsumexp(s: torch.Tensor) -> torch.Tensor:
    """
    Для каждого k вернуть logsumexp(s[k:]).
    Реализация без in-place через logcumsumexp по перевёрнутому вектору.
    """
    # вычитаем максимум для численной устойчивости
    m = torch.max(s)
    st = s - m
    # лог-сумма от хвоста: flip -> logcumsumexp -> flip назад
    rev = torch.logcumsumexp(st.flip(0), dim=0).flip(0)
    return rev + m


@torch.jit.script
def plackett_luce_nll(scores: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """
    Полный PL-NLL: sum_k [logsumexp(s[k:]) - s[k]] в наблюдённом порядке.
    scores: (N,) или (N,1)
    order:  (N,) long — индексы от победителя к последнему
    """
    scores = _ensure_1d(scores).contiguous()
    order = _ensure_1d(order).to(dtype=torch.long).contiguous()

    # переставляем скоры в наблюдённый порядок (winner-first)
    s = scores.gather(0, order)             # (N,)
    denom = _suffix_logsumexp(s)            # (N,)
    nll = denom - s
    return nll.sum()


@torch.jit.script
def plackett_luce_topk_nll(scores: torch.Tensor, order: torch.Tensor, topk: int) -> torch.Tensor:
    """
    PL-NLL только для первых K позиций (leaderboard@K).
    """
    scores = _ensure_1d(scores).contiguous()
    order = _ensure_1d(order).to(dtype=torch.long).contiguous()

    s = scores.gather(0, order)             # (N,)
    denom = _suffix_logsumexp(s)            # (N,)
    k = min(int(topk), s.size(0))
    nll = denom[:k] - s[:k]
    return nll.sum()


@torch.jit.script
def listmle_nll(scores: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """
    Классический ListMLE по наблюдённой перестановке — формула совпадает с PL.
    Оставляем отдельной функцией для ясности API.
    """
    return plackett_luce_nll(scores, order)


class PlackettLuceLoss(nn.Module):
    def __init__(self, topk: Optional[int] = None) -> None:
        super().__init__()
        self.topk = topk

    def forward(self, scores: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        if self.topk is None:
            return plackett_luce_nll(scores, order)
        return plackett_luce_topk_nll(scores, order, int(self.topk))
