# FILE: src/training/losses.py
from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn


@torch.jit.script
def _pl_cumlogsumexp(sorted_scores: torch.Tensor) -> torch.Tensor:
    """
    Cumulative logsumexp from the tail: for s[k:] compute logsumexp(s[k:]).
    sorted_scores: [N] (scores already permuted in observed order: winner first)
    Returns: [N] tensor with denom_k = logsumexp(s[k:])
    """
    # numerical stability: subtract max
    s = sorted_scores
    maxv = torch.max(s)
    se = torch.exp(s - maxv)
    # cumulative sum from end, then log and add max back
    cumsum_rev = torch.flip(torch.cumsum(torch.flip(se, dims=[0]), dim=0), dims=[0])
    return torch.log(cumsum_rev) + maxv


def plackett_luce_nll(scores: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """
    Plackett–Luce negative log-likelihood for a single list (one race).

    Args:
        scores: [N] real-valued scores (higher = better rank)
        order:  [N] indices giving observed order, winner first (0..N-1)

    Returns:
        Scalar tensor: NLL = -sum_k log( exp(s[π_k]) / sum_{j>=k} exp(s[π_j]) )
                     = sum_k (logsumexp(s[k:]) - s[k]) in the observed order
    """
    s = scores[order]                     # [N], reorder so true winner is first
    denom = _pl_cumlogsumexp(s)           # [N]
    return torch.sum(denom - s)


def plackett_luce_topk_nll(scores: torch.Tensor, order: torch.Tensor, k: int) -> torch.Tensor:
    """
    Top-K variant of PL-NLL: учитывает только первые k мест (k<=N).
    Полезно, если важнее верх списка.
    """
    s = scores[order]
    k = int(min(max(k, 1), s.numel()))
    denom = _pl_cumlogsumexp(s)
    return torch.sum(denom[:k] - s[:k])


def listmle_nll(scores: torch.Tensor, order: torch.Tensor, jitter: float = 0.0) -> torch.Tensor:
    """
    ListMLE (Cao et al., 2007), реализован через тот же рецепт, что и PL.
    При 'order' = наблюдаемому порядку победителей, совпадает с PL-NLL.

    Args:
        scores: [N]
        order:  [N] winner-first
        jitter: добавляет N(0, jitter) к scores для tie-breaking (0 = выкл)

    Returns:
        Scalar NLL
    """
    s = scores
    if jitter > 0:
        s = s + torch.randn_like(s) * jitter
    return plackett_luce_nll(s, order)


class PlackettLuceLoss(nn.Module):
    """nn.Module-обёртка над plackett_luce_nll (удобно для оптимайзера/логгера)."""
    def __init__(self, topk: Optional[int] = None):
        super().__init__()
        self.topk = topk

    def forward(self, scores: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        if self.topk is None:
            return plackett_luce_nll(scores, order)
        return plackett_luce_topk_nll(scores, order, self.topk)


__all__ = [
    "plackett_luce_nll",
    "plackett_luce_topk_nll",
    "listmle_nll",
    "PlackettLuceLoss",
]
