# FILE: src/training/metrics.py
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


ArrayLike = Union[np.ndarray, "torch.Tensor", Sequence[float], Sequence[int]]


# ---------- helpers ----------

def _to_numpy(x: ArrayLike) -> np.ndarray:
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def order_to_ranks(order: ArrayLike) -> np.ndarray:
    """
    Convert winner-first order (indices 0..N-1) to rank vector of size N with values 1..N.
      order = [winner_idx, ..., last_idx]  →  ranks[i] = place of i (1 = best)
    """
    ord_np = _to_numpy(order).astype(int)
    N = int(ord_np.size)
    ranks = np.empty(N, dtype=np.float32)
    ranks[ord_np] = np.arange(1, N + 1, dtype=np.float32)
    return ranks


def scores_to_order(scores: ArrayLike, descending: bool = True) -> np.ndarray:
    """
    Convert scores to order (indices sorted by score).
    """
    s = _to_numpy(scores).astype(np.float64)
    return np.argsort(-s if descending else s).astype(int)


def spearmanr_from_ranks(a_ranks: ArrayLike, b_ranks: ArrayLike) -> float:
    """
    Spearman rho between two rank vectors of equal length.
    Returns NaN if length < 2 or degenerate variance.
    """
    a = _to_numpy(a_ranks).astype(np.float64)
    b = _to_numpy(b_ranks).astype(np.float64)
    if a.size < 2:
        return float("nan")
    # handle constant ranks safely
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    # Pearson correlation of ranks
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    return float(np.clip((a * b).mean(), -1.0, 1.0))


def mae_positions(pred_ranks: ArrayLike, true_ranks: ArrayLike) -> float:
    """
    Mean Absolute Error between predicted and true finishing positions.
    """
    p = _to_numpy(pred_ranks).astype(np.float64)
    t = _to_numpy(true_ranks).astype(np.float64)
    if p.size == 0:
        return float("nan")
    return float(np.mean(np.abs(p - t)))


def top1_accuracy(pred_order: ArrayLike, true_order: ArrayLike) -> float:
    """
    1.0 if predicted winner equals true winner, else 0.0.
    """
    p = _to_numpy(pred_order).astype(int)
    t = _to_numpy(true_order).astype(int)
    if p.size == 0 or t.size == 0:
        return 0.0
    return float(p[0] == t[0])


def ndcg_at_k(pred_scores: ArrayLike, true_ranks: ArrayLike, k: Optional[int] = None) -> float:
    """
    nDCG@k with "relevance" derived from inverse of true ranks (higher relevance for better places).
    Useful if you care more about top positions.
    """
    s = _to_numpy(pred_scores).astype(np.float64)
    tr = _to_numpy(true_ranks).astype(np.float64)
    N = s.size
    if N == 0:
        return float("nan")
    if k is None or k > N:
        k = N

    # relevance: higher for better (rank=1 → rel=1, rank=N → rel≈1/N)
    rel = 1.0 / tr
    # predicted order
    idx = np.argsort(-s)
    rel_pred = rel[idx][:k]
    # ideal order
    idx_ideal = np.argsort(-rel)
    rel_ideal = rel[idx_ideal][:k]

    def _dcg(vals):
        denom = np.log2(np.arange(2, k + 2))
        return float(np.sum(vals / denom))

    dcg = _dcg(rel_pred)
    idcg = _dcg(rel_ideal) + 1e-12
    return float(dcg / idcg)


# ---------- per-race wrapper ----------

def metrics_for_race(pred_scores: ArrayLike, true_order: ArrayLike) -> Dict[str, float]:
    """
    Compute core metrics for a single race:
      - spearman: Spearman between predicted and true ranks
      - mae_pos:  MAE of predicted vs true finishing positions
      - top1:     Winner accuracy
      - ndcg5:    nDCG@5 (optional, uses inverse ranks as relevance)
    """
    pred_scores = _to_numpy(pred_scores)
    true_order = _to_numpy(true_order).astype(int)

    pred_order = scores_to_order(pred_scores, descending=True)
    pred_ranks = order_to_ranks(pred_order)
    true_ranks = order_to_ranks(true_order)

    return {
        "spearman": spearmanr_from_ranks(pred_ranks, true_ranks),
        "mae_pos": mae_positions(pred_ranks, true_ranks),
        "top1": top1_accuracy(pred_order, true_order),
        "ndcg5": ndcg_at_k(pred_scores, true_ranks, k=5),
    }


# ---------- aggregate across races ----------

def aggregate_metrics(per_race: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Average metrics across races (ignores NaNs).
    """
    if not per_race:
        return {}
    keys = sorted({k for d in per_race for k in d.keys()})
    out: Dict[str, float] = {}
    for k in keys:
        vals = np.array([d.get(k, np.nan) for d in per_race], dtype=np.float64)
        out[k] = float(np.nanmean(vals))
    return out


__all__ = [
    "order_to_ranks",
    "scores_to_order",
    "spearmanr_from_ranks",
    "mae_positions",
    "top1_accuracy",
    "ndcg_at_k",
    "metrics_for_race",
    "aggregate_metrics",
]
