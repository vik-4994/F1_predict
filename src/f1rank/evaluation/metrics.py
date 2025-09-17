from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def ndcg_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 3) -> float:
    k = min(k, len(y_true))
    order = np.argsort(y_pred)[:k]
    gains = 1 / y_true[order]
    dcg = np.sum(gains / np.log2(np.arange(2, k + 2)))
    ideal_order = np.argsort(y_true)[:k]
    idcg = np.sum((1 / y_true[ideal_order]) / np.log2(np.arange(2, k + 2)))
    return float(dcg / idcg) if idcg > 0 else np.nan

def racewise_metrics(df: pd.DataFrame, pred_col: str, k_list=(3,5)) -> dict:
    stats = {"spearman": [], **{f"ndcg@{k}": [] for k in k_list}}
    for _, grp in df.groupby("raceId"):
        if grp["finish_pos"].nunique() < 2: 
            continue
        pr = grp[pred_col].rank(method="average", ascending=True).values
        target = grp["finish_pos"].values.astype(float)
        sr = spearmanr(pr, target).correlation
        stats["spearman"].append(sr)
        for k in k_list:
            stats[f"ndcg@{k}"].append(ndcg_k(target, grp[pred_col].values.astype(float), k))
    return {k: (float(np.nanmean(v)) if len(v) else np.nan) for k, v in stats.items()}
