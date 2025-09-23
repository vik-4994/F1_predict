# FILE: src/training/engine.py
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any
import json
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .losses import PlackettLuceLoss, plackett_luce_nll
from .metrics import metrics_for_race, aggregate_metrics
from .featureset import FeatureScaler, save_feature_cols, load_feature_cols


# -------------------------
# Core train/eval routines
# -------------------------

@torch.no_grad()
def _forward_scores(model: nn.Module, X: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    model.eval()
    return model(X.to(device))


def train_one_epoch(
    model: nn.Module,
    dataset: Dataset,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    grad_clip: float = 2.0,
    shuffle: bool = True,
) -> float:
    """
    Одну эпоху обходим гонки (1 элемент датасета = 1 гонка).
    Возвращает суммарный NLL по эпохе.
    """
    model.train()
    if loss_fn is None:
        # лист-вайз по всему списку (вся гонка)
        def loss_fn(scores: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
            return plackett_luce_nll(scores, order)

    order_idx = np.random.permutation(len(dataset)) if shuffle else np.arange(len(dataset))
    total_loss = 0.0

    for idx in order_idx:
        item = dataset[idx]
        X: torch.Tensor = item["X"].to(device)         # [N,F]
        order = item["order"].to(device)               # [N]

        scores = model(X)                              # [N]
        loss = loss_fn(scores, order)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: Dataset,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Считает метрики по каждой гонке и усредняет.
    Возвращает:
      {
        "mean": {spearman, mae_pos, top1, ndcg5},
        "per_race": [ {year, round, spearman, ...}, ... ]
      }
    """
    model.eval()
    per_race: List[Dict[str, float]] = []
    details: List[Dict[str, Any]] = []

    for i in range(len(dataset)):
        item = dataset[i]
        X = item["X"].to(device)
        scores = model(X)

        m = metrics_for_race(scores, item["order"])
        per_race.append(m)

        d = dict(m)
        d.update({"year": int(item["year"]), "round": int(item["round"])})
        details.append(d)

    return {
        "mean": aggregate_metrics(per_race),
        "per_race": details,
    }


# -------------------------
# Checkpointing / artifacts
# -------------------------

def save_checkpoint(
    artifacts_dir: Path,
    model: nn.Module,
    scaler: FeatureScaler,
    feature_cols: List[str],
    meta: Optional[Dict[str, Any]] = None,
):
    """
    Сохраняем:
      - ranker.pt        (state_dict модели)
      - scaler.json      (параметры скейлера)
      - feature_cols.txt (порядок фич)
      - meta.json        (конфиг/метрики/любая служебная инфа)
    """
    adir = Path(artifacts_dir)
    adir.mkdir(parents=True, exist_ok=True)

    # модель
    torch.save(model.state_dict(), adir / "ranker.pt")

    # скейлер
    scaler.save(adir / "scaler.json")

    # фичи (порядок важен!)
    save_feature_cols(adir / "feature_cols.txt", feature_cols)

    # метаданные
    meta = dict(meta or {})
    # полезно записать in_dim, чтобы при загрузке проверить совместимость
    meta.setdefault("in_dim", int(len(feature_cols)))
    with open(adir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_checkpoint(
    artifacts_dir: Path,
    make_model: Callable[..., nn.Module],
    device: str = "cpu",
) -> Tuple[nn.Module, FeatureScaler, List[str], Dict[str, Any]]:
    """
    Загружает артефакты и возвращает (model, scaler, feature_cols, meta).
    Требует фабрику make_model(in_dim, hidden=..., dropout=...) — см. models/mlp_ranker.make_model.
    """
    adir = Path(artifacts_dir)
    with open(adir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols = load_feature_cols(adir / "feature_cols.txt")
    scaler = FeatureScaler.load(adir / "scaler.json")

    # параметры модели из meta (бэкенд-совместимость)
    in_dim = meta.get("in_dim", len(feature_cols))
    hidden = meta.get("hidden", [256, 128])
    dropout = meta.get("dropout", 0.10)

    model = make_model(in_dim=in_dim, hidden=hidden, dropout=dropout).to(device)
    state = torch.load(adir / "ranker.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model, scaler, feature_cols, meta


# -------------------------
# Pretty logging helper
# -------------------------

def format_metrics(epoch: int, train_loss: float, val_mean: Dict[str, float]) -> str:
    sp = val_mean.get("spearman", float("nan"))
    mae = val_mean.get("mae_pos", float("nan"))
    t1 = val_mean.get("top1", float("nan"))
    nd = val_mean.get("ndcg5", float("nan"))
    return (f"Epoch {epoch:03d} | train_nll={train_loss:.2f} | "
            f"val_spearman={sp:.4f} | val_mae={mae:.3f} | top1={t1:.3f} | ndcg5={nd:.3f}")


__all__ = [
    "train_one_epoch",
    "evaluate",
    "save_checkpoint",
    "load_checkpoint",
    "format_metrics",
]
