# src/training/engine.py
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any
import json
import math
import hashlib  # for feature_cols hash

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from .dataset import collate_races
from torch.utils.data import DataLoader

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


def train_one_epoch(model, dataset, optimizer, device="cpu", batch_size: int = 1, shuffle: bool = True):
    model.train()
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_races)
    loss_fn = PlackettLuceLoss()
    running = 0.0
    n = 0

    for races in dl:  # collate возвращает список гонок
        for item in races:
            X = torch.as_tensor(item["X"], dtype=torch.float32, device=device)
            order = torch.as_tensor(item["order"], dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)

            # ВАЖНО: никаких in-place с outputs!
            scores = model(X)                 # (N,)
            loss = loss_fn(scores, order)     # только чистые функциональные операции

            loss.backward()
            optimizer.step()

            running += float(loss.detach().cpu())
            n += 1

            # Если хотите посчитать метрики на train — делайте это только на detached-копии
            # и под no_grad(), и НИЧЕГО in-place:
            # with torch.no_grad():
            #     pred_order = torch.argsort(scores.detach(), descending=True)
            #     _ = metrics_for_race(scores.detach(), order)

    return running / max(n, 1)


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

def _sha1_of_list(xs: List[str]) -> str:
    h = hashlib.sha1()
    for s in xs:
        h.update(s.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


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
    meta_out: Dict[str, Any] = dict(meta or {})
    # полезные поля по умолчанию
    meta_out.setdefault("in_dim", int(len(feature_cols)))
    meta_out.setdefault("num_features", int(len(feature_cols)))
    meta_out.setdefault("feature_cols_sha1", _sha1_of_list(list(feature_cols)))
    # число параметров модели (для контроля совместимости)
    try:
        meta_out.setdefault("num_params", int(sum(p.numel() for p in model.parameters())))
    except Exception:
        pass
    # если модель содержит подсказки по конфигу — сохраним
    for k in ("hidden", "dropout"):
        if k not in meta_out and hasattr(model, k):
            try:
                meta_out[k] = getattr(model, k)
            except Exception:
                pass

    with open(adir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)


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
