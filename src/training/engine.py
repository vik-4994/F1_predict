                        
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any
import json
import hashlib                         

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from .dataset import collate_races
from torch.utils.data import DataLoader

from .losses import PlackettLuceLoss
from .metrics import aggregate_metrics, metrics_for_race, outcome_metrics
from .featureset import FeatureScaler, save_feature_cols, load_feature_cols

                           
                          
                           

def _unpack_model_output(output: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(output, dict):
        rank_scores = output.get("rank_scores")
        status_logits = output.get("status_logits")
        if rank_scores is None:
            raise ValueError("Model output dict must contain 'rank_scores'")
        return rank_scores, status_logits
    if isinstance(output, (list, tuple)):
        if not output:
            raise ValueError("Model returned an empty tuple/list")
        rank_scores = output[0]
        status_logits = output[1] if len(output) > 1 else None
        return rank_scores, status_logits
    return output, None


def _rank_loss_for_item(
    loss_fn: nn.Module,
    rank_scores: torch.Tensor,
    item: Dict[str, Any],
    device: str,
) -> torch.Tensor:
    finish_idx = item.get("finish_idx")
    finish_order = item.get("finish_order")
    if finish_idx is None or finish_order is None:
        order = torch.as_tensor(item["order"], dtype=torch.long, device=device)
        if order.numel() < 2:
            return rank_scores.sum() * 0.0
        return loss_fn(rank_scores, order)

    finish_idx = torch.as_tensor(finish_idx, dtype=torch.long, device=device)
    finish_order = torch.as_tensor(finish_order, dtype=torch.long, device=device)
    if finish_idx.numel() < 2 or finish_order.numel() < 2:
        return rank_scores.sum() * 0.0
    finish_scores = rank_scores.index_select(0, finish_idx)
    return loss_fn(finish_scores, finish_order)


def train_one_epoch(
    model,
    dataset,
    optimizer,
    device="cpu",
    batch_size: int = 1,
    shuffle: bool = True,
    status_loss_weight: float = 1.0,
    status_class_weights: torch.Tensor | None = None,
):
    model.train()
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_races)
    loss_fn = PlackettLuceLoss()
    running = 0.0
    n = 0

    for races in dl:                                   
        for item in races:
            X = torch.as_tensor(item["X"], dtype=torch.float32, device=device)
            race_weight = float(item.get("race_weight", 1.0))

            optimizer.zero_grad(set_to_none=True)

            output = model(X)
            rank_scores, status_logits = _unpack_model_output(output)

            rank_loss = _rank_loss_for_item(loss_fn, rank_scores, item, device)
            status_loss = rank_loss.detach() * 0.0
            if status_logits is not None and "status_target" in item:
                status_target = torch.as_tensor(item["status_target"], dtype=torch.long, device=device)
                class_weights = status_class_weights.to(device) if status_class_weights is not None else None
                status_loss = F.cross_entropy(status_logits, status_target, weight=class_weights)
            loss = rank_loss + float(status_loss_weight) * status_loss
            weighted_loss = loss * race_weight

            weighted_loss.backward()
            optimizer.step()

            running += float(weighted_loss.detach().cpu())
            n += 1

                                                                                           
                                                 
                                   
                                                                              
                                                              

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
        output = model(X)
        rank_scores, status_logits = _unpack_model_output(output)

        if "finish_idx" in item and "finish_order" in item:
            finish_idx = torch.as_tensor(item["finish_idx"], dtype=torch.long, device=device)
            finish_order = torch.as_tensor(item["finish_order"], dtype=torch.long, device=device)
            if finish_idx.numel() >= 2 and finish_order.numel() >= 2:
                finish_scores = rank_scores.index_select(0, finish_idx)
                m = metrics_for_race(finish_scores, finish_order)
            else:
                m = {"spearman": float("nan"), "mae_pos": float("nan"), "top1": float("nan"), "ndcg5": float("nan")}
        else:
            m = metrics_for_race(rank_scores, item["order"])

        if status_logits is not None and "status_target" in item:
            status_target = torch.as_tensor(item["status_target"], dtype=torch.long, device=device)
            status_pred = torch.argmax(status_logits, dim=-1)
            m.update(outcome_metrics(status_target, status_pred))
        per_race.append(m)

        d = dict(m)
        d.update({"year": int(item["year"]), "round": int(item["round"])})
        details.append(d)

    return {
        "mean": aggregate_metrics(per_race),
        "per_race": details,
    }


                           
                           
                           

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

            
    torch.save(model.state_dict(), adir / "ranker.pt")

             
    scaler.save(adir / "scaler.json")

                           
    save_feature_cols(adir / "feature_cols.txt", feature_cols)

                
    meta_out: Dict[str, Any] = dict(meta or {})
                                
    meta_out.setdefault("in_dim", int(len(feature_cols)))
    meta_out.setdefault("num_features", int(len(feature_cols)))
    meta_out.setdefault("feature_cols_sha1", _sha1_of_list(list(feature_cols)))
                                                          
    try:
        meta_out.setdefault("num_params", int(sum(p.numel() for p in model.parameters())))
    except Exception:
        pass
                                                          
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

                                                     
    in_dim = meta.get("in_dim", len(feature_cols))
    hidden = meta.get("hidden", [256, 128])
    dropout = meta.get("dropout", 0.10)

    try:
        model = make_model(
            in_dim=in_dim,
            hidden=hidden,
            dropout=dropout,
            num_status_classes=int(meta.get("num_status_classes", 0)),
        ).to(device)
    except TypeError:
        model = make_model(in_dim=in_dim, hidden=hidden, dropout=dropout).to(device)
    state = torch.load(adir / "ranker.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model, scaler, feature_cols, meta


                           
                       
                           

def format_metrics(epoch: int, train_loss: float, val_mean: Dict[str, float]) -> str:
    sp = val_mean.get("spearman", float("nan"))
    mae = val_mean.get("mae_pos", float("nan"))
    t1 = val_mean.get("top1", float("nan"))
    nd = val_mean.get("ndcg5", float("nan"))
    status_acc = val_mean.get("status_acc", float("nan"))
    status_f1 = val_mean.get("status_macro_f1", float("nan"))
    return (f"Epoch {epoch:03d} | train_nll={train_loss:.2f} | "
            f"val_spearman={sp:.4f} | val_mae={mae:.3f} | top1={t1:.3f} | ndcg5={nd:.3f} | "
            f"status_acc={status_acc:.3f} | status_f1={status_f1:.3f}")


__all__ = [
    "train_one_epoch",
    "evaluate",
    "save_checkpoint",
    "load_checkpoint",
    "format_metrics",
]
