#!/usr/bin/env python3
# scripts/train.py
from __future__ import annotations

from pathlib import Path
import json
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# всё берём из нашего training-пакета
from src.training import (
    # cfg & utils
    from_args, TrainConfig,
    set_seed, get_device, log, count_parameters,
    # io
    load_all, build_train_table, time_split,
    # features
    select_feature_cols, fit_scaler_on_df, transform_with_scaler_df,
    # dataset
    RaceListDataset,
    # engine
    train_one_epoch, evaluate, save_checkpoint, format_metrics,
)


# --------------------------------------------------------------------------------------
# Простая MLP-модель для ранжирования (скорер)
# --------------------------------------------------------------------------------------
class MLPScorer(nn.Module):
    """Лёгкий MLP: [in] -> Linear/ReLU/Dropout * L -> Linear(1) -> scores[N]
    """
    def __init__(self, in_dim: int, hidden: Sequence[int], dropout: float = 0.10):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in (hidden or []):
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N,F) -> (N,)
        out = self.net(x)
        return out.squeeze(-1)


def make_mlp_ranker(in_dim: int, hidden: Sequence[int], dropout: float) -> nn.Module:
    return MLPScorer(in_dim=in_dim, hidden=hidden, dropout=dropout)


# --------------------------------------------------------------------------------------
# Тренировочный раннер
# --------------------------------------------------------------------------------------

def main() -> None:
    # 1) конфиг из CLI
    cfg: TrainConfig = from_args()
    set_seed(cfg.seed, deterministic=True)
    device = get_device(cfg.device)
    log(f"Device: {device}")

    # 2) загрузка данных
    F, T = load_all(cfg.features_path, cfg.targets_path)
    if F.empty or T.empty:
        log("❌ Нет данных: проверьте пути --features / --targets")
        return

    # 3) join фич с таргетами + time split
    FT = build_train_table(F, T, dnf_position=cfg.dnf_position)
    TR, VA = time_split(FT, val_last=cfg.val_last)
    ntr = TR[["year", "round"]].drop_duplicates().shape[0]
    nva = VA[["year", "round"]].drop_duplicates().shape[0]
    log(f"Train races: {ntr} | Val races: {nva}")

    # 4) выбор признаков и скейлинг по train
    feature_cols = select_feature_cols(TR)
    scaler = fit_scaler_on_df(TR, feature_cols)

    TR_scaled = transform_with_scaler_df(TR, feature_cols, scaler, as_array=False).astype("float32")
    VA_scaled = transform_with_scaler_df(VA, feature_cols, scaler, as_array=False).astype("float32")

    # важный момент: не переписываем inplace колонки Int64 — просто подменяем блок фич float32
    TRn = pd.concat([TR.drop(columns=feature_cols), TR_scaled], axis=1)
    VAn = pd.concat([VA.drop(columns=feature_cols), VA_scaled], axis=1)

    # 5) датасеты (1 элемент = 1 гонка)
    tr_ds = RaceListDataset(TRn, feature_cols=feature_cols, dnf_position=cfg.dnf_position)
    va_ds = RaceListDataset(VAn, feature_cols=feature_cols, dnf_position=cfg.dnf_position)

    # 6) модель + оптимизатор
    model = make_mlp_ranker(in_dim=len(feature_cols), hidden=cfg.hidden, dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    log(f"Model params: {count_parameters(model):,} | in_dim={len(feature_cols)} | hidden={cfg.hidden} | drop={cfg.dropout}")

    # 7) обучение
    best = {"epoch": -1, "val_spearman": -1e9, "meta": {}}
    try:
        for epoch in range(1, cfg.epochs + 1):
            train_loss = train_one_epoch(model, tr_ds, opt, device=device)
            val = evaluate(model, va_ds, device=device)
            if (epoch % max(1, cfg.log_every)) == 0:
                log(format_metrics(epoch, train_loss, val["mean"]))

            # сохранение лучшего по Spearman
            if val["mean"].get("spearman", -1e9) > best["val_spearman"]:
                best["epoch"] = epoch
                best["val_spearman"] = float(val["mean"]["spearman"])
                best["meta"] = {
                    "hidden": cfg.hidden,
                    "dropout": cfg.dropout,
                    "lr": cfg.lr,
                    "weight_decay": cfg.weight_decay,
                    "epochs": cfg.epochs,
                    "val_last": cfg.val_last,
                    "seed": cfg.seed,
                    "device": str(device),
                    "in_dim": len(feature_cols),
                    "best_epoch": epoch,
                    "val_mean": val["mean"],
                }
                save_checkpoint(
                    artifacts_dir=cfg.artifacts_dir(),
                    model=model,
                    scaler=scaler,
                    feature_cols=feature_cols,
                    meta=best["meta"],
                )
    except KeyboardInterrupt:
        log("⏹️  Interrupted — finishing up…")

    # финальный лог
    if best["epoch"] > 0:
        log(
            f"✅ Saved best to {cfg.artifacts_dir()}/ "
            f"(epoch {best['epoch']}, spearman {best['val_spearman']:.4f})"
        )
    else:
        log("⚠️  Не удалось улучшить метрику — артефакты не сохранены")


if __name__ == "__main__":
    main()
