#!/usr/bin/env python3
                  
from __future__ import annotations

from pathlib import Path
import json
import sys
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.frame_utils import filter_feature_cols
from src.scenario_support import (
    FUTURE_FEATURE_PROFILE,
    FUTURE_MODE,
    normalize_feature_profile,
    select_feature_profile_cols,
)

                                     
from src.training import (
                 
    from_args, TrainConfig,
    set_seed, get_device, log, count_parameters,
        
    load_all, build_train_table, time_split,
              
    select_feature_cols, fit_scaler_on_df, transform_with_scaler_df,
             
    RaceListDataset,
            
    train_one_epoch, evaluate, save_checkpoint, format_metrics,
)


                                                                                        
                                              
                                                                                        
class MLPScorer(nn.Module):
    """Lightweight MLP scorer: [in] -> Linear/ReLU/Dropout * L -> Linear(1)."""
    def __init__(self, in_dim: int, hidden: Sequence[int], dropout: float = 0.10):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in (hidden or []):
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:                 
        out = self.net(x)
        return out.squeeze(-1)


def make_mlp_ranker(in_dim: int, hidden: Sequence[int], dropout: float) -> nn.Module:
    return MLPScorer(in_dim=in_dim, hidden=hidden, dropout=dropout)


def _apply_feature_ablation(feature_cols: List[str], cfg: TrainConfig) -> tuple[List[str], List[str]]:
    kept, dropped = filter_feature_cols(
        feature_cols,
        drop_prefixes=cfg.drop_prefixes or [],
        drop_contains=cfg.drop_contains or [],
        drop_exact=cfg.drop_cols or [],
        keep_prefixes=cfg.keep_prefixes or [],
    )
    if not kept:
        raise ValueError("Feature ablation removed all feature columns")
    return kept, dropped


def _apply_feature_profile(feature_cols: List[str], cfg: TrainConfig) -> tuple[List[str], List[str]]:
    profile = normalize_feature_profile(cfg.feature_profile)
    kept = select_feature_profile_cols(feature_cols, profile)
    dropped = [col for col in feature_cols if col not in set(kept)]
    return kept, dropped


                                                                                        
                      
                                                                                        

def main() -> None:
                      
    cfg: TrainConfig = from_args()
    set_seed(cfg.seed, deterministic=True)
    device = get_device(cfg.device)
    log(f"Device: {device}")

                        
    F, T = load_all(cfg.features_path, cfg.targets_path)
    if F.empty or T.empty:
        log("No data loaded. Check --features and --targets paths.")
        return

                                          
    FT = build_train_table(F, T, dnf_position=cfg.dnf_position)
    TR, VA = time_split(FT, val_last=cfg.val_last)
    ntr = TR[["year", "round"]].drop_duplicates().shape[0]
    nva = VA[["year", "round"]].drop_duplicates().shape[0]
    log(f"Train races: {ntr} | Val races: {nva}")

                                            
    feature_cols = select_feature_cols(TR)
    original_feature_count = len(feature_cols)
    feature_cols, profile_dropped_cols = _apply_feature_profile(feature_cols, cfg)
    profile_kept_count = len(feature_cols)
    feature_cols, dropped_cols = _apply_feature_ablation(feature_cols, cfg)
    profile = normalize_feature_profile(cfg.feature_profile)
    if profile_dropped_cols:
        preview = ", ".join(profile_dropped_cols[:12])
        if len(profile_dropped_cols) > 12:
            preview += ", ..."
        log(
            f"Feature profile '{profile}': kept {profile_kept_count} / {original_feature_count} features | "
            f"dropped {len(profile_dropped_cols)} [{preview}]"
        )
    if dropped_cols:
        preview = ", ".join(dropped_cols[:12])
        if len(dropped_cols) > 12:
            preview += ", ..."
        log(
            f"Ablation: kept {len(feature_cols)} / {len(feature_cols) + len(dropped_cols)} features | "
            f"dropped {len(dropped_cols)} [{preview}]"
        )
    elif any((cfg.drop_prefixes, cfg.drop_contains, cfg.drop_cols, cfg.keep_prefixes)):
        log(f"Ablation: filters applied but no feature columns were dropped | kept {len(feature_cols)}")
    scaler = fit_scaler_on_df(TR, feature_cols)

    TR_scaled = transform_with_scaler_df(TR, feature_cols, scaler, as_array=False).astype("float32")
    VA_scaled = transform_with_scaler_df(VA, feature_cols, scaler, as_array=False).astype("float32")

                                                                                              
    TRn = pd.concat([TR.drop(columns=feature_cols), TR_scaled], axis=1)
    VAn = pd.concat([VA.drop(columns=feature_cols), VA_scaled], axis=1)

                                       
    tr_ds = RaceListDataset(TRn, feature_cols=feature_cols, dnf_position=cfg.dnf_position)
    va_ds = RaceListDataset(VAn, feature_cols=feature_cols, dnf_position=cfg.dnf_position)

                             
    model = make_mlp_ranker(in_dim=len(feature_cols), hidden=cfg.hidden, dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    log(f"Model params: {count_parameters(model):,} | in_dim={len(feature_cols)} | hidden={cfg.hidden} | drop={cfg.dropout}")

                 
    best = {"epoch": -1, "val_spearman": -1e9, "meta": {}}
    try:
        for epoch in range(1, cfg.epochs + 1):
            train_loss = train_one_epoch(model, tr_ds, opt, device=device)
            val = evaluate(model, va_ds, device=device)
            if (epoch % max(1, cfg.log_every)) == 0:
                log(format_metrics(epoch, train_loss, val["mean"]))

                                            
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
                    "drop_prefixes": cfg.drop_prefixes or [],
                    "drop_contains": cfg.drop_contains or [],
                    "drop_cols": cfg.drop_cols or [],
                    "keep_prefixes": cfg.keep_prefixes or [],
                    "feature_profile": profile,
                    "scenario_mode": FUTURE_MODE if profile == FUTURE_FEATURE_PROFILE else "observed",
                    "profile_dropped_feature_cols": profile_dropped_cols,
                    "dropped_feature_cols": dropped_cols,
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

                   
    if best["epoch"] > 0:
        log(
            f"✅ Saved best to {cfg.artifacts_dir()}/ "
            f"(epoch {best['epoch']}, spearman {best['val_spearman']:.4f})"
        )
    else:
        log("Could not improve the validation metric; artifacts were not saved.")


if __name__ == "__main__":
    main()
