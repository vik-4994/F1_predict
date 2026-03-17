#!/usr/bin/env python3
                  
from __future__ import annotations

from pathlib import Path
import sys
from typing import List

import numpy as np
import pandas as pd
import torch

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
from src.training.config import (
    BASELINE_DROP_COLS,
    BASELINE_DROP_CONTAINS,
    BASELINE_DROP_PREFIXES,
    FUTURE_TRAIN_RECENCY_HALF_LIFE,
)
from src.training.models.race_outcome_ranker import make_model as make_mlp_ranker
from src.training.outcomes import OUTCOME_LABELS

                                     
from src.training import (
                 
    from_args, TrainConfig,
    set_seed, get_device, log, count_parameters,
        
    load_all, build_train_table, time_split, race_recency_weights,
    combine_race_weight_maps, default_era_weights, regulation_era_race_weights, summarize_era_weights,
              
    select_feature_cols, fit_scaler_on_df, transform_with_scaler_df,
             
    RaceListDataset,
            
    train_one_epoch, evaluate, save_checkpoint, format_metrics,
)


                                                                                        
                                              
                                                                                        
def _status_class_weights(df: pd.DataFrame) -> torch.Tensor:
    counts = (
        pd.to_numeric(df.get("outcome_id"), errors="coerce")
        .dropna()
        .astype(int)
        .value_counts()
        .reindex(range(len(OUTCOME_LABELS)), fill_value=0)
        .astype(float)
    )
    safe = counts.to_numpy(dtype=np.float32, copy=False)
    safe = np.where(safe > 0, safe, 1.0)
    weights = safe.sum() / safe
    weights = weights / max(float(weights.mean()), 1e-6)
    weights = np.clip(weights, 0.5, 8.0)
    return torch.tensor(weights, dtype=torch.float32)


def _resolved_ablation_filters(cfg: TrainConfig) -> tuple[List[str], List[str], List[str], bool]:
    profile = normalize_feature_profile(cfg.feature_profile)
    drop_prefixes = list(cfg.drop_prefixes or [])
    drop_contains = list(cfg.drop_contains or [])
    drop_cols = list(cfg.drop_cols or [])
    uses_relaxed_future_defaults = (
        profile == FUTURE_FEATURE_PROFILE
        and not (cfg.keep_prefixes or [])
        and drop_prefixes == list(BASELINE_DROP_PREFIXES)
        and drop_contains == list(BASELINE_DROP_CONTAINS)
        and drop_cols == list(BASELINE_DROP_COLS)
    )
    if uses_relaxed_future_defaults:
        return [], [], [], True
    return drop_prefixes, drop_contains, drop_cols, False


def _apply_feature_ablation(
    feature_cols: List[str],
    cfg: TrainConfig,
) -> tuple[List[str], List[str], List[str], List[str], List[str], bool]:
    drop_prefixes, drop_contains, drop_cols, uses_relaxed_future_defaults = _resolved_ablation_filters(cfg)
    kept, dropped = filter_feature_cols(
        feature_cols,
        drop_prefixes=drop_prefixes,
        drop_contains=drop_contains,
        drop_exact=drop_cols,
        keep_prefixes=cfg.keep_prefixes or [],
    )
    if not kept:
        raise ValueError("Feature ablation removed all feature columns")
    return kept, dropped, drop_prefixes, drop_contains, drop_cols, uses_relaxed_future_defaults


def _apply_feature_profile(feature_cols: List[str], cfg: TrainConfig) -> tuple[List[str], List[str]]:
    profile = normalize_feature_profile(cfg.feature_profile)
    kept = select_feature_profile_cols(feature_cols, profile)
    dropped = [col for col in feature_cols if col not in set(kept)]
    return kept, dropped


def _resolved_train_recency_half_life(cfg: TrainConfig) -> float | None:
    if cfg.train_recency_half_life is not None:
        value = float(cfg.train_recency_half_life)
        return value if np.isfinite(value) and value > 0 else None
    profile = normalize_feature_profile(cfg.feature_profile)
    if profile == FUTURE_FEATURE_PROFILE:
        return float(FUTURE_TRAIN_RECENCY_HALF_LIFE)
    return None


def _resolved_era_weights(cfg: TrainConfig) -> dict[str, float]:
    weights = default_era_weights()
    weights.update({str(k): float(v) for k, v in (cfg.era_weights or {}).items()})
    return weights


def _metric_or_neg_inf(metrics: dict[str, float], key: str) -> float:
    value = float(metrics.get(key, float("nan")))
    return value if np.isfinite(value) else float("-inf")


def _selection_score(metrics: dict[str, float]) -> float:
    spearman = _metric_or_neg_inf(metrics, "spearman")
    status_macro_f1 = _metric_or_neg_inf(metrics, "status_macro_f1")

    if np.isfinite(spearman) and np.isfinite(status_macro_f1):
        return spearman + status_macro_f1
    if np.isfinite(status_macro_f1):
        return status_macro_f1
    return spearman


                                                                                        
                      
                                                                                        

def main() -> None:
                      
    cfg: TrainConfig = from_args()
    set_seed(cfg.seed, deterministic=True)
    device = get_device(cfg.device)
    log(f"Device: {device}")

                        
    F, T = load_all(cfg.features_path, cfg.targets_path)
    if F.empty or T.empty:
        log("No data loaded. Check --features and --targets paths.")
        return

                                          
    FT = build_train_table(F, T, dnf_position=cfg.dnf_position, dsq_position=cfg.dsq_position)
    TR, VA = time_split(FT, val_last=cfg.val_last)
    ntr = TR[["year", "round"]].drop_duplicates().shape[0]
    nva = VA[["year", "round"]].drop_duplicates().shape[0]
    log(f"Train races: {ntr} | Val races: {nva}")

                                            
    feature_cols = select_feature_cols(TR)
    original_feature_count = len(feature_cols)
    feature_cols, profile_dropped_cols = _apply_feature_profile(feature_cols, cfg)
    profile_kept_count = len(feature_cols)
    feature_cols, dropped_cols, drop_prefixes, drop_contains, drop_cols, uses_relaxed_future_defaults = _apply_feature_ablation(feature_cols, cfg)
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
    elif any((drop_prefixes, drop_contains, drop_cols, cfg.keep_prefixes)):
        log(f"Ablation: filters applied but no feature columns were dropped | kept {len(feature_cols)}")
    if uses_relaxed_future_defaults:
        log("Future profile: using relaxed ablation defaults to keep strategy/pit-ops priors.")

    train_recency_half_life = _resolved_train_recency_half_life(cfg)
    recency_race_weights = race_recency_weights(TR, train_recency_half_life)
    era_weight_overrides = _resolved_era_weights(cfg)
    era_race_weights = (
        regulation_era_race_weights(TR, era_weights=era_weight_overrides)
        if cfg.use_regulation_era_weights
        else {}
    )
    train_race_weights = combine_race_weight_maps(recency_race_weights, era_race_weights)
    if train_recency_half_life is not None and recency_race_weights:
        vals = list(recency_race_weights.values())
        log(
            "Train recency weighting: "
            f"half_life={train_recency_half_life:g} | min={min(vals):.3f} | max={max(vals):.3f}"
        )
    if cfg.use_regulation_era_weights and era_race_weights:
        era_summary = summarize_era_weights(TR, train_race_weights)
        if era_summary:
            bits = [
                (
                    f"{row['era']} races={int(row['races'])} "
                    f"mean={float(row['weight_mean']):.3f} "
                    f"range=[{float(row['weight_min']):.3f},{float(row['weight_max']):.3f}]"
                )
                for row in era_summary
            ]
            log("Regulation-era weighting: " + " | ".join(bits))
    if train_race_weights:
        vals = list(train_race_weights.values())
        log(f"Combined race weights: min={min(vals):.3f} | max={max(vals):.3f}")
    scaler = fit_scaler_on_df(TR, feature_cols)

    TR_scaled = transform_with_scaler_df(TR, feature_cols, scaler, as_array=False).astype("float32")
    VA_scaled = transform_with_scaler_df(VA, feature_cols, scaler, as_array=False).astype("float32")

                                                                                              
    TRn = pd.concat([TR.drop(columns=feature_cols), TR_scaled], axis=1)
    VAn = pd.concat([VA.drop(columns=feature_cols), VA_scaled], axis=1)
    status_class_weights = _status_class_weights(TRn)

                                       
    tr_ds = RaceListDataset(
        TRn,
        feature_cols=feature_cols,
        dnf_position=cfg.dnf_position,
        dsq_position=cfg.dsq_position,
        race_weights=train_race_weights,
    )
    va_ds = RaceListDataset(
        VAn,
        feature_cols=feature_cols,
        dnf_position=cfg.dnf_position,
        dsq_position=cfg.dsq_position,
    )

                             
    model = make_mlp_ranker(
        in_dim=len(feature_cols),
        hidden=cfg.hidden,
        dropout=cfg.dropout,
        num_status_classes=len(OUTCOME_LABELS),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    log(f"Model params: {count_parameters(model):,} | in_dim={len(feature_cols)} | hidden={cfg.hidden} | drop={cfg.dropout}")
    log(f"Outcome classes: {list(OUTCOME_LABELS)} | class_weights={status_class_weights.tolist()}")

                 
    best = {"epoch": -1, "selection_score": -1e9, "val_spearman": -1e9, "meta": {}}
    try:
        for epoch in range(1, cfg.epochs + 1):
            train_loss = train_one_epoch(
                model,
                tr_ds,
                opt,
                device=device,
                status_loss_weight=cfg.status_loss_weight,
                status_class_weights=status_class_weights,
            )
            val = evaluate(model, va_ds, device=device)
            if (epoch % max(1, cfg.log_every)) == 0:
                log(format_metrics(epoch, train_loss, val["mean"]))

            selection_score = _selection_score(val["mean"])
            if selection_score > best["selection_score"]:
                best["epoch"] = epoch
                best["selection_score"] = float(selection_score)
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
                    "selection_metric": "spearman_plus_status_macro_f1",
                    "selection_score": float(selection_score),
                    "drop_prefixes": drop_prefixes,
                    "drop_contains": drop_contains,
                    "drop_cols": drop_cols,
                    "keep_prefixes": cfg.keep_prefixes or [],
                    "feature_profile": profile,
                    "target_mode": "multitask_finish_dnf_dsq",
                    "outcome_labels": list(OUTCOME_LABELS),
                    "num_status_classes": len(OUTCOME_LABELS),
                    "status_loss_weight": cfg.status_loss_weight,
                    "dnf_position": cfg.dnf_position,
                    "dsq_position": cfg.dsq_position,
                    "status_class_weights": status_class_weights.tolist(),
                    "use_regulation_era_weights": cfg.use_regulation_era_weights,
                    "era_weights": era_weight_overrides,
                    "scenario_mode": FUTURE_MODE if profile == FUTURE_FEATURE_PROFILE else "observed",
                    "profile_dropped_feature_cols": profile_dropped_cols,
                    "dropped_feature_cols": dropped_cols,
                    "train_recency_half_life": train_recency_half_life,
                    "train_race_weight_min": min(train_race_weights.values()) if train_race_weights else 1.0,
                    "train_race_weight_max": max(train_race_weights.values()) if train_race_weights else 1.0,
                    "train_recency_weight_min": min(recency_race_weights.values()) if recency_race_weights else 1.0,
                    "train_recency_weight_max": max(recency_race_weights.values()) if recency_race_weights else 1.0,
                    "train_era_weight_min": min(era_race_weights.values()) if era_race_weights else 1.0,
                    "train_era_weight_max": max(era_race_weights.values()) if era_race_weights else 1.0,
                    "train_era_weight_summary": summarize_era_weights(TR, train_race_weights),
                    "uses_relaxed_future_ablation": uses_relaxed_future_defaults,
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
            f"(epoch {best['epoch']}, selection {best['selection_score']:.4f}, spearman {best['val_spearman']:.4f})"
        )
    else:
        log("Could not improve the validation metric; artifacts were not saved.")


if __name__ == "__main__":
    main()
