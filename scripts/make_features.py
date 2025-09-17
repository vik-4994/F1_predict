#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

from src.f1rank.io.paths import load_config
from src.f1rank.data.loading import load_raw
from src.f1rank.targets.results_target import build_target_from_results
from src.f1rank.features.qualifying import make_qualifying_features
from src.f1rank.features.rolling import driver_constructor_rollups

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg.paths.features.mkdir(parents=True, exist_ok=True)
    dfs = load_raw(cfg.paths)
    target = build_target_from_results(dfs["results"], dfs["status"], dfs["races"], dfs["qualifying"], cfg.target)
    q_feats = make_qualifying_features(dfs["qualifying"])
    drv_roll, con_roll = driver_constructor_rollups(target, q_feats)

    feat = target.merge(q_feats[["raceId","driverId","best_q_ms","constructorId"]], on=["raceId","driverId"], how="left").merge(drv_roll, on=["raceId","driverId"], how="left").merge(con_roll, on=["raceId","driverId"], how="left")

    if "constructorId" not in feat.columns:
        cx, cy = "constructorId_x", "constructorId_y"
        if cx in feat.columns or cy in feat.columns:
            feat["constructorId"] = feat.get(cx)
            if cy in feat.columns:
                feat["constructorId"] = feat["constructorId"].fillna(feat[cy])
            
            feat.drop(columns=[c for c in (cx, cy) if c in feat.columns], inplace=True)

    cat_cols = ["constructorId"]
    feat[cat_cols] = feat[cat_cols].astype("category")

    y_max = int(feat["year"].max())
    split1 = y_max - int(cfg.split.get("valid_last_years", 3)) - int(cfg.split.get("test_last_years", 2))
    split2 = y_max - int(cfg.split.get("test_last_years", 2))
    feat["split"] = np.select(
        [feat["year"] <= split1, (feat["year"] > split1) & (feat["year"] <= split2), feat["year"] > split2],
        ["train","valid","test"],
        "train"
    )
    for name in ["train","valid","test"]:
        out = feat.loc[feat["split"]==name].drop(columns=["split"]).reset_index(drop=True)
        out.to_parquet(cfg.paths.features / f"{name}.parquet")
        print(name, "→", len(out))
    print("Done. Features at:", cfg.paths.features)

if __name__ == "__main__":
    main()
