#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import OrdinalEncoder
import joblib

from src.f1rank.io.paths import load_config
from src.f1rank.modeling.ranker import RankerWrapper, group_by_race

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model/lgbm_ranker.yaml")
    ap.add_argument("--base", default="configs/base.yaml")
    args = ap.parse_args()

    base = load_config(args.base)
    import yaml
    params = yaml.safe_load(Path(args.config).read_text()).get("params", {})
    fdir = base.paths.features
    train = pd.read_parquet(fdir / "train.parquet")
    valid = pd.read_parquet(fdir / "valid.parquet")
    # Feature set
    ycol = "finish_pos"
    drop_cols = ["finish_pos","race_ts","status"]
    feat_cols = [c for c in train.columns if c not in drop_cols]
    # Simple numeric/cat split
    Xtr = train[feat_cols].copy()
    Xva = valid[feat_cols].copy()
    ytr = train[ycol].astype(float).values
    yva = valid[ycol].astype(float).values
    # Group sizes for ranker
    gtr = group_by_race(train, "raceId")
    gva = group_by_race(valid, "raceId")

    # Impute numeric NaNs with medians
    medians = {c: float(Xtr[c].median()) for c in Xtr.columns if str(Xtr[c].dtype).startswith(("float","int"))}
    for df in (Xtr, Xva):
        for c, m in medians.items():
            if c in df.columns:
                df[c] = df[c].fillna(m)

    model = RankerWrapper(params=params, objective="lambdarank")
    model.fit(Xtr, ytr, groups=gtr)

    # Save artifacts
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base.paths.artifacts / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model.model, "feature_cols": feat_cols, "impute_medians": medians}, run_dir / "model.pkl")
    schema = {
        "feature_cols": feat_cols,
        "rows": {"train": int(len(train)), "valid": int(len(valid))},
        "split_years": [int(train["year"].min()), int(valid["year"].max())],
        "params": params,
    }
    (run_dir / "schema.json").write_text(json.dumps(schema, indent=2))
    (base.paths.artifacts / "latest").symlink_to(run_dir, target_is_directory=True)
    print("Saved run at:", run_dir)

if __name__ == "__main__":
    main()
