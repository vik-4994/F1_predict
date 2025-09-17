#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import joblib

from src.f1rank.io.paths import load_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="latest")
    ap.add_argument("--base", default="configs/base.yaml")
    ap.add_argument("--race-id", type=int, required=True)
    args = ap.parse_args()

    cfg = load_config(args.base)
    run_dir = cfg.paths.artifacts / args.run
    if args.run == "latest":
        run_dir = (cfg.paths.artifacts / "latest").resolve()
    bundle = joblib.load(run_dir / "model.pkl")
    feat_dir = cfg.paths.features
    import pandas as pd
    for split in ["train","valid","test"]:
        fp = feat_dir / f"{split}.parquet"
        if fp.exists():
            df = pd.read_parquet(fp)
            sub = df[df["raceId"]==args.race_id].copy()
            if len(sub):
                X = sub[bundle["feature_cols"]].copy()
                for c, m in bundle.get("impute_medians", {}).items():
                    if c in X.columns:
                        X[c] = X[c].fillna(m)
                sub["pred"] = bundle["model"].predict(X)
                out = sub.sort_values("pred")[["driverId","constructorId","grid","pred"]]
                print(out.to_json(orient="records", force_ascii=False, indent=2))
                return
    print("raceId not found in features")

if __name__ == "__main__":
    main()
