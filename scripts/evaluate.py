#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import joblib

from src.f1rank.io.paths import load_config
from src.f1rank.evaluation.metrics import racewise_metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="latest")
    ap.add_argument("--base", default="configs/base.yaml")
    args = ap.parse_args()

    cfg = load_config(args.base)
    run_dir = cfg.paths.artifacts / args.run
    if args.run == "latest":
        run_dir = (cfg.paths.artifacts / "latest").resolve()
    bundle = joblib.load(run_dir / "model.pkl")
    feat_dir = cfg.paths.features
    test = pd.read_parquet(feat_dir / "test.parquet")
    Xte = test[bundle["feature_cols"]].copy()
    for c, m in bundle.get("impute_medians", {}).items():
        if c in Xte.columns:
            Xte[c] = Xte[c].fillna(m)
    test["pred"] = bundle["model"].predict(Xte)
    metrics = racewise_metrics(test, "pred", k_list=(3,5))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
