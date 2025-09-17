#!/usr/bin/env python
from __future__ import annotations
import argparse, shutil
from pathlib import Path
from src.f1rank.io.paths import load_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--from", dest="from_dir", required=False, help="Папка с CSV (если не data/raw)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    src = Path(args.from_dir) if args.from_dir else cfg.paths.raw
    dst = cfg.paths.raw
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.glob("*.csv"):
        shutil.copy2(p, dst / p.name)
        print("copied:", p.name)
    print("done →", dst)
if __name__ == "__main__":
    main()
