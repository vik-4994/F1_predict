#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training import load_artifacts, sanitize_frame_columns, transform_with_scaler_df


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _group_for(col: str) -> str:
    if col.startswith("track_is_") or col.startswith("track_"):
        return "track"
    if col.startswith("weather_pre_"):
        return "weather"
    if col.startswith("hist_pre_"):
        return "history"
    if col.startswith("tele_pre_"):
        return "telemetry"
    if col.startswith("quali_pre_"):
        return "qualifying"
    if col.startswith(("expected_", "first_stint_", "undercut_", "overcut_")):
        return "strategy"
    if col.startswith(("compound_", "tyre_", "expected_deg_")):
        return "tyres"
    if col.startswith("reliab_"):
        return "reliability"
    if col.startswith(("pitcrew_", "slowstop_", "pitcrew_time_", "pit_ops_")):
        return "pit_ops"
    if col.startswith(("lap1_", "traffic_", "net_pass_")):
        return "traffic"
    if col.startswith(("driver_team_pre_", "driver_trackc_pre_")):
        return "driver_priors"
    return "other"


def inspect(artifacts_dir: Path, dump_path: Path, drivers_filter: List[str], topk: int, device: str = "cpu") -> None:
    arts = load_artifacts(artifacts_dir, device=device)
    if arts.model is None:
        raise RuntimeError("No model found in artifacts directory")

    df = sanitize_frame_columns(_read_table(dump_path))
    if df.empty:
        raise RuntimeError("Empty dump file")

    if drivers_filter:
        df = df[df["Driver"].astype(str).isin(drivers_filter)].copy()
        if df.empty:
            raise RuntimeError("No matching drivers in dump")

    df = df.reset_index(drop=True)
    X = transform_with_scaler_df(df, arts.feature_cols, arts.scaler, as_array=True).astype(np.float32, copy=False)

    model = arts.model
    model.eval()

    xt = torch.tensor(X, dtype=torch.float32, device=arts.device, requires_grad=True)
    with torch.enable_grad():
        scores = model(xt)
        if isinstance(scores, (list, tuple)):
            scores = scores[0]
        scores = scores.reshape(-1)

    probs = torch.softmax(scores.detach(), dim=0).cpu().numpy()
    table = (
        pd.DataFrame(
            {
                "Driver": df["Driver"].astype(str).to_numpy(),
                "score": scores.detach().cpu().numpy(),
                "win_prob_softmax": probs,
            }
        )
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

    print("\n=== RANKS ===")
    print(table.to_string(index=False, float_format=lambda x: f"{x: .6f}"))

    print("\n=== PER-DRIVER ATTRIBUTIONS (grad * input) ===")
    for i, driver in enumerate(df["Driver"].astype(str).tolist()):
        model.zero_grad(set_to_none=True)
        if xt.grad is not None:
            xt.grad.zero_()
        scores[i].backward(retain_graph=True)

        grad = xt.grad[i].detach().cpu().numpy()
        contrib = grad * X[i]
        pos_idx = np.argsort(-contrib)[:topk]
        neg_idx = np.argsort(contrib)[:topk]

        print(f"\n--- {driver} ---")
        print("Top + contributions:")
        for j in pos_idx:
            print(f"  {arts.feature_cols[j]:40s}  contrib={contrib[j]: .4f}   z={X[i, j]: .3f}")

        print("Top - contributions:")
        for j in neg_idx:
            print(f"  {arts.feature_cols[j]:40s}  contrib={contrib[j]: .4f}   z={X[i, j]: .3f}")

        group_sums: Dict[str, float] = {}
        for j, col in enumerate(arts.feature_cols):
            group = _group_for(col)
            group_sums[group] = group_sums.get(group, 0.0) + abs(float(contrib[j]))
        group_df = pd.DataFrame(
            sorted(group_sums.items(), key=lambda item: -item[1]),
            columns=["group", "|contrib|"],
        )
        print("\nGroup |contrib| sums:")
        print(group_df.to_string(index=False, float_format=lambda x: f"{x: .4f}"))


def main() -> None:
    ap = argparse.ArgumentParser("Inspect features dump with attributions")
    ap.add_argument("--artifacts", required=True, help="Папка с ranker.pt / scaler.json / feature_cols.txt")
    ap.add_argument("--dump", required=True, help="CSV/Parquet with feature rows")
    ap.add_argument("--drivers", default="", help="CSV of drivers to inspect; empty = all")
    ap.add_argument("--topk", type=int, default=12, help="How many positive/negative features to print")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    drivers_filter = [s.strip() for s in args.drivers.split(",") if s.strip()]
    try:
        inspect(Path(args.artifacts), Path(args.dump), drivers_filter, args.topk, device=args.device)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
