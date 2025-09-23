#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import torch

# ----- utils -----

def read_feature_cols(artifacts: Path) -> List[str]:
    p = artifacts / "feature_cols.txt"
    cols = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return cols

def read_scaler(artifacts: Path) -> Dict[str, List[float]]:
    p = artifacts / "scaler.json"
    return json.loads(p.read_text(encoding="utf-8"))

def load_model(artifacts: Path, device: str = "cpu") -> torch.nn.Module:
    # универсальный загрузчик из src.training.inference
    sys.path.insert(0, str(Path.cwd()))
    from src.training.inference import load_artifacts
    model, scaler, feature_cols, meta, device_used = load_artifacts(artifacts, device=device)
    model.eval()
    return model

def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def _build_X(df: pd.DataFrame, feature_cols: List[str], scaler: Dict[str, List[float]]) -> Tuple[np.ndarray, List[str]]:
    # гарантируем наличие всех столбцов по порядку обучающего сета
    out = []
    for c in feature_cols:
        if c in df.columns:
            out.append(df[c].values)
        else:
            out.append(np.full(len(df), np.nan, dtype=float))
    X = np.vstack(out).T.astype(float)

    # Импьютация медиан (из train-скейлера), нормализация и клиппинг как в проде
    mu = np.asarray(scaler["mu"], dtype=float)
    sigma = np.asarray(scaler["sigma"], dtype=float)
    med = np.asarray(scaler["med"], dtype=float)

    # nan -> med
    nan_mask = ~np.isfinite(X)
    if nan_mask.any():
        X[nan_mask] = np.take(med, np.where(nan_mask)[1])

    Z = (X - mu) / np.where(sigma == 0.0, 1.0, sigma)
    Z = np.clip(Z, -8.0, 8.0)
    return Z, feature_cols

def _groups_for(col: str) -> str:
    # группируем по источнику фичи/модулю
    if col.startswith("track_is_"): return "track_onehot"
    if col.startswith("weather_"): return "weather_basic"
    if col.startswith("history_"): return "history_form"
    if col.startswith("telemetry_"): return "telemetry_history_pre"
    if col.startswith("quali_pre_"): return "quali_priors_pre"
    if col.startswith("strategy_pre_"): return "strategy_priors_pre"
    if col.startswith("tyre_pre_"): return "tyre_priors_pre"
    if col.startswith("dev_trend_pre_"): return "dev_trend_pre"
    if col.startswith("reliability_risk_pre_"): return "reliability_risk_pre"
    if col.startswith("pit_ops_risk_pre_"): return "pit_ops_risk_pre"
    if col.startswith("traffic_overtake_pre_"): return "traffic_overtake_pre"
    if col.startswith("driver_team_pre_"): return "driver_team_priors_pre"
    if col.startswith("pit_ops_pre_"): return "pit_ops_pre"
    return "other"

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    ex = np.exp(x)
    s = ex.sum()
    return ex / s if s > 0 else np.full_like(ex, 1.0 / len(ex))

# ----- main logic -----

def inspect(artifacts: Path, dump_path: Path, drivers_filter: List[str], topk: int, device: str = "cpu"):
    df = _read_table(dump_path)
    if df.empty:
        print("Empty dump file", file=sys.stderr); sys.exit(2)

    # фильтруем выбранных пилотов (если указан список)
    if drivers_filter:
        df = df[df["Driver"].astype(str).isin(drivers_filter)].copy()
        if df.empty:
            print("No matching drivers in dump", file=sys.stderr); sys.exit(3)

    feature_cols = read_feature_cols(artifacts)
    scaler = read_scaler(artifacts)

    # Матрица нормализованных фич Z, порядок как в train
    Z, cols = _build_X(df, feature_cols, scaler)

    # torch-предикт и атрибуции grad*input
    model = load_model(artifacts, device=device)
    tZ = torch.tensor(Z, dtype=torch.float32, requires_grad=True)
    scores = model(tZ).squeeze(-1)  # (N,)
    # убеждаемся в порядке:
    order = torch.argsort(scores, descending=True).detach().cpu().numpy()
    probs = _softmax(scores.detach().cpu().numpy())

    # печать таблицы рангов
    print("\n=== RANKS ===")
    tab = (pd.DataFrame({
        "Driver": df["Driver"].values,
        "score": scores.detach().cpu().numpy(),
        "win_prob_softmax": probs,
    }).sort_values("score", ascending=False).reset_index(drop=True))
    print(tab.to_string(index=False, float_format=lambda x: f"{x: .6f}"))

    # атрибуции для выбранных пилотов
    print("\n=== PER-DRIVER ATTRIBUTIONS (grad * input) ===")
    for i in range(len(df)):
        # если указан фильтр и пилота нет в нём — пропускаем
        if drivers_filter and str(df.iloc[i]["Driver"]) not in drivers_filter:
            continue

        model.zero_grad(set_to_none=True)
        if tZ.grad is not None:
            tZ.grad.zero_()
        scores[i].backward(retain_graph=True)
        grad = tZ.grad[i].detach().cpu().numpy()  # (D,)
        contrib = grad * Z[i]  # grad * input (у нас input уже нормализован)
        # топ + / -
        pos_idx = np.argsort(-contrib)[:topk]
        neg_idx = np.argsort(contrib)[:topk]

        print(f"\n--- {df.iloc[i]['Driver']} ---")
        print("Top + contributions:")
        for j in pos_idx:
            print(f"  {cols[j]:40s}  contrib={contrib[j]: .4f}   z={Z[i,j]: .3f}")

        print("Top - contributions:")
        for j in neg_idx:
            print(f"  {cols[j]:40s}  contrib={contrib[j]: .4f}   z={Z[i,j]: .3f}")

        # групповые вклады
        group_sums: Dict[str, float] = {}
        for j, c in enumerate(cols):
            g = _groups_for(c)
            group_sums[g] = group_sums.get(g, 0.0) + abs(contrib[j])
        group_df = pd.DataFrame(sorted(group_sums.items(), key=lambda x: -x[1]), columns=["group","|contrib|"])
        print("\nGroup |contrib| sums:")
        print(group_df.to_string(index=False, float_format=lambda x: f"{x: .4f}"))

def main():
    ap = argparse.ArgumentParser("Inspect features dump with attributions")
    ap.add_argument("--artifacts", required=True, help="Папка с ranker.pt / scaler.json / feature_cols.txt")
    ap.add_argument("--dump", required=True, help="CSV/Parquet из --dump-features")
    ap.add_argument("--drivers", default="", help="Кого печатать (пример: 'ALB,NOR,VER,TSU'); пусто = все из дампа")
    ap.add_argument("--topk", type=int, default=12, help="Сколько фич показывать в топах +/-")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    drivers_filter = [s.strip() for s in args.drivers.split(",") if s.strip()]
    inspect(Path(args.artifacts), Path(args.dump), drivers_filter, args.topk, device=args.device)

if __name__ == "__main__":
    main()
