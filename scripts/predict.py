# scripts/predict.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch

pd.set_option("mode.copy_on_write", True)

# наш раннер инференса (см. .from_dir/.rank)
from src.training.inference import InferenceRunner  # :contentReference[oaicite:1]{index=1}

ALPHA_NUM = "abcdefghijklmnopqrstuvwxyz0123456789"

def _load_artifact_scaler(artifacts_dir: Path):
    """Пытаемся достать mean/std из артефактов (scaler.json)."""
    import json
    sj = Path(artifacts_dir) / "scaler.json"
    if not sj.exists(): return None
    with open(sj, "r") as f:
        obj = json.load(f)
    # ожидаем форматы {'mean': [...], 'std': [...], 'feature_cols': [...] } или отдельно feature_cols.txt
    return obj

def _to_tensor(X: np.ndarray, device=None):
    if device is None:
        device = "cpu"
    return torch.tensor(X, dtype=torch.float32, device=device, requires_grad=True)



def _explain_drivers(runner, artifacts_dir: Path, df: pd.DataFrame,
                     drivers: List[str], temperature: float = 1.2, topn: int = 10):
    """
    Печатает по каждому драйверу:
      [A] top |z-score| признаков (после стандартизации скейлером артефактов)
      [B] top grad*input (локальная важность)
      [C] top Δscore, если фичу прибить к среднему (what-if)
    """
    import sys
    import numpy as np
    try:
        import torch
    except Exception:
        print("[warn] PyTorch unavailable; --explain disabled", file=sys.stderr)
        return

    from src.training.featureset import transform_with_scaler_df  # тот же трансформер, что в инференсе

    # 1) порядок признаков берём из раннера/артефактов
    feat_cols = list(getattr(runner, "feature_columns", runner.artifacts.feature_cols))  # :contentReference[oaicite:1]{index=1}

    # 2) берём строки по нужным драйверам в заданном порядке
    present = set(df["Driver"].astype(str))
    drivers = [d for d in drivers if d in present]
    if not drivers:
        print("[warn] --explain: ни одного из запрошенных драйверов нет в DF", file=sys.stderr)
        return
    sub = df.set_index("Driver").loc[drivers, :].copy()

    # 3) стандартизируем ровно как в инференсе
    X = transform_with_scaler_df(sub, feat_cols, runner.artifacts.scaler, as_array=True).astype(np.float32, copy=False)  # :contentReference[oaicite:2]{index=2}

    # 4) модель берём из артефактов раннера
    model = runner.artifacts.model  # <-- ключевая правка :contentReference[oaicite:3]{index=3}
    if model is None:
        print("[warn] No model in artifacts; cannot compute explanations.", file=sys.stderr)
        return
    model.eval()

    device = next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else torch.device("cpu")
    xt = _to_tensor(X, device=device)  # [D,F], requires_grad=True

    with torch.enable_grad():
        scores = model(xt)  # [D]

    # 5) градиенты и вклад grad*input
    grads = []
    for i in range(xt.shape[0]):
        model.zero_grad(set_to_none=True)
        if xt.grad is not None:
            xt.grad.zero_()
        scores[i].backward(retain_graph=True)
        grads.append(xt.grad[i].detach().cpu().numpy().copy())
    grad = np.vstack(grads)                              # [D,F]
    contrib = grad * xt.detach().cpu().numpy()           # grad*input (в стандартизированном пространстве)

    # 6) what-if: прибиваем фичу к среднему (z=0) и смотрим Δscore
    with torch.no_grad():
        base = model(xt).detach().cpu().numpy()          # [D]
    deltas = np.zeros_like(contrib, dtype=np.float32)
    for j in range(X.shape[1]):
        X_mut = xt.detach().clone()
        X_mut[:, j] = 0.0                                # 0 ⇒ (value-mean)/std = 0
        with torch.no_grad():
            s_mut = model(X_mut).detach().cpu().numpy()
        deltas[:, j] = base - s_mut

    # 7) печать топов
    for idx, drv in enumerate(drivers):
        print(f"\n=== EXPLAIN {drv} ===")
        z = xt[idx].detach().cpu().numpy()
        topn_eff = min(topn, len(feat_cols))

        # [A] |z|
        topz = np.argsort(-np.abs(z))[:topn_eff]
        print("[A] top |z-score| features")
        for j in topz:
            print(f"  {feat_cols[j]:40s}  z={z[j]:+7.3f}")

        # [B] grad*input
        c = contrib[idx]
        topc = np.argsort(-np.abs(c))[:topn_eff]
        print("[B] top grad*input (локальная важность)")
        for j in topc:
            print(f"  {feat_cols[j]:40s}  g*x={c[j]:+9.5f}")

        # [C] Δscore при приведении к среднему
        d = deltas[idx]
        topd = np.argsort(-np.abs(d))[:topn_eff]
        print("[C] top Δscore if set to mean")
        for j in topd:
            print(f"  {feat_cols[j]:40s}  Δscore={d[j]:+9.5f}")



def _norm(s: str) -> str:
    s = s.strip().lower()
    out, prev_us = [], False
    for ch in s:
        if ch in ALPHA_NUM:
            out.append(ch); prev_us = False
        else:
            if not prev_us: out.append("_"); prev_us = True
    if out and out[-1] == "_": out.pop()
    return "".join(out)

def _find_col(all_cols: List[str], aliases: List[str]) -> Optional[str]:
    cols_norm = {c: _norm(c) for c in all_cols}
    norm2col = {v: k for k, v in cols_norm.items()}
    for a in aliases:
        na = _norm(a)
        if na in norm2col: return norm2col[na]
    for a in aliases:
        na = _norm(a)
        for c, nc in cols_norm.items():
            if na in nc: return c
    return None

WEATHER_MAP: Dict[str, List[str]] = {
    "AirTemp": ["weather_air_temp","air_temp","air_temp_c","air_temperature_c","met_air_c","weather::air_temp"],
    "TrackTemp": ["weather_track_temp","track_temp","track_temp_c","track_temperature_c","weather::track_temp"],
    "Humidity": ["weather_humidity","rel_humidity","relative_humidity","humidity"],
    "WindSpeed": ["weather_wind_speed","wind_speed","wind_speed_mps","wind_speed_kph","wind_mps","wind_kph"],
    "WindDirection": ["weather_wind_dir","wind_direction","wind_direction_deg","wind_dir"],
}
GRID_ALIASES = ["grid","grid_pos","grid_position","start_grid","start_grid_pos","starting_grid","start_position","quali_grid","quali_grid_pos"]

def _onehot_track_columns(cols: List[str]) -> List[str]:
    return [c for c in cols if c.startswith(("trk::","track::","track_is_"))]

def _resolve_track_onehot_col(track_onehots: List[str], track_name: str) -> Optional[str]:
    import difflib
    base2col = {}
    for c in track_onehots:
        base = c
        for pfx in ("trk::","track::","track_is_"):
            if base.startswith(pfx): base = base[len(pfx):]
        base2col[_norm(base)] = c
    target = _norm(track_name)
    if target in base2col: return base2col[target]
    t2 = _norm("is_" + track_name)
    if t2 in base2col: return base2col[t2]
    best, score = None, 0.0
    for b in base2col:
        r = difflib.SequenceMatcher(a=b, b=target).ratio()
        if r > score: best, score = b, r
    return base2col[best] if best and score >= 0.80 else None

def fmt_percent(x: float) -> str: return f"{x * 100:.1f}%"

def pretty_print_table(df: pd.DataFrame, topk: int, cols_mode: str = "mini") -> None:
    show = df.iloc[:topk].copy()
    if cols_mode == "mini":
        cols = [c for c in ("rank","Driver","year","round","score","p_win") if c in show.columns]
        show = show[cols].rename(columns={"rank":"pos"})
        show["p_win_%"] = show["p_win"].astype(float).map(fmt_percent)
        show = show.drop(columns=["p_win"])
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(" pos Driver  year  round     score p_win_%")
        for _, r in show.iterrows():
            print(f"{int(r['pos']):4d} {str(r['Driver']):>5s}  {int(r['year']):4d} {int(r['round']):6d}  "
                  f"{float(r['score']):10.6f}   {str(r['p_win_%']):>5s}")

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser("Predict race ranking (custom mode)")
    ap.add_argument("--artifacts", type=str, required=True)
    ap.add_argument("--features", type=str, required=True)
    ap.add_argument("--mode", type=str, default="custom")
    ap.add_argument("--drivers", type=str, required=True, help='CSV "VER,NOR,..."')
    ap.add_argument("--track", type=str, required=True)
    ap.add_argument("--weather-json", type=str, default="{}", help='{"AirTemp":27,...}')
    ap.add_argument("--history-window", type=int, default=3)
    ap.add_argument("--sim-year", type=int, required=True)
    ap.add_argument("--sim-round", type=int, required=True)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--cols", type=str, default="mini", choices=["mini","full"])
    ap.add_argument("--grid", type=str, default=None, help='e.g. "VER=1,NOR=2,..."')
    ap.add_argument("--tau", type=float, default=1.2, help="softmax temperature")
    ap.add_argument("--explain", type=str, default=None, help='CSV драйверов для объяснения, напр. "VER,ALB"')
    return ap.parse_args(argv)

def _parse_grid(s: Optional[str]) -> Dict[str, float]:
    if not s: return {}
    out = {}
    for part in s.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            try: out[k.strip()] = float(v)
            except ValueError: pass
    return out

def _apply_weather(df: pd.DataFrame, weather_json: Dict[str, float]) -> None:
    if not weather_json: return
    cols = list(df.columns)
    for key, aliases in WEATHER_MAP.items():
        if key not in weather_json: continue
        col = _find_col(cols, aliases)
        if col is not None:
            df.loc[:, col] = np.float32(weather_json[key])
        else:
            print(f"[warn] weather key '{key}' not mapped to any feature column", file=sys.stderr)

def _apply_track_onehot(df: pd.DataFrame, track_name: str) -> None:
    onehots = _onehot_track_columns(list(df.columns))
    if not onehots:
        print("[warn] no track one-hot columns found", file=sys.stderr); return
    df.loc[:, onehots] = np.float32(0.0)
    col = _resolve_track_onehot_col(onehots, track_name)
    if col is None:
        print(f"[warn] track '{track_name}' did not match any one-hot; leaving all zeros", file=sys.stderr); return
    df.loc[:, col] = np.float32(1.0)

def _apply_grid(df: pd.DataFrame, grid_map: Dict[str, float]) -> None:
    if not grid_map: return
    col = _find_col(list(df.columns), GRID_ALIASES)
    if col is None:
        print("[warn] no grid column found to set start positions", file=sys.stderr); return
    for drv, pos in grid_map.items():
        m = (df["Driver"] == drv)
        if m.any(): df.loc[m, col] = np.float32(pos)
        else: print(f"[warn] driver '{drv}' not present in current DF; grid ignored", file=sys.stderr)

def _astype_floatwise(df: pd.DataFrame) -> None:
    # Приводим только плавающие к float32; int оставляем как int (исключит предупреждения)
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df.loc[:, c] = df[c].astype(np.float32, copy=False)
        # pandas 'boolean' → float32 при необходимости
        elif str(df[c].dtype) == "boolean":
            df.loc[:, c] = df[c].astype(np.float32, copy=False)

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    features_pq = Path(args.features)

    # 1) читаем фичи
    base_df = pd.read_parquet(features_pq)

    # 2) фильтр по драйверам
    drivers = [d.strip() for d in args.drivers.split(",") if d.strip()]
    df = base_df.loc[base_df["Driver"].isin(drivers)].reset_index(drop=True).copy()

    # 3) схлопываем до одной строки на пилота (берём самую свежую запись по year,round)
    if "year" in df.columns and "round" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["round"] = pd.to_numeric(df["round"], errors="coerce")
        df = (df.sort_values(["Driver","year","round"])
                .groupby("Driver", as_index=False, sort=False)
                .tail(1)
                .reset_index(drop=True))
    else:
        df = df.drop_duplicates("Driver", keep="last").reset_index(drop=True)

    # 4) выставляем target гонку (типы оставляем int)
    if "year" in df.columns:  df.loc[:, "year"]  = np.int32(args.sim_year)
    if "round" in df.columns: df.loc[:, "round"] = np.int32(args.sim_round)

    # 5) трек/погода/грид
    _apply_track_onehot(df, args.track)
    _apply_weather(df, json.loads(args.weather_json) if args.weather_json else {})
    _apply_grid(df, _parse_grid(args.grid))

    # 6) типы: только float-колонки → float32 (не трогаем year/round)
    _astype_floatwise(df)
    df = df.copy()  # дефрагментация

    # 7) инференс: используем корректный API раннера
    runner = InferenceRunner.from_dir(args.artifacts)  # :contentReference[oaicite:2]{index=2}
    rank_df = runner.rank(
        df,
        temperature=float(args.tau),
        by=("year","round"),         # группировка по гонке, softmax и ранги внутри (см. inference.rank) :contentReference[oaicite:3]{index=3}
        include_probs=True,
        ascending=False
    )

    # 8) печать (используем готовые колонки rank/score/p_win)
    # гарантируем человекочитаемый вывод
    pretty_print_table(rank_df, topk=int(args.topk), cols_mode=args.cols)

    if args.explain:
        exp_list = [s.strip() for s in args.explain.split(",") if s.strip()]
        # оставим только тех, кто есть в df
        exp_list = [d for d in exp_list if d in set(df["Driver"].tolist())]
        if exp_list:
            _explain_drivers(runner, Path(args.artifacts), df, exp_list, temperature=float(args.tau), topn=10)
        else:
            print("[warn] none of --explain drivers found in DF", file=sys.stderr)

if __name__ == "__main__":
    main()
