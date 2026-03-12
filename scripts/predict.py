from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import featurize_pre
from src.training import InferenceRunner, sanitize_frame_columns, transform_with_scaler_df

pd.set_option("mode.copy_on_write", True)

ALPHA_NUM = "abcdefghijklmnopqrstuvwxyz0123456789"
WEATHER_MAP: Dict[str, List[str]] = {
    "AirTemp": ["weather_air_temp", "air_temp", "air_temp_c", "air_temperature_c", "met_air_c", "weather::air_temp"],
    "TrackTemp": ["weather_track_temp", "track_temp", "track_temp_c", "track_temperature_c", "weather::track_temp"],
    "Humidity": ["weather_humidity", "rel_humidity", "relative_humidity", "humidity"],
    "WindSpeed": ["weather_wind_speed", "wind_speed", "wind_speed_mps", "wind_speed_kph", "wind_mps", "wind_kph"],
    "WindDirection": ["weather_wind_dir", "wind_direction", "wind_direction_deg", "wind_dir"],
}
GRID_ALIASES = ["grid", "grid_pos", "grid_position", "start_grid", "start_grid_pos", "starting_grid", "start_position", "quali_grid", "quali_grid_pos"]


def _to_tensor(X: np.ndarray, device: str = "cpu") -> torch.Tensor:
    return torch.tensor(X, dtype=torch.float32, device=device, requires_grad=True)


def _softmax_stable(x: np.ndarray, tau: float) -> np.ndarray:
    if tau <= 0:
        tau = 1.0
    m = np.max(x)
    e = np.exp((x - m) / tau)
    s = e.sum()
    return e / s if s > 0 else np.full_like(e, 1.0 / len(e))


def _norm(s: str) -> str:
    s = s.strip().lower()
    out, prev_us = [], False
    for ch in s:
        if ch in ALPHA_NUM:
            out.append(ch)
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    if out and out[-1] == "_":
        out.pop()
    return "".join(out)


def _find_col(all_cols: List[str], aliases: List[str]) -> Optional[str]:
    cols_norm = {c: _norm(c) for c in all_cols}
    norm2col = {v: k for k, v in cols_norm.items()}
    for alias in aliases:
        norm_alias = _norm(alias)
        if norm_alias in norm2col:
            return norm2col[norm_alias]
    for alias in aliases:
        norm_alias = _norm(alias)
        for col, norm_col in cols_norm.items():
            if norm_alias in norm_col:
                return col
    return None


def _onehot_track_columns(cols: List[str]) -> List[str]:
    return [c for c in cols if c.startswith(("trk::", "track::", "track_is_"))]


def _resolve_track_onehot_col(track_onehots: List[str], track_name: str) -> Optional[str]:
    import difflib

    base2col = {}
    for col in track_onehots:
        base = col
        for prefix in ("trk::", "track::", "track_is_"):
            if base.startswith(prefix):
                base = base[len(prefix):]
        base2col[_norm(base)] = col

    target = _norm(track_name)
    if target in base2col:
        return base2col[target]

    best, score = None, 0.0
    for base in base2col:
        ratio = difflib.SequenceMatcher(a=base, b=target).ratio()
        if ratio > score:
            best, score = base, ratio
    return base2col[best] if best and score >= 0.80 else None


def fmt_percent(x: float) -> str:
    return f"{x * 100:.1f}%"


def pretty_print_table(df: pd.DataFrame, topk: int, cols_mode: str = "mini") -> None:
    show = df.iloc[:topk].copy()
    if cols_mode == "mini":
        cols = [c for c in ("rank", "Driver", "year", "round", "score", "p_win") if c in show.columns]
        show = show[cols].rename(columns={"rank": "pos"})
        show["p_win_%"] = show["p_win"].astype(float).map(fmt_percent)
        show = show.drop(columns=["p_win"])
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(" pos Driver  year  round     score p_win_%")
        for _, row in show.iterrows():
            print(
                f"{int(row['pos']):4d} {str(row['Driver']):>5s}  {int(row['year']):4d} {int(row['round']):6d}  "
                f"{float(row['score']):10.6f}   {str(row['p_win_%']):>5s}"
            )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser("Predict race ranking")
    ap.add_argument("--artifacts", type=str, required=True)
    ap.add_argument("--features", type=str, default=None, help="Fallback legacy features table")
    ap.add_argument("--raw-dir", type=str, default=None, help="Raw CSV directory to rebuild pre-race features for the target event")
    ap.add_argument("--mode", type=str, default="custom")
    ap.add_argument("--drivers", type=str, required=True, help='CSV "VER,NOR,..."')
    ap.add_argument("--track", type=str, required=True)
    ap.add_argument("--weather-json", type=str, default="{}", help='{"AirTemp":27,...}')
    ap.add_argument("--history-window", type=int, default=3)
    ap.add_argument("--sim-year", type=int, required=True)
    ap.add_argument("--sim-round", type=int, required=True)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--cols", type=str, default="mini", choices=["mini", "full"])
    ap.add_argument("--grid", type=str, default=None, help='e.g. "VER=1,NOR=2,..."')
    ap.add_argument("--tau", type=float, default=1.2, help="softmax temperature")
    ap.add_argument("--grid-weight", type=float, default=0.0, help="penalize score by start grid (0=off)")
    ap.add_argument("--explain", type=str, default=None, help='CSV drivers for explanation, e.g. "VER,ALB"')
    return ap.parse_args(argv)


def _parse_grid(s: Optional[str]) -> Dict[str, float]:
    if not s:
        return {}
    out = {}
    for part in s.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        try:
            out[key.strip()] = float(value)
        except ValueError:
            continue
    return out


def _apply_weather(df: pd.DataFrame, weather_json: Dict[str, float]) -> None:
    if not weather_json:
        return
    cols = list(df.columns)
    for key, aliases in WEATHER_MAP.items():
        if key not in weather_json:
            continue
        col = _find_col(cols, aliases)
        if col is not None:
            df.loc[:, col] = np.float32(weather_json[key])
        else:
            print(f"[warn] weather key '{key}' not mapped to any feature column", file=sys.stderr)


def _apply_track_onehot(df: pd.DataFrame, track_name: str) -> None:
    onehots = _onehot_track_columns(list(df.columns))
    if not onehots:
        print("[warn] no track one-hot columns found", file=sys.stderr)
        return
    df.loc[:, onehots] = np.float32(0.0)
    col = _resolve_track_onehot_col(onehots, track_name)
    if col is None:
        print(f"[warn] track '{track_name}' did not match any one-hot; leaving all zeros", file=sys.stderr)
        return
    df.loc[:, col] = np.float32(1.0)


def _apply_grid(df: pd.DataFrame, grid_map: Dict[str, float]) -> None:
    if not grid_map:
        return
    col = _find_col(list(df.columns), GRID_ALIASES)
    if col is None:
        print("[warn] no grid column found to set start positions", file=sys.stderr)
        return
    for drv, pos in grid_map.items():
        mask = df["Driver"] == drv
        if bool(mask.any()):
            df.loc[mask, col] = np.float32(pos)
        else:
            print(f"[warn] driver '{drv}' not present in current DF; grid ignored", file=sys.stderr)


def _astype_floatwise(df: pd.DataFrame) -> None:
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df.loc[:, col] = df[col].astype(np.float32, copy=False)
        elif str(df[col].dtype) == "boolean":
            df.loc[:, col] = df[col].astype(np.float32, copy=False)


def _build_grid_pos_map(df_features: pd.DataFrame, user_grid_map: Dict[str, float]) -> Dict[str, float]:
    if user_grid_map:
        return {str(k): float(v) for k, v in user_grid_map.items()}
    col = _find_col(list(df_features.columns), GRID_ALIASES)
    if col is None:
        return {}
    mapping_df = df_features[["Driver", col]].dropna().copy()
    mapping_df[col] = pd.to_numeric(mapping_df[col], errors="coerce")
    mapping_df = mapping_df.dropna()
    return {str(row["Driver"]): float(row[col]) for _, row in mapping_df.iterrows()}


def _apply_grid_weight(
    rank_df: pd.DataFrame,
    df_features: pd.DataFrame,
    user_grid_map: Dict[str, float],
    grid_weight: float,
    tau: float,
    groupby: Sequence[str] = ("year", "round"),
    beta: float = 0.25,
) -> pd.DataFrame:
    if grid_weight <= 0:
        return rank_df

    grid_map = _build_grid_pos_map(df_features, user_grid_map)
    if not grid_map:
        return rank_df

    df = rank_df.copy()
    df["__grid_pos__"] = df["Driver"].map(grid_map).fillna(10.0)
    df["score"] = df["score"] - float(grid_weight) * float(beta) * (df["__grid_pos__"] - 1.0)

    gb = list(groupby) if groupby else []
    if gb and all(col in df.columns for col in gb):
        def _recalc(group: pd.DataFrame) -> pd.DataFrame:
            group = group.copy()
            p = _softmax_stable(group["score"].to_numpy(dtype=np.float64), tau=float(tau))
            group["p_win"] = p.astype(np.float32)
            group = group.sort_values("score", ascending=False)
            group["rank"] = np.arange(1, len(group) + 1, dtype=np.int32)
            return group

        df = df.groupby(gb, group_keys=False).apply(_recalc)
    else:
        p = _softmax_stable(df["score"].to_numpy(dtype=np.float64), tau=float(tau))
        df["p_win"] = p.astype(np.float32)
        df = df.sort_values("score", ascending=False)
        df["rank"] = np.arange(1, len(df) + 1, dtype=np.int32)

    return df.drop(columns=["__grid_pos__"]).reset_index(drop=True)


def _ordered_driver_slice(df: pd.DataFrame, drivers: List[str]) -> pd.DataFrame:
    order = {drv: idx for idx, drv in enumerate(drivers)}
    out = df[df["Driver"].astype(str).isin(drivers)].copy()
    out["__order__"] = out["Driver"].astype(str).map(order)
    out = out.sort_values("__order__", kind="mergesort").drop_duplicates("Driver", keep="first")
    return out.drop(columns=["__order__"]).reset_index(drop=True)


def _build_legacy_features(features_path: Path, drivers: List[str], sim_year: int, sim_round: int) -> pd.DataFrame:
    base_df = sanitize_frame_columns(pd.read_parquet(features_path))
    df = _ordered_driver_slice(base_df, drivers)
    if df.empty:
        raise RuntimeError("No matching drivers found in features table")

    if "year" in df.columns and "round" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["round"] = pd.to_numeric(df["round"], errors="coerce")
        df = (
            df.sort_values(["Driver", "year", "round"])
            .groupby("Driver", as_index=False, sort=False)
            .tail(1)
            .reset_index(drop=True)
        )

    if "year" in df.columns:
        df.loc[:, "year"] = np.int32(sim_year)
    else:
        df["year"] = np.int32(sim_year)
    if "round" in df.columns:
        df.loc[:, "round"] = np.int32(sim_round)
    else:
        df["round"] = np.int32(sim_round)
    return df


def _build_scenario_features(raw_dir: Path, drivers: List[str], sim_year: int, sim_round: int, track: str) -> pd.DataFrame:
    ctx = {
        "raw_dir": raw_dir,
        "year": sim_year,
        "round": sim_round,
        "track": track,
        "drivers": drivers,
        "roster": drivers,
        "mode": "auto",
        "allow_fallback_actual": True,
    }
    df = featurize_pre(ctx)
    df = sanitize_frame_columns(df)
    if df.empty:
        raise RuntimeError("Failed to rebuild scenario features from raw data")

    df = _ordered_driver_slice(df, drivers)
    if df.empty:
        raise RuntimeError("Requested drivers are missing from rebuilt scenario features")

    if "year" not in df.columns:
        df["year"] = np.int32(sim_year)
    else:
        df.loc[:, "year"] = np.int32(sim_year)
    if "round" not in df.columns:
        df["round"] = np.int32(sim_round)
    else:
        df.loc[:, "round"] = np.int32(sim_round)
    return df


def _explain_drivers(runner: InferenceRunner, df: pd.DataFrame, drivers: List[str], topn: int = 10) -> None:
    feat_cols = list(runner.feature_columns)
    present = set(df["Driver"].astype(str))
    drivers = [drv for drv in drivers if drv in present]
    if not drivers:
        print("[warn] none of --explain drivers found in DF", file=sys.stderr)
        return

    sub = df.set_index("Driver").loc[drivers].copy()
    X = transform_with_scaler_df(sub, feat_cols, runner.artifacts.scaler, as_array=True).astype(np.float32, copy=False)

    model = runner.artifacts.model
    if model is None:
        print("[warn] No model in artifacts; cannot compute explanations.", file=sys.stderr)
        return
    model.eval()

    device = next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else torch.device("cpu")
    xt = _to_tensor(X, device=str(device))

    with torch.enable_grad():
        scores = model(xt)
        if isinstance(scores, (list, tuple)):
            scores = scores[0]
        scores = scores.reshape(-1)

    grads = []
    for i in range(xt.shape[0]):
        model.zero_grad(set_to_none=True)
        if xt.grad is not None:
            xt.grad.zero_()
        scores[i].backward(retain_graph=True)
        grads.append(xt.grad[i].detach().cpu().numpy().copy())
    grad = np.vstack(grads)
    contrib = grad * xt.detach().cpu().numpy()

    with torch.no_grad():
        base = model(xt)
        if isinstance(base, (list, tuple)):
            base = base[0]
        base = base.reshape(-1).detach().cpu().numpy()

    deltas = np.zeros_like(contrib, dtype=np.float32)
    for j in range(X.shape[1]):
        X_mut = xt.detach().clone()
        X_mut[:, j] = 0.0
        with torch.no_grad():
            s_mut = model(X_mut)
            if isinstance(s_mut, (list, tuple)):
                s_mut = s_mut[0]
            s_mut = s_mut.reshape(-1).detach().cpu().numpy()
        deltas[:, j] = base - s_mut

    for idx, drv in enumerate(drivers):
        print(f"\n=== EXPLAIN {drv} ===")
        z = xt[idx].detach().cpu().numpy()
        topn_eff = min(topn, len(feat_cols))

        topz = np.argsort(-np.abs(z))[:topn_eff]
        print("[A] top |z-score| features")
        for j in topz:
            print(f"  {feat_cols[j]:40s}  z={z[j]:+7.3f}")

        c = contrib[idx]
        topc = np.argsort(-np.abs(c))[:topn_eff]
        print("[B] top grad*input")
        for j in topc:
            print(f"  {feat_cols[j]:40s}  g*x={c[j]:+9.5f}")

        d = deltas[idx]
        topd = np.argsort(-np.abs(d))[:topn_eff]
        print("[C] top Δscore if set to mean")
        for j in topd:
            print(f"  {feat_cols[j]:40s}  Δscore={d[j]:+9.5f}")


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    drivers = [drv.strip() for drv in args.drivers.split(",") if drv.strip()]
    if not drivers:
        raise SystemExit("No drivers provided")

    if args.raw_dir:
        df = _build_scenario_features(Path(args.raw_dir), drivers, int(args.sim_year), int(args.sim_round), args.track)
    else:
        if not args.features:
            raise SystemExit("Either --raw-dir or --features is required")
        print("[warn] using legacy prediction mode from historical feature rows; pass --raw-dir for rebuilt scenario features", file=sys.stderr)
        df = _build_legacy_features(Path(args.features), drivers, int(args.sim_year), int(args.sim_round))

    _apply_track_onehot(df, args.track)
    _apply_weather(df, json.loads(args.weather_json) if args.weather_json else {})
    grid_map = _parse_grid(args.grid)
    _apply_grid(df, grid_map)

    _astype_floatwise(df)
    df = sanitize_frame_columns(df.copy())

    runner = InferenceRunner.from_dir(args.artifacts)
    rank_df = runner.rank(
        df,
        temperature=float(args.tau),
        by=("year", "round"),
        include_probs=True,
        ascending=False,
    )

    rank_df = _apply_grid_weight(
        rank_df=rank_df,
        df_features=df,
        user_grid_map=grid_map,
        grid_weight=float(args.grid_weight),
        tau=float(args.tau),
        groupby=("year", "round"),
        beta=0.25,
    )

    pretty_print_table(rank_df, topk=int(args.topk), cols_mode=args.cols)

    if args.explain:
        exp_list = [drv.strip() for drv in args.explain.split(",") if drv.strip()]
        _explain_drivers(runner, df, exp_list, topn=10)


if __name__ == "__main__":
    main(sys.argv[1:])
