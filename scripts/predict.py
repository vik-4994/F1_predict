#!/usr/bin/env python3
# scripts/predict.py — удобный предикт: вероятность победы в %, сортировка по позициям
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.training import InferenceRunner, log

# =====================================================================================
#                                         utils
# =====================================================================================

def _slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _load_all_features(features_path: Path) -> pd.DataFrame:
    p = Path(features_path)
    if p.is_dir():
        ap = p / "all_features.parquet"
        if ap.exists():
            return _read_table(ap)
        ap_csv = ap.with_suffix(".csv")
        return _read_table(ap_csv)
    return _read_table(p)


def _parse_drivers(arg: str) -> List[str]:
    return [x.strip() for x in arg.split(",") if x.strip()]


def _load_lineup(lineup_path: Optional[str], drivers_arg: Optional[str]) -> List[str]:
    if lineup_path:
        p = Path(lineup_path)
        if not p.exists():
            raise FileNotFoundError(f"lineup file not found: {p}")
        if p.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
            except Exception as e:
                raise RuntimeError("PyYAML is required for .yaml lineup; install: pip install pyyaml") from e
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
        else:
            data = json.loads(p.read_text(encoding="utf-8"))
        seq = data["drivers"] if isinstance(data, dict) and "drivers" in data else data
        return [str(x).strip() for x in seq if str(x).strip()]
    if drivers_arg:
        return _parse_drivers(drivers_arg)
    raise ValueError("Provide --lineup FILE.(json|yaml) or --drivers 'VER,PER,...'")


def _ensure_feature_columns(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in feature_cols:
        if c not in out.columns:
            out[c] = np.nan
    return out


# =====================================================================================
#                          weather & track overrides (robust)
# =====================================================================================

_WEATHER_MAP: List[Tuple[str, Tuple[str, ...]]] = [
    ("airtemp", ("weather_pre_air_temp_mean", "temp_air", "weather_air_temp")),
    ("tracktemp", ("weather_pre_track_temp_mean", "temp_track", "weather_track_temp")),
    ("humidity", ("weather_pre_humidity_mean", "humidity")),
    ("windspeed", ("weather_pre_wind_kph_mean", "wind_speed", "wind_kph")),
    ("winddirection", ("weather_pre_wind_dir_mean", "wind_dir")),
]


def _override_weather(df: pd.DataFrame, weather_json: Optional[str]) -> pd.DataFrame:
    if not weather_json:
        return df
    try:
        vals_in = json.loads(weather_json)
    except Exception as e:
        log(f"⚠️  Bad --weather-json: {e}")
        return df

    norm: Dict[str, float] = {}
    for k, v in vals_in.items():
        k2 = _slugify(str(k)).replace("_", "")
        try:
            norm[k2] = float(v)
        except Exception:
            pass

    out = df.copy()
    for key, candidates in _WEATHER_MAP:
        if key not in norm:
            continue
        val = norm[key]
        for col in candidates:
            if col in out.columns:
                out.loc[:, col] = val
    return out


def _override_track(df: pd.DataFrame, track_name: Optional[str], feature_cols: Sequence[str]) -> Tuple[pd.DataFrame, Optional[str]]:
    if not track_name:
        return df, None
    slug = _slugify(track_name)
    out = df.copy()

    track_cols = [c for c in feature_cols if c.startswith("track_is_")]
    for c in track_cols:
        if c in out.columns:
            out.loc[:, c] = 0.0

    chosen = f"track_is_{slug}"
    if chosen in feature_cols:
        if chosen not in out.columns:
            out[chosen] = 0.0
        out.loc[:, chosen] = 1.0
        return out, chosen
    else:
        print(f"[warn] track column '{chosen}' отсутствует в feature_cols — модель обучалась без этого трека", file=sys.stderr)
        if chosen not in out.columns:
            out[chosen] = 1.0
        else:
            out.loc[:, chosen] = 1.0
        return out, None


# =====================================================================================
#                       custom features (track-aware improvements)
# =====================================================================================

def _make_custom_features(
    allF: pd.DataFrame,
    drivers: List[str],
    history_window: int,
    sim_year: int,
    sim_round: int,
) -> pd.DataFrame:
    if allF.empty:
        raise RuntimeError("all_features is empty; provide --features pointing to all_features(.parquet/.csv) or its folder")

    allF = allF.copy()
    allF["Driver"] = allF["Driver"].astype(str)

    parts = []
    for drv in drivers:
        hist = allF[allF["Driver"] == drv].sort_values(["year", "round"])
        if hist.empty:
            parts.append(pd.DataFrame({"Driver": [drv], "year": [sim_year], "round": [sim_round]}))
            continue
        tail = hist.tail(history_window)
        num_cols = [c for c in tail.columns if pd.api.types.is_numeric_dtype(tail[c])]
        prof = tail[num_cols].mean(numeric_only=True).to_frame().T
        prof["Driver"] = drv
        prof["year"] = sim_year
        prof["round"] = sim_round
        parts.append(prof)
    df = pd.concat(parts, ignore_index=True, sort=False)
    for col in ("year", "round"):
        if col not in df.columns:
            df[col] = sim_year if col == "year" else sim_round
    return df


def _inject_track_same_features(
    df_custom: pd.DataFrame,
    allF: pd.DataFrame,
    chosen_track_col: Optional[str],
    feature_cols: Sequence[str],
    driver_col: str = "Driver",
) -> pd.DataFrame:
    if not chosen_track_col or chosen_track_col not in allF.columns:
        return df_custom

    out = df_custom.copy()
    need = [c for c in feature_cols if c.startswith("track_same_")]
    if not need:
        return out

    allF2 = allF[allF[chosen_track_col] == 1].copy()
    if allF2.empty:
        return out

    allF2 = allF2.sort_values([driver_col, "year", "round"]).drop_duplicates([driver_col], keep="last")
    cols = [driver_col, *[c for c in need if c in allF2.columns]]
    have = allF2[cols].set_index(driver_col)

    for i, row in out.iterrows():
        drv = str(row.get(driver_col, ""))
        if drv in have.index:
            for c in have.columns:
                if c in out.columns:
                    out.at[i, c] = have.at[drv, c]
    return out


# =====================================================================================
#                           sorting & pretty printing helpers
# =====================================================================================

def _finalize_for_output(df: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    d = df.copy()
    if "rank" in d.columns:
        try:
            d["pos"] = d["rank"].astype(int)
        except Exception:
            d["pos"] = d["rank"]
    if "p_win" in d.columns:
        d["p_win_%"] = (d["p_win"] * 100.0).map(lambda x: f"{x:.1f}%")
    if sort_by == "rank":
        if {"year", "round", "rank"}.issubset(d.columns):
            d = d.sort_values(["year", "round", "rank", "Driver"])  # race-mode
        else:
            d = d.sort_values(["rank", "Driver"])  # custom-mode
    elif sort_by == "p_win":
        d = d.sort_values(["p_win", "score"], ascending=[False, False])
    elif sort_by == "score":
        d = d.sort_values(["score"], ascending=False)
    return d.reset_index(drop=True)


def _choose_columns(df: pd.DataFrame, mode: str) -> List[str]:
    always_front = [c for c in ["pos", "Driver", "Team", "year", "round", "score", "p_win_%"] if c in df.columns]
    if mode == "mini":
        tail = [c for c in ["GridPosition", "finish_position"] if c in df.columns]
        return always_front + tail
    if mode == "core":
        extra = [
            "GridPosition", "finish_position", "quali_pre_pos_p50", "quali_pre_top10_rate",
            "hist_pre_best10_pace_p50_s", "hist_pre_clean_share_mean",
            "driver_trend", "team_dev_trend",
        ]
        return always_front + [c for c in extra if c in df.columns]
    return list(df.columns)


def _print_table(df: pd.DataFrame, topk: Optional[int], cols_mode: str) -> None:
    d = df.copy()
    if topk is not None and topk > 0:
        d = d.head(topk)
    cols = _choose_columns(d, cols_mode)
    d = d.reindex(columns=cols)
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 160,
                           "display.float_format", lambda v: f"{v:0.6g}"):
        print(d.to_string(index=False))


def _export(df: pd.DataFrame, fmt: str, out_path: Optional[str]) -> None:
    if not out_path:
        return
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        df.to_json(p, orient="records", force_ascii=False, indent=2)
    elif fmt == "csv":
        df.to_csv(p, index=False)
    elif fmt == "parquet":
        df.to_parquet(p, index=False)
    else:
        raise ValueError(f"Unsupported --format for export: {fmt}")
    log(f"💾 Saved predictions to {p}")


# =====================================================================================
#                                          eval
# =====================================================================================

def _evaluate_against_targets(pred: pd.DataFrame, targets_path: Path, year: int, round_: int) -> None:
    try:
        T = _read_table(targets_path)
        if T.empty:
            log("⚠️  Targets file is empty or not found — skip eval")
            return
        T = T[(T["year"] == year) & (T["round"] == round_)].copy()
        if T.empty:
            log("⚠️  No targets for specified race — skip eval")
            return
        cand = ["finish_pos", "finish_position", "finish_order", "position", "place"]
        finish_col = next((c for c in cand if c in T.columns), None)
        if finish_col is None:
            log("⚠️  Targets missing finish position column — skip eval")
            return
        for c in ["Driver", finish_col]:
            if c not in T.columns:
                log("⚠️  Targets missing required columns — skip eval")
                return
        M = pred.merge(T[["Driver", finish_col]], on="Driver", how="inner")
        if M.empty:
            log("⚠️  No overlapping drivers with targets — skip eval")
            return
        M = M.copy()
        M["true_rank"] = M[finish_col].rank(method="min", ascending=True).astype(int)
        mae = float(np.mean(np.abs(M["true_rank"].to_numpy() - M["rank"].to_numpy())))
        tr = pd.Series(M["true_rank"].to_numpy(dtype=float)).rank(method="average").to_numpy()
        pr = pd.Series(M["rank"].to_numpy(dtype=float)).rank(method="average").to_numpy()
        sp = float(np.corrcoef(tr, pr)[0, 1]) if len(M) >= 2 else np.nan
        top1 = 1.0 if (M.sort_values("rank").iloc[0]["Driver"] == M.sort_values("true_rank").iloc[0]["Driver"]) else 0.0
        log(f"Eval: spearman={sp:.4f}, mae_rank={mae:.3f}, top1={top1:.3f}  (n={len(M)})")
    except Exception as e:
        log(f"⚠️  Eval failed: {e}")


# =====================================================================================
#                                          CLI
# =====================================================================================

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("Predict finishing order")
    ap.add_argument("--artifacts", default="models/ranker_v1", help="Папка с ranker.pt / scaler.json / feature_cols.txt")
    ap.add_argument("--features", default="data/processed", help="Путь к features: либо all_features.parquet, либо папка с per-race")
    ap.add_argument("--mode", choices=["race", "custom"], default="race", help="Режим предикта")

    # race-mode
    ap.add_argument("--year", type=int, help="Год гонки (race mode)")
    ap.add_argument("--round", dest="round_", type=int, help="Номер этапа (round) (race mode)")
    ap.add_argument("--eval", action="store_true", help="В режиме race: посчитать метрики против targets")
    ap.add_argument("--targets", default="data/processed/all_targets.parquet", help="Путь к all_targets для оценки (race mode)")

    # custom-mode
    ap.add_argument("--lineup", help="JSON/YAML файл со списком пилотов (ключ 'drivers' или массив)")
    ap.add_argument("--drivers", help="Альтернатива файлу: строка 'VER,PER,LEC,...'")
    ap.add_argument("--history-window", type=int, default=3, help="Сколько последних гонок усреднять на профиль пилота (custom mode)")
    ap.add_argument("--sim-year", type=int, default=2099, help="Год для синтетической гонки (custom mode)")
    ap.add_argument("--sim-round", type=int, default=1, help="Раунд для синтетической гонки (custom mode)")

    # common overrides
    ap.add_argument("--track", help="Название трека/ивента (например, 'Monza', 'Jeddah', 'Spa'). Для custom/race (если пригодно)")
    ap.add_argument("--weather-json", help='JSON с погодой: {"AirTemp":28,"TrackTemp":42,"Humidity":0.45,"WindSpeed":3.2,"WindDirection":180}')
    ap.add_argument("--temp", type=float, default=1.0, help="Softmax temperature для перерасчёта вероятностей")

    # output / pretty
    ap.add_argument("--topk", type=int, default=20, help="Сколько верхних строк показывать")
    ap.add_argument("--cols", choices=["mini", "core", "all"], default="mini", help="Набор колонок для вывода")
    ap.add_argument("--format", choices=["table", "json", "csv", "parquet"], default="table", help="Формат вывода/сохранения")
    ap.add_argument("--out", help="Путь для сохранения предикта, если формат json/csv/parquet")
    ap.add_argument("--sort-by", choices=["rank", "p_win", "score"], default="rank", help="Сортировка вывода: по позиции, вероятности, или скору")

    return ap


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    runner = InferenceRunner.from_dir(args.artifacts)
    feature_cols = runner.artifacts.feature_cols

    allF = _load_all_features(Path(args.features))
    if allF.empty:
        log("❌ Не найден файл с фичами (all_features.*)")
        sys.exit(2)

    chosen_track_col: Optional[str] = None

    if args.mode == "race":
        if args.year is None or args.round_ is None:
            ap.error("--mode race требует --year и --round")
        df_race = allF[(allF.get("year") == args.year) & (allF.get("round") == args.round_)].copy()
        if df_race.empty:
            log("❌ Нет строк для указанной гонки в all_features")
            sys.exit(2)
        df_race = _ensure_feature_columns(df_race, feature_cols)
        if args.track:
            df_race, chosen_track_col = _override_track(df_race, args.track, feature_cols)
        df_race = _override_weather(df_race, args.weather_json)
        out = runner.rank(df_race, temperature=args.temp, by=("year", "round"), include_probs=True)
        if args.eval:
            _evaluate_against_targets(out, Path(args.targets), int(df_race["year"].iloc[0]), int(df_race["round"].iloc[0]))
    else:
        drivers = _load_lineup(args.lineup, args.drivers)
        if not drivers:
            ap.error("Пустой список пилотов")
        df_custom = _make_custom_features(
            allF,
            drivers=drivers,
            history_window=args.history_window,
            sim_year=args.sim_year,
            sim_round=args.sim_round,
        )
        df_custom = _ensure_feature_columns(df_custom, feature_cols)
        df_custom, chosen_track_col = _override_track(df_custom, args.track, feature_cols)
        df_custom = _override_weather(df_custom, args.weather_json)
        if chosen_track_col:
            df_custom = _inject_track_same_features(df_custom, allF, chosen_track_col, feature_cols)
        out = runner.rank(df_custom, temperature=args.temp, by=None, include_probs=True)

    out = _finalize_for_output(out, args.sort_by)

    if args.format == "table":
        _print_table(out, args.topk, args.cols)
    else:
        df_to_save = out.head(args.topk) if (args.topk and args.topk > 0) else out
        _export(df_to_save, args.format, args.out)
        if args.out is None:
            if args.format == "json":
                print(df_to_save.to_json(orient="records", force_ascii=False, indent=2))
            else:
                log("⚠️  Для csv/parquet укажите --out PATH")


if __name__ == "__main__":
    main()
