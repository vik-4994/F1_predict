#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .utils import read_csv_if_exists
from .weekend_helpers import current_roster, to_seconds

__all__ = ["featurize"]


def _load_qlaps(raw_dir: Path, year: int, rnd: int) -> pd.DataFrame:
    df = read_csv_if_exists(raw_dir / f"laps_{year}_{rnd}_Q.csv")
    if df.empty:
        return df
    out = df.copy()
    if "Driver" not in out.columns:
        for col in ("Abbreviation", "code", "driverRef"):
            if col in out.columns:
                out = out.rename(columns={col: "Driver"})
                break
    if "Driver" not in out.columns:
        return pd.DataFrame()
    out["Driver"] = out["Driver"].astype(str)
    lap_src = out["LapTime"] if "LapTime" in out.columns else out.get("milliseconds", pd.Series(np.nan, index=out.index))
    out["lap_sec"] = to_seconds(lap_src)
    progress_src = None
    for col in ("Time", "Sector3SessionTime", "LapStartTime"):
        if col in out.columns:
            progress_src = col
            break
    out["progress_sec"] = to_seconds(out[progress_src]) if progress_src else np.nan
    out = out[out["lap_sec"].notna() & out["progress_sec"].notna()].copy()
    if out.empty:
        return out
    out["deleted_flag"] = out["Deleted"].fillna(False).astype(bool) if "Deleted" in out.columns else False
    out["accurate_flag"] = out["IsAccurate"].fillna(False).astype(bool) if "IsAccurate" in out.columns else True
    out["green_flag"] = out["TrackStatus"].astype(str).str.strip().eq("1") if "TrackStatus" in out.columns else True
    return out


def _rank_pct(series: pd.Series, *, ascending: bool) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype=float)
    vals = pd.to_numeric(series, errors="coerce")
    mask = vals.notna()
    n = int(mask.sum())
    if n == 0:
        return out
    ranks = vals[mask].rank(method="average", ascending=ascending)
    if n == 1:
        out.loc[mask] = 1.0
    else:
        out.loc[mask] = 1.0 - (ranks - 1.0) / float(n - 1)
    return out


def featurize(ctx: Dict) -> pd.DataFrame:
    raw_dir = Path(ctx.get("raw_dir", "data/raw_csv"))
    year = int(ctx["year"])
    rnd = int(ctx["round"])
    drivers = current_roster(raw_dir, year, rnd, ctx.get("drivers"))
    if not drivers:
        return pd.DataFrame()
    base = pd.DataFrame({"Driver": drivers})
    for col in (
        "qevo_pre_best_lap_progress_pct",
        "qevo_pre_final_push_progress_pct",
        "qevo_pre_late_push_share",
        "qevo_pre_window_gap_s",
        "qevo_pre_timing_luck_s",
        "qevo_pre_push_laps_n",
        "qevo_pre_window_rank_pct",
    ):
        base[col] = np.nan

    qlaps = _load_qlaps(raw_dir, year, rnd)
    if qlaps.empty:
        return base

    timed = qlaps.copy()
    source = timed[timed["accurate_flag"] & timed["green_flag"] & ~timed["deleted_flag"]].copy()
    if source.empty:
        source = timed.copy()

    max_progress = float(pd.to_numeric(source["progress_sec"], errors="coerce").max())
    if not np.isfinite(max_progress) or max_progress <= 0:
        return base
    source["progress_pct"] = pd.to_numeric(source["progress_sec"], errors="coerce") / max_progress
    source["progress_pct"] = source["progress_pct"].clip(0.0, 1.0)

    session_best = float(pd.to_numeric(source["lap_sec"], errors="coerce").min()) if source["lap_sec"].notna().any() else np.nan

    rows: List[Dict[str, float | str]] = []
    for drv, sub in source.groupby("Driver", sort=False):
        sub = sub.sort_values("progress_pct", kind="mergesort")
        best_idx = pd.to_numeric(sub["lap_sec"], errors="coerce").idxmin()
        best_row = sub.loc[best_idx]
        best_prog = float(pd.to_numeric(best_row["progress_pct"], errors="coerce"))
        final_prog = float(pd.to_numeric(sub["progress_pct"], errors="coerce").max())
        window = source.loc[(source["progress_pct"] - best_prog).abs() <= 0.075]
        if window.empty:
            window = source
        window_best = float(pd.to_numeric(window["lap_sec"], errors="coerce").min()) if window["lap_sec"].notna().any() else np.nan
        rows.append(
            {
                "Driver": str(drv),
                "qevo_pre_best_lap_progress_pct": best_prog,
                "qevo_pre_final_push_progress_pct": final_prog,
                "qevo_pre_late_push_share": float((pd.to_numeric(sub["progress_pct"], errors="coerce") >= 0.60).mean()),
                "qevo_pre_window_gap_s": float(pd.to_numeric(best_row["lap_sec"], errors="coerce") - window_best) if np.isfinite(window_best) else np.nan,
                "qevo_pre_timing_luck_s": float(window_best - session_best) if np.isfinite(window_best) and np.isfinite(session_best) else np.nan,
                "qevo_pre_push_laps_n": float(len(sub)),
            }
        )

    out = base.merge(pd.DataFrame(rows), on="Driver", how="left", suffixes=("", "_new"))
    for col in [c for c in out.columns if c.endswith("_new")]:
        root = col[:-4]
        out[root] = out[col]
        out = out.drop(columns=[col])
    out["qevo_pre_window_rank_pct"] = _rank_pct(out["qevo_pre_window_gap_s"], ascending=True)
    keep = [
        "Driver",
        "qevo_pre_best_lap_progress_pct",
        "qevo_pre_final_push_progress_pct",
        "qevo_pre_late_push_share",
        "qevo_pre_window_gap_s",
        "qevo_pre_timing_luck_s",
        "qevo_pre_push_laps_n",
        "qevo_pre_window_rank_pct",
    ]
    return out[keep]
