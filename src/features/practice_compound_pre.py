#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .practice_longrun_pre import SESSION_PRIORITY, _current_drivers, _fit_slope, _load_session_laps

__all__ = ["featurize"]

_COMPOUNDS = ("S", "M", "H")


def _empty_base(drivers: Sequence[str]) -> pd.DataFrame:
    base = pd.DataFrame({"Driver": list(drivers)})
    base["prac_cmp_pre_dry_compounds_n"] = np.nan
    for comp in _COMPOUNDS:
        low = comp.lower()
        base[f"prac_cmp_pre_{low}_best_s"] = np.nan
        base[f"prac_cmp_pre_{low}_longrun_pace_s"] = np.nan
        base[f"prac_cmp_pre_{low}_deg_s_per_lap"] = np.nan
    base["prac_cmp_pre_crossover_sm_s"] = np.nan
    base["prac_cmp_pre_crossover_mh_s"] = np.nan
    return base


def _best_lap_rows(laps: pd.DataFrame) -> pd.DataFrame:
    if laps.empty:
        return pd.DataFrame()
    rows = (
        laps.groupby(["Driver", "compound_norm", "session"], dropna=False, sort=False)["lap_sec"]
        .min()
        .reset_index()
        .rename(columns={"lap_sec": "best_s"})
    )
    rows["session_ord"] = rows["session"].map(SESSION_PRIORITY).fillna(0).astype(int)
    return rows


def _longrun_rows(laps: pd.DataFrame) -> pd.DataFrame:
    if laps.empty:
        return pd.DataFrame()
    rows = []
    for (drv, stint_id, comp), stint in laps.groupby(["Driver", "Stint", "compound_norm"], dropna=False, sort=False):
        stint = stint.sort_values("tyre_age")
        if len(stint) < 5:
            continue
        pace = pd.to_numeric(stint["lap_sec"], errors="coerce")
        slope = _fit_slope(pd.to_numeric(stint["tyre_age"], errors="coerce"), pace)
        rows.append(
            {
                "Driver": str(drv),
                "compound_norm": str(comp),
                "session": str(stint["session"].iloc[0]),
                "session_ord": int(SESSION_PRIORITY.get(str(stint["session"].iloc[0]), 0)),
                "longrun_pace_s": float(pace.median()) if pace.notna().any() else np.nan,
                "deg_s_per_lap": float(slope) if np.isfinite(slope) else np.nan,
                "longrun_laps": int(len(stint)),
            }
        )
    return pd.DataFrame(rows)


def _select_driver_rows(best_rows: pd.DataFrame, long_rows: pd.DataFrame) -> pd.DataFrame:
    drivers = sorted(set(best_rows.get("Driver", pd.Series(dtype=str)).tolist()) | set(long_rows.get("Driver", pd.Series(dtype=str)).tolist()))
    rows: List[Dict[str, float | str]] = []
    for drv in drivers:
        row: Dict[str, float | str] = {"Driver": str(drv)}
        compounds_seen = set()
        drv_best = best_rows.loc[best_rows["Driver"] == drv].copy() if not best_rows.empty else pd.DataFrame()
        drv_long = long_rows.loc[long_rows["Driver"] == drv].copy() if not long_rows.empty else pd.DataFrame()
        for comp in _COMPOUNDS:
            low = comp.lower()
            comp_best = drv_best.loc[drv_best["compound_norm"] == comp].copy()
            comp_long = drv_long.loc[drv_long["compound_norm"] == comp].copy()
            if not comp_best.empty:
                pick = comp_best.sort_values(["best_s", "session_ord"], ascending=[True, False]).iloc[0]
                row[f"prac_cmp_pre_{low}_best_s"] = float(pick["best_s"])
                compounds_seen.add(comp)
            else:
                row[f"prac_cmp_pre_{low}_best_s"] = np.nan
            if not comp_long.empty:
                pick = comp_long.sort_values(["longrun_laps", "session_ord", "longrun_pace_s"], ascending=[False, False, True]).iloc[0]
                row[f"prac_cmp_pre_{low}_longrun_pace_s"] = float(pick["longrun_pace_s"]) if pd.notna(pick["longrun_pace_s"]) else np.nan
                row[f"prac_cmp_pre_{low}_deg_s_per_lap"] = float(pick["deg_s_per_lap"]) if pd.notna(pick["deg_s_per_lap"]) else np.nan
                compounds_seen.add(comp)
            else:
                row[f"prac_cmp_pre_{low}_longrun_pace_s"] = np.nan
                row[f"prac_cmp_pre_{low}_deg_s_per_lap"] = np.nan
        row["prac_cmp_pre_dry_compounds_n"] = float(len(compounds_seen))
        s_best = row.get("prac_cmp_pre_s_best_s", np.nan)
        m_best = row.get("prac_cmp_pre_m_best_s", np.nan)
        h_best = row.get("prac_cmp_pre_h_best_s", np.nan)
        row["prac_cmp_pre_crossover_sm_s"] = float(m_best - s_best) if pd.notna(s_best) and pd.notna(m_best) else np.nan
        row["prac_cmp_pre_crossover_mh_s"] = float(h_best - m_best) if pd.notna(m_best) and pd.notna(h_best) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def featurize(ctx: dict) -> pd.DataFrame:
    raw_dir = Path(ctx.get("raw_dir", "data/raw_csv"))
    year = int(ctx["year"])
    rnd = int(ctx["round"])
    sessions: Sequence[str] = ctx.get("practice_sessions", ("FP3", "FP2", "FP1"))

    drivers = _current_drivers(raw_dir, year, rnd, ctx.get("drivers"))
    if not drivers:
        return pd.DataFrame()
    base = _empty_base(drivers)

    best_frames = []
    long_frames = []
    for session in sessions:
        laps = _load_session_laps(raw_dir, year, rnd, str(session).upper())
        if laps.empty:
            continue
        best_rows = _best_lap_rows(laps)
        long_rows = _longrun_rows(laps)
        if not best_rows.empty:
            best_frames.append(best_rows)
        if not long_rows.empty:
            long_frames.append(long_rows)

    if not best_frames and not long_frames:
        return base

    best_rows = pd.concat(best_frames, ignore_index=True, sort=False) if best_frames else pd.DataFrame()
    long_rows = pd.concat(long_frames, ignore_index=True, sort=False) if long_frames else pd.DataFrame()
    out = _select_driver_rows(best_rows, long_rows)
    return base.drop(columns=[c for c in base.columns if c != "Driver"]).merge(out, on="Driver", how="left")
