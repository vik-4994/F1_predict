#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .utils import read_csv_if_exists
from .weekend_helpers import current_roster, current_team_map, pairwise_team_delta, to_seconds

__all__ = ["featurize"]


_RESULT_FILES = {
    "S": ("results_{year}_{rnd}_S.csv",),
    "SQ": ("results_{year}_{rnd}_SQ.csv", "results_{year}_{rnd}_SS.csv"),
}
_TIME_COLS = ("Q1", "Q2", "Q3", "Time")


def _event_format(raw_dir: Path, year: int, rnd: int) -> str:
    for name in (f"meta_{year}_{rnd}.csv", f"meta_{year}_{rnd}_Q.csv", f"session_info_{year}_{rnd}.csv"):
        df = read_csv_if_exists(raw_dir / name)
        if df.empty:
            continue
        for col in ("EventFormat", "Type", "Name"):
            if col in df.columns and df[col].notna().any():
                value = str(df[col].dropna().iloc[0]).strip().lower()
                if value:
                    return value
    schedule_path = raw_dir / f"schedule_{year}.csv"
    if schedule_path.exists():
        df = pd.read_csv(schedule_path)
        round_col = "round" if "round" in df.columns else ("RoundNumber" if "RoundNumber" in df.columns else None)
        if round_col is not None:
            row = df.loc[pd.to_numeric(df[round_col], errors="coerce") == int(rnd)]
            if not row.empty and "EventFormat" in row.columns and row["EventFormat"].notna().any():
                return str(row["EventFormat"].dropna().iloc[0]).strip().lower()
    return ""


def _is_sprint_weekend(raw_dir: Path, year: int, rnd: int) -> bool:
    fmt = _event_format(raw_dir, year, rnd)
    if "sprint" in fmt:
        return True
    return any((raw_dir / pattern.format(year=year, rnd=rnd)).exists() for pats in _RESULT_FILES.values() for pattern in pats)


def _load_results(raw_dir: Path, year: int, rnd: int, session: str) -> pd.DataFrame:
    for pattern in _RESULT_FILES.get(session, ()):
        df = read_csv_if_exists(raw_dir / pattern.format(year=year, rnd=rnd))
        if not df.empty:
            out = df.copy()
            if "Abbreviation" in out.columns and "Driver" not in out.columns:
                out = out.rename(columns={"Abbreviation": "Driver"})
            if "Driver" not in out.columns:
                continue
            out["Driver"] = out["Driver"].astype(str)
            return out
    return pd.DataFrame()


def _best_lap_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Driver", "best_s"])
    out = pd.DataFrame({"Driver": df["Driver"].astype(str)})
    cols = [c for c in _TIME_COLS if c in df.columns]
    if cols:
        arr = pd.concat([to_seconds(df[c]) for c in cols], axis=1)
        out["best_s"] = arr.min(axis=1, skipna=True)
    else:
        out["best_s"] = np.nan
    return out.drop_duplicates("Driver")


def _position_points_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Driver", "finish_pos", "grid_pos", "points"])
    out = pd.DataFrame({"Driver": df["Driver"].astype(str)})
    out["finish_pos"] = pd.to_numeric(df["Position"], errors="coerce") if "Position" in df.columns else np.nan
    out["grid_pos"] = pd.to_numeric(df["GridPosition"], errors="coerce") if "GridPosition" in df.columns else np.nan
    out["points"] = pd.to_numeric(df["Points"], errors="coerce") if "Points" in df.columns else np.nan
    return out.drop_duplicates("Driver")


def _empty_base(drivers: Sequence[str]) -> pd.DataFrame:
    base = pd.DataFrame({"Driver": list(drivers)})
    for col in (
        "sprint_pre_format_flag",
        "sprint_pre_has_data",
        "sprint_pre_sq_pos",
        "sprint_pre_sq_best_s",
        "sprint_pre_sq_gap_to_best_s",
        "sprint_pre_sq_tm_delta_s",
        "sprint_pre_finish_pos",
        "sprint_pre_gain_vs_grid",
        "sprint_pre_points",
        "sprint_pre_gain_abs",
        "sprint_pre_tm_finish_delta",
    ):
        base[col] = np.nan
    return base


def featurize(ctx: Dict) -> pd.DataFrame:
    raw_dir = Path(ctx.get("raw_dir", "data/raw_csv"))
    year = int(ctx["year"])
    rnd = int(ctx["round"])
    drivers = current_roster(raw_dir, year, rnd, ctx.get("drivers"))
    if not drivers:
        return pd.DataFrame()

    base = _empty_base(drivers)
    sprint_flag = float(_is_sprint_weekend(raw_dir, year, rnd))
    base["sprint_pre_format_flag"] = sprint_flag

    sq = _load_results(raw_dir, year, rnd, "SQ")
    sres = _load_results(raw_dir, year, rnd, "S")
    if sq.empty and sres.empty:
        base["sprint_pre_has_data"] = 0.0
        return base

    out = base.copy()
    out["sprint_pre_has_data"] = 1.0
    teams = current_team_map(raw_dir, year, rnd)

    if not sq.empty:
        sq_best = _best_lap_table(sq).rename(columns={"best_s": "sq_best_s"})
        sq_pos = pd.DataFrame(
            {
                "Driver": sq["Driver"].astype(str),
                "sq_pos": pd.to_numeric(sq["Position"], errors="coerce") if "Position" in sq.columns else np.nan,
            }
        ).drop_duplicates("Driver")
        out = out.merge(sq_best, on="Driver", how="left").merge(sq_pos, on="Driver", how="left")
        if out["sq_best_s"].notna().any():
            pole = float(out["sq_best_s"].min())
            out["sprint_pre_sq_gap_to_best_s"] = pd.to_numeric(out["sq_best_s"], errors="coerce") - pole
        out["sprint_pre_sq_best_s"] = out["sq_best_s"]
        out["sprint_pre_sq_pos"] = out["sq_pos"]
        if not teams.empty:
            tmp = out[["Driver", "sq_best_s"]].merge(teams, on="Driver", how="left")
            tmp = pairwise_team_delta(tmp, ["sq_best_s"])
            out = out.merge(tmp[["Driver", "sq_best_s_tm_delta"]], on="Driver", how="left")
            out["sprint_pre_sq_tm_delta_s"] = out["sq_best_s_tm_delta"]

    if not sres.empty:
        s_tbl = _position_points_table(sres)
        out = out.merge(s_tbl, on="Driver", how="left")
        out["sprint_pre_finish_pos"] = out["finish_pos"]
        out["sprint_pre_gain_vs_grid"] = pd.to_numeric(out["grid_pos"], errors="coerce") - pd.to_numeric(out["finish_pos"], errors="coerce")
        out["sprint_pre_gain_abs"] = pd.to_numeric(out["sprint_pre_gain_vs_grid"], errors="coerce").abs()
        out["sprint_pre_points"] = out["points"]
        if not teams.empty:
            tmp = out[["Driver", "sprint_pre_finish_pos"]].merge(teams, on="Driver", how="left")
            tmp = pairwise_team_delta(tmp, ["sprint_pre_finish_pos"])
            out = out.merge(tmp[["Driver", "sprint_pre_finish_pos_tm_delta"]], on="Driver", how="left")
            out["sprint_pre_tm_finish_delta"] = out["sprint_pre_finish_pos_tm_delta"]

    drop_cols = [c for c in ("sq_best_s", "sq_pos", "sq_best_s_tm_delta", "finish_pos", "grid_pos", "points", "sprint_pre_finish_pos_tm_delta") if c in out.columns]
    return out.drop(columns=drop_cols, errors="ignore")
