#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .practice_longrun_pre import featurize as feat_practice
from .utils import read_csv_if_exists

__all__ = ["featurize"]


def _current_drivers(raw_dir: Path, year: int, rnd: int, drivers: Optional[Iterable[str]]) -> List[str]:
    if drivers:
        vals = [str(d).strip() for d in drivers if str(d).strip()]
        if vals:
            return list(dict.fromkeys(vals))
    for src in (
        f"results_{year}_{rnd}_Q.csv",
        f"entrylist_{year}_{rnd}_Q.csv",
        f"results_{year}_{rnd}.csv",
        f"entrylist_{year}_{rnd}.csv",
    ):
        df = read_csv_if_exists(raw_dir / src)
        if df.empty:
            continue
        for col in ("Abbreviation", "Driver", "code", "driverRef", "BroadcastName"):
            if col in df.columns and df[col].notna().any():
                vals = df[col].astype(str).dropna().drop_duplicates().tolist()
                if vals:
                    return vals
    return []


def _to_seconds(series: pd.Series) -> pd.Series:
    try:
        td = pd.to_timedelta(series, errors="coerce")
        if td.notna().any():
            sec = td.dt.total_seconds()
            if sec.notna().mean() > 0.5:
                return sec
    except Exception:
        pass
    num = pd.to_numeric(series, errors="coerce")
    med = num.replace([np.inf, -np.inf], np.nan).median()
    if pd.notna(med) and med > 1e3:
        return num / 1000.0
    return num


def _team_map(raw_dir: Path, year: int, rnd: int) -> pd.DataFrame:
    for src in (
        f"results_{year}_{rnd}_Q.csv",
        f"entrylist_{year}_{rnd}_Q.csv",
        f"results_{year}_{rnd}.csv",
        f"entrylist_{year}_{rnd}.csv",
    ):
        df = read_csv_if_exists(raw_dir / src)
        if df.empty:
            continue
        dcol = next((c for c in ("Abbreviation", "Driver", "code", "driverRef") if c in df.columns), None)
        tcol = next((c for c in ("TeamName", "Team", "Constructor", "ConstructorName", "TeamName") if c in df.columns), None)
        if dcol and tcol:
            out = pd.DataFrame({"Driver": df[dcol].astype(str), "Team": df[tcol].astype(str)})
            out = out.dropna(subset=["Driver"]).drop_duplicates("Driver")
            if not out.empty:
                return out
    return pd.DataFrame(columns=["Driver", "Team"])


def _best_quali_table(raw_dir: Path, year: int, rnd: int) -> pd.DataFrame:
    q = read_csv_if_exists(raw_dir / f"results_{year}_{rnd}_Q.csv")
    if q.empty:
        q = read_csv_if_exists(raw_dir / f"qualifying_{year}_{rnd}.csv")
    if q.empty:
        return pd.DataFrame(columns=["Driver", "q_best_s", "q_pos"])

    dcol = next((c for c in ("Abbreviation", "Driver", "code", "driverRef") if c in q.columns), None)
    if dcol is None:
        return pd.DataFrame(columns=["Driver", "q_best_s", "q_pos"])
    out = pd.DataFrame({"Driver": q[dcol].astype(str)})
    q_cols = [c for c in ("Q1", "Q2", "Q3", "Time") if c in q.columns]
    if q_cols:
        arr = pd.concat([_to_seconds(q[c]) for c in q_cols], axis=1)
        out["q_best_s"] = arr.min(axis=1, skipna=True)
    else:
        out["q_best_s"] = np.nan
    if "Position" in q.columns:
        out["q_pos"] = pd.to_numeric(q["Position"], errors="coerce")
    else:
        out["q_pos"] = np.nan
    return out.drop_duplicates("Driver")


def _quali_sector_table(raw_dir: Path, year: int, rnd: int) -> pd.DataFrame:
    laps = read_csv_if_exists(raw_dir / f"laps_{year}_{rnd}_Q.csv")
    if laps.empty:
        return pd.DataFrame(columns=["Driver", "q_s1_best_s", "q_s2_best_s", "q_s3_best_s"])
    if "Driver" not in laps.columns:
        for c in ("Abbreviation", "code", "driverRef"):
            if c in laps.columns:
                laps = laps.rename(columns={c: "Driver"})
                break
    if "Driver" not in laps.columns:
        return pd.DataFrame(columns=["Driver", "q_s1_best_s", "q_s2_best_s", "q_s3_best_s"])

    work = laps.copy()
    if "TrackStatus" in work.columns:
        work = work[work["TrackStatus"].astype(str).str.strip().eq("1")].copy()
    if "IsAccurate" in work.columns:
        work = work[work["IsAccurate"].fillna(False).astype(bool)].copy()
    if work.empty:
        return pd.DataFrame(columns=["Driver", "q_s1_best_s", "q_s2_best_s", "q_s3_best_s"])

    for src, dst in (("Sector1Time", "q_s1_best_s"), ("Sector2Time", "q_s2_best_s"), ("Sector3Time", "q_s3_best_s")):
        if src in work.columns:
            work[dst] = _to_seconds(work[src])
        else:
            work[dst] = np.nan

    agg = (
        work.groupby("Driver", dropna=False)[["q_s1_best_s", "q_s2_best_s", "q_s3_best_s"]]
        .min()
        .reset_index()
    )
    return agg


def _pairwise_team_delta(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if df.empty or "Team" not in df.columns:
        return df
    out = df.copy()
    team_size = out.groupby("Team")["Driver"].transform("nunique")
    out["wknd_tm_has_pair"] = (team_size == 2).astype(float)

    mate = out[["Driver", "Team", *cols]].copy()
    mate = mate.rename(columns={"Driver": "Teammate", **{c: f"{c}__mate" for c in cols}})
    paired = out.merge(mate, on="Team", how="left")
    paired = paired[paired["Driver"] != paired["Teammate"]].copy()
    if paired.empty:
        for col in cols:
            out[f"{col}_tm_delta"] = np.nan
        return out

    keep = ["Driver", "Team", "wknd_tm_has_pair"]
    for col in cols:
        paired[f"{col}_tm_delta"] = pd.to_numeric(paired[col], errors="coerce") - pd.to_numeric(paired[f"{col}__mate"], errors="coerce")
        keep.append(f"{col}_tm_delta")

    reduced = paired[keep].drop_duplicates("Driver")
    out = out.drop(columns=["wknd_tm_has_pair"], errors="ignore").merge(reduced, on=["Driver", "Team"], how="left")
    return out


def featurize(ctx: Dict) -> pd.DataFrame:
    raw_dir = Path(ctx.get("raw_dir", "data/raw_csv"))
    year = int(ctx["year"])
    rnd = int(ctx["round"])

    drivers = _current_drivers(raw_dir, year, rnd, ctx.get("drivers"))
    if not drivers:
        return pd.DataFrame()
    base = pd.DataFrame({"Driver": drivers})

    teams = _team_map(raw_dir, year, rnd)
    qbest = _best_quali_table(raw_dir, year, rnd)
    qsec = _quali_sector_table(raw_dir, year, rnd)
    practice = feat_practice(ctx)

    out = base.merge(teams, on="Driver", how="left")
    out = out.merge(qbest, on="Driver", how="left")
    out = out.merge(qsec, on="Driver", how="left")
    if practice is not None and not practice.empty:
        pkeep = [
            c
            for c in (
                "Driver",
                "prac_pre_longrun_pace_s",
                "prac_pre_longrun_deg_s_per_lap",
                "prac_pre_shortrun_best3_s",
            )
            if c in practice.columns
        ]
        out = out.merge(practice[pkeep], on="Driver", how="left")

    delta_cols = [
        c
        for c in (
            "q_best_s",
            "q_pos",
            "q_s1_best_s",
            "q_s2_best_s",
            "q_s3_best_s",
            "prac_pre_longrun_pace_s",
            "prac_pre_longrun_deg_s_per_lap",
            "prac_pre_shortrun_best3_s",
        )
        if c in out.columns
    ]
    out = _pairwise_team_delta(out, delta_cols)

    rename_map = {
        "wknd_tm_has_pair": "wknd_pre_tm_has_pair",
        "q_best_s_tm_delta": "wknd_pre_q_tm_delta_s",
        "q_pos_tm_delta": "wknd_pre_q_pos_tm_delta",
        "q_s1_best_s_tm_delta": "wknd_pre_q_s1_tm_delta_s",
        "q_s2_best_s_tm_delta": "wknd_pre_q_s2_tm_delta_s",
        "q_s3_best_s_tm_delta": "wknd_pre_q_s3_tm_delta_s",
        "prac_pre_longrun_pace_s_tm_delta": "wknd_pre_prac_longrun_tm_delta_s",
        "prac_pre_longrun_deg_s_per_lap_tm_delta": "wknd_pre_prac_deg_tm_delta_s_per_lap",
        "prac_pre_shortrun_best3_s_tm_delta": "wknd_pre_prac_shortrun_tm_delta_s",
    }
    out = out.rename(columns=rename_map)

    sector_delta_cols = [c for c in ("wknd_pre_q_s1_tm_delta_s", "wknd_pre_q_s2_tm_delta_s", "wknd_pre_q_s3_tm_delta_s") if c in out.columns]
    if sector_delta_cols:
        sec = out[sector_delta_cols].apply(pd.to_numeric, errors="coerce")
        out["wknd_pre_q_sector_tm_delta_std_s"] = sec.std(axis=1, ddof=0)
        out["wknd_pre_q_sector_tm_delta_max_s"] = sec.max(axis=1)
        out["wknd_pre_q_sector_tm_delta_min_s"] = sec.min(axis=1)
    else:
        out["wknd_pre_q_sector_tm_delta_std_s"] = np.nan
        out["wknd_pre_q_sector_tm_delta_max_s"] = np.nan
        out["wknd_pre_q_sector_tm_delta_min_s"] = np.nan

    if {"q_s1_best_s", "q_s2_best_s", "q_s3_best_s"}.issubset(out.columns):
        sec_abs = out[["q_s1_best_s", "q_s2_best_s", "q_s3_best_s"]].apply(pd.to_numeric, errors="coerce")
        out["wknd_pre_q_sector_balance_std_s"] = sec_abs.std(axis=1, ddof=0)
        out["wknd_pre_q_sector_balance_span_s"] = sec_abs.max(axis=1) - sec_abs.min(axis=1)
    else:
        out["wknd_pre_q_sector_balance_std_s"] = np.nan
        out["wknd_pre_q_sector_balance_span_s"] = np.nan

    keep = [
        "Driver",
        "wknd_pre_tm_has_pair",
        "wknd_pre_q_tm_delta_s",
        "wknd_pre_q_pos_tm_delta",
        "wknd_pre_q_s1_tm_delta_s",
        "wknd_pre_q_s2_tm_delta_s",
        "wknd_pre_q_s3_tm_delta_s",
        "wknd_pre_q_sector_tm_delta_std_s",
        "wknd_pre_q_sector_tm_delta_max_s",
        "wknd_pre_q_sector_tm_delta_min_s",
        "wknd_pre_q_sector_balance_std_s",
        "wknd_pre_q_sector_balance_span_s",
        "wknd_pre_prac_longrun_tm_delta_s",
        "wknd_pre_prac_deg_tm_delta_s_per_lap",
        "wknd_pre_prac_shortrun_tm_delta_s",
    ]
    for col in keep:
        if col not in out.columns:
            out[col] = np.nan
    return out[keep]

