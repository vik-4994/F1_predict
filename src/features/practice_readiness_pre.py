#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .utils import read_csv_if_exists
from .weekend_helpers import current_roster, current_team_map, pairwise_team_delta, to_seconds

__all__ = ["featurize"]


SESSION_ORDER = {"FP1": 1, "FP2": 2, "FP3": 3}
DRY_COMPOUNDS = {"SOFT", "MEDIUM", "HARD", "S", "M", "H"}


def _load_session_laps(raw_dir: Path, year: int, rnd: int, session: str) -> pd.DataFrame:
    df = read_csv_if_exists(raw_dir / f"laps_{year}_{rnd}_{session}.csv")
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

    lap_src = out["LapTime"] if "LapTime" in out.columns else out.get("milliseconds", pd.Series(np.nan, index=out.index))
    out["lap_sec"] = to_seconds(lap_src)
    out = out[out["lap_sec"].notna()].copy()
    if out.empty:
        return out

    out["Driver"] = out["Driver"].astype(str)
    out["session"] = str(session).upper()
    out["session_ord"] = float(SESSION_ORDER.get(out["session"].iloc[0], 0))
    out["is_accurate"] = out["IsAccurate"].fillna(False).astype(bool) if "IsAccurate" in out.columns else True
    if "TrackStatus" in out.columns:
        out["is_green"] = out["TrackStatus"].astype(str).str.strip().eq("1")
    else:
        out["is_green"] = True
    if "Compound" in out.columns:
        out["is_dry"] = out["Compound"].astype(str).str.strip().str.upper().isin(DRY_COMPOUNDS)
    else:
        out["is_dry"] = False
    return out


def _session_summary(laps: pd.DataFrame) -> pd.DataFrame:
    if laps.empty:
        return pd.DataFrame()
    grp = laps.groupby("Driver", dropna=False, sort=False)
    out = grp.agg(
        session_laps=("lap_sec", "size"),
        accurate_share=("is_accurate", "mean"),
        green_share=("is_green", "mean"),
        dry_share=("is_dry", "mean"),
    ).reset_index()
    out["session"] = str(laps["session"].iloc[0]).upper()
    out["session_ord"] = float(laps["session_ord"].iloc[0])
    return out


def featurize(ctx: Dict) -> pd.DataFrame:
    raw_dir = Path(ctx.get("raw_dir", "data/raw_csv"))
    year = int(ctx["year"])
    rnd = int(ctx["round"])
    sessions: Sequence[str] = ctx.get("practice_sessions_all", ("FP1", "FP2", "FP3"))

    drivers = current_roster(raw_dir, year, rnd, ctx.get("drivers"))
    if not drivers:
        return pd.DataFrame()
    base = pd.DataFrame({"Driver": drivers})
    for col in (
        "ready_pre_sessions_seen_n",
        "ready_pre_total_laps",
        "ready_pre_accurate_share",
        "ready_pre_green_share",
        "ready_pre_dry_share",
        "ready_pre_latest_session_ord",
        "ready_pre_latest_session_laps",
        "ready_pre_missing_fp2_flag",
        "ready_pre_missing_fp3_flag",
        "ready_pre_total_laps_tm_delta",
        "ready_pre_accurate_share_tm_delta",
        "ready_pre_issue_index",
    ):
        base[col] = np.nan

    summaries: List[pd.DataFrame] = []
    available_sessions = set()
    for session in sessions:
        laps = _load_session_laps(raw_dir, year, rnd, str(session).upper())
        if laps.empty:
            continue
        available_sessions.add(str(session).upper())
        sess_sum = _session_summary(laps)
        if not sess_sum.empty:
            summaries.append(sess_sum)

    if not summaries:
        return base

    merged = pd.concat(summaries, ignore_index=True, sort=False)
    rows = []
    for drv in drivers:
        sub = merged.loc[merged["Driver"] == drv].copy()
        if sub.empty:
            rows.append({"Driver": drv})
            continue
        total_laps = float(pd.to_numeric(sub["session_laps"], errors="coerce").sum())
        latest_ord = float(pd.to_numeric(sub["session_ord"], errors="coerce").max())
        latest = sub.loc[pd.to_numeric(sub["session_ord"], errors="coerce") == latest_ord].iloc[-1]
        weights = pd.to_numeric(sub["session_laps"], errors="coerce").fillna(0.0)

        def _weighted(col: str) -> float:
            vals = pd.to_numeric(sub[col], errors="coerce")
            mask = vals.notna() & weights.gt(0)
            if not bool(mask.any()):
                return np.nan
            return float(np.average(vals[mask], weights=weights[mask]))

        row = {
            "Driver": drv,
            "ready_pre_sessions_seen_n": float(len(sub)),
            "ready_pre_total_laps": total_laps,
            "ready_pre_accurate_share": _weighted("accurate_share"),
            "ready_pre_green_share": _weighted("green_share"),
            "ready_pre_dry_share": _weighted("dry_share"),
            "ready_pre_latest_session_ord": latest_ord,
            "ready_pre_latest_session_laps": float(pd.to_numeric(latest["session_laps"], errors="coerce")),
            "ready_pre_missing_fp2_flag": (
                float(not bool((sub["session"] == "FP2").any())) if "FP2" in available_sessions else np.nan
            ),
            "ready_pre_missing_fp3_flag": (
                float(not bool((sub["session"] == "FP3").any())) if "FP3" in available_sessions else np.nan
            ),
        }
        rows.append(row)

    out = base.merge(pd.DataFrame(rows), on="Driver", how="left", suffixes=("", "_new"))
    for col in [c for c in out.columns if c.endswith("_new")]:
        root = col[:-4]
        out[root] = out[col]
        out = out.drop(columns=[col])

    teams = current_team_map(raw_dir, year, rnd)
    if not teams.empty:
        delta = out[["Driver", "ready_pre_total_laps", "ready_pre_accurate_share"]].merge(teams, on="Driver", how="left")
        delta = pairwise_team_delta(delta, ["ready_pre_total_laps", "ready_pre_accurate_share"])
        keep = [c for c in ("Driver", "ready_pre_total_laps_tm_delta", "ready_pre_accurate_share_tm_delta") if c in delta.columns]
        if keep:
            out = out.drop(columns=[c for c in keep if c != "Driver"], errors="ignore").merge(delta[keep], on="Driver", how="left")

    acc = pd.to_numeric(out["ready_pre_accurate_share"], errors="coerce")
    lap_delta = pd.to_numeric(out["ready_pre_total_laps_tm_delta"], errors="coerce")
    issue = pd.Series(0.0, index=out.index, dtype=float)
    for col, weight in (("ready_pre_missing_fp3_flag", 0.60), ("ready_pre_missing_fp2_flag", 0.35)):
        vals = pd.to_numeric(out[col], errors="coerce")
        issue = issue + vals.fillna(0.0) * weight
    issue = issue + np.clip((-lap_delta.fillna(0.0)) / 25.0, 0.0, 1.0)
    issue = issue + np.clip(0.90 - acc.fillna(0.90), 0.0, 0.90)
    out["ready_pre_issue_index"] = np.clip(issue, 0.0, 3.0)

    keep = [
        "Driver",
        "ready_pre_sessions_seen_n",
        "ready_pre_total_laps",
        "ready_pre_accurate_share",
        "ready_pre_green_share",
        "ready_pre_dry_share",
        "ready_pre_latest_session_ord",
        "ready_pre_latest_session_laps",
        "ready_pre_missing_fp2_flag",
        "ready_pre_missing_fp3_flag",
        "ready_pre_total_laps_tm_delta",
        "ready_pre_accurate_share_tm_delta",
        "ready_pre_issue_index",
    ]
    for col in keep:
        if col not in out.columns:
            out[col] = np.nan
    return out[keep]
