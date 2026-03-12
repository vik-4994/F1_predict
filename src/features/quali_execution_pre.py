#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .utils import read_csv_if_exists
from .weekend_helpers import current_roster, current_team_map, pairwise_team_delta, to_seconds

__all__ = ["featurize"]


def _load_results_q(raw_dir: Path, year: int, rnd: int) -> pd.DataFrame:
    for name in (f"results_{year}_{rnd}_Q.csv", f"qualifying_{year}_{rnd}.csv", f"quali_{year}_{rnd}.csv"):
        df = read_csv_if_exists(raw_dir / name)
        if df.empty:
            continue
        out = df.copy()
        if "Abbreviation" in out.columns and "Driver" not in out.columns:
            out = out.rename(columns={"Abbreviation": "Driver"})
        if "Driver" not in out.columns:
            continue
        out["Driver"] = out["Driver"].astype(str)
        return out
    return pd.DataFrame()


def _load_laps_q(raw_dir: Path, year: int, rnd: int) -> pd.DataFrame:
    df = read_csv_if_exists(raw_dir / f"laps_{year}_{rnd}_Q.csv")
    if df.empty:
        return df
    out = df.copy()
    if "Abbreviation" in out.columns and "Driver" not in out.columns:
        out = out.rename(columns={"Abbreviation": "Driver"})
    if "Driver" not in out.columns:
        return pd.DataFrame()
    out["Driver"] = out["Driver"].astype(str)
    out["lap_sec"] = to_seconds(out["LapTime"]) if "LapTime" in out.columns else np.nan
    for src, dst in (("Sector1Time", "s1_s"), ("Sector2Time", "s2_s"), ("Sector3Time", "s3_s")):
        out[dst] = to_seconds(out[src]) if src in out.columns else np.nan
    return out


def _best_lap_from_results(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    cols = [c for c in ("Q1", "Q2", "Q3", "Time") if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index, dtype=float)
    arr = pd.concat([to_seconds(df[c]) for c in cols], axis=1)
    return arr.min(axis=1, skipna=True)


def _stage_reached(df: pd.DataFrame) -> pd.Series:
    stage = pd.Series(0.0, index=df.index, dtype=float)
    if "Q1" in df.columns:
        stage = np.where(to_seconds(df["Q1"]).notna(), 1.0, stage)
    if "Q2" in df.columns:
        stage = np.where(to_seconds(df["Q2"]).notna(), 2.0, stage)
    if "Q3" in df.columns:
        stage = np.where(to_seconds(df["Q3"]).notna(), 3.0, stage)
    return pd.Series(stage, index=df.index, dtype=float)


def _run_id(laps: pd.DataFrame) -> pd.Series:
    if "Stint" in laps.columns:
        stint = pd.to_numeric(laps["Stint"], errors="coerce")
        if stint.notna().any():
            return stint
    if "PitOutTime" in laps.columns:
        return laps["PitOutTime"].notna().cumsum().replace(0, 1)
    return pd.Series(1, index=laps.index, dtype=float)


def _empty_base(drivers: List[str]) -> pd.DataFrame:
    base = pd.DataFrame({"Driver": drivers})
    for col in (
        "qexec_pre_best_lap_s",
        "qexec_pre_gap_to_pole_s",
        "qexec_pre_gap_to_tm_s",
        "qexec_pre_ideal_lap_gap_s",
        "qexec_pre_final_run_improve_s",
        "qexec_pre_deleted_lap_share",
        "qexec_pre_stage_reached",
        "qexec_pre_timed_laps_n",
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

    qres = _load_results_q(raw_dir, year, rnd)
    qlaps = _load_laps_q(raw_dir, year, rnd)
    if qres.empty and qlaps.empty:
        return base

    metrics = pd.DataFrame({"Driver": drivers})
    if not qres.empty:
        tmp = qres[["Driver"]].copy()
        tmp["best_lap_s"] = _best_lap_from_results(qres)
        tmp["stage_reached"] = _stage_reached(qres)
        metrics = metrics.merge(tmp.drop_duplicates("Driver"), on="Driver", how="left")
    else:
        metrics["best_lap_s"] = np.nan
        metrics["stage_reached"] = np.nan

    if not qlaps.empty:
        timed = qlaps[qlaps["lap_sec"].notna()].copy()
        deleted = qlaps.copy()
        if "Deleted" in deleted.columns:
            deleted["Deleted"] = deleted["Deleted"].fillna(False).astype(bool)
        else:
            deleted["Deleted"] = False
        timed["run_id"] = _run_id(timed)

        good = timed.copy()
        if "TrackStatus" in good.columns:
            good = good[good["TrackStatus"].astype(str).str.strip().eq("1")].copy()
        if "IsAccurate" in good.columns:
            good = good[good["IsAccurate"].fillna(False).astype(bool)].copy()
        if "Deleted" in good.columns:
            good = good[~good["Deleted"].fillna(False).astype(bool)].copy()

        rows = []
        source = good if not good.empty else timed
        for drv, sub in source.groupby("Driver", sort=False):
            sub = sub.sort_values(["run_id", "lap_sec"], kind="mergesort")
            best_lap = float(pd.to_numeric(sub["lap_sec"], errors="coerce").min()) if sub["lap_sec"].notna().any() else np.nan
            ideal = np.nan
            if {"s1_s", "s2_s", "s3_s"}.issubset(sub.columns):
                sec_best = sub[["s1_s", "s2_s", "s3_s"]].apply(pd.to_numeric, errors="coerce").min()
                if sec_best.notna().all():
                    ideal = float(sec_best.sum())
            final_improve = np.nan
            if "run_id" in sub.columns and sub["run_id"].notna().any():
                last_run = pd.to_numeric(sub["run_id"], errors="coerce").max()
                final_run = sub.loc[pd.to_numeric(sub["run_id"], errors="coerce") == last_run]
                earlier = sub.loc[pd.to_numeric(sub["run_id"], errors="coerce") < last_run]
                if not final_run.empty and not earlier.empty and final_run["lap_sec"].notna().any() and earlier["lap_sec"].notna().any():
                    final_improve = float(earlier["lap_sec"].min() - final_run["lap_sec"].min())
            raw_drv = deleted.loc[deleted["Driver"] == drv].copy()
            timed_mask = raw_drv["lap_sec"].notna()
            deleted_share = float(raw_drv.loc[timed_mask, "Deleted"].mean()) if bool(timed_mask.any()) else np.nan
            rows.append(
                {
                    "Driver": str(drv),
                    "best_lap_s_laps": best_lap,
                    "ideal_lap_gap_s": float(best_lap - ideal) if pd.notna(best_lap) and pd.notna(ideal) else np.nan,
                    "final_run_improve_s": final_improve,
                    "deleted_lap_share": deleted_share,
                    "timed_laps_n": float(pd.to_numeric(sub["lap_sec"], errors="coerce").notna().sum()),
                }
            )
        lap_metrics = pd.DataFrame(rows)
        metrics = metrics.merge(lap_metrics, on="Driver", how="left")
        metrics["best_lap_s"] = metrics["best_lap_s"].fillna(metrics["best_lap_s_laps"])
    else:
        metrics["ideal_lap_gap_s"] = np.nan
        metrics["final_run_improve_s"] = np.nan
        metrics["deleted_lap_share"] = np.nan
        metrics["timed_laps_n"] = np.nan

    if metrics["best_lap_s"].notna().any():
        pole = float(metrics["best_lap_s"].min())
        metrics["gap_to_pole_s"] = pd.to_numeric(metrics["best_lap_s"], errors="coerce") - pole
    else:
        metrics["gap_to_pole_s"] = np.nan

    teams = current_team_map(raw_dir, year, rnd)
    if not teams.empty:
        tmp = metrics[["Driver", "best_lap_s"]].merge(teams, on="Driver", how="left")
        tmp = pairwise_team_delta(tmp, ["best_lap_s"])
        metrics = metrics.merge(tmp[["Driver", "best_lap_s_tm_delta"]], on="Driver", how="left")
        metrics["gap_to_tm_s"] = metrics["best_lap_s_tm_delta"]
    else:
        metrics["gap_to_tm_s"] = np.nan

    out = base.merge(metrics, on="Driver", how="left")
    out["qexec_pre_best_lap_s"] = out["best_lap_s"]
    out["qexec_pre_gap_to_pole_s"] = out["gap_to_pole_s"]
    out["qexec_pre_gap_to_tm_s"] = out["gap_to_tm_s"]
    out["qexec_pre_ideal_lap_gap_s"] = out["ideal_lap_gap_s"]
    out["qexec_pre_final_run_improve_s"] = out["final_run_improve_s"]
    out["qexec_pre_deleted_lap_share"] = out["deleted_lap_share"]
    out["qexec_pre_stage_reached"] = out["stage_reached"]
    out["qexec_pre_timed_laps_n"] = out["timed_laps_n"]
    return out[base.columns]
