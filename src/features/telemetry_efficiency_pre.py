#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .utils import read_csv_if_exists
from .weekend_helpers import current_roster, current_team_map, pairwise_team_delta

__all__ = ["featurize"]


SESSION_PRIORITY = {"FP2": 2, "FP3": 3, "Q": 4}
SESSION_ORDER = ("Q", "FP3", "FP2")


def _telemetry_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {}
    work = df.copy()
    work["Speed"] = pd.to_numeric(work.get("Speed", np.nan), errors="coerce")
    work["Throttle"] = pd.to_numeric(work.get("Throttle", np.nan), errors="coerce")
    work["DRS"] = pd.to_numeric(work.get("DRS", np.nan), errors="coerce")
    if "Brake" in work.columns:
        work["Brake"] = work["Brake"].astype(str).str.strip().str.upper().map({"TRUE": 1.0, "FALSE": 0.0, "1": 1.0, "0": 0.0})
    else:
        work["Brake"] = np.nan

    speed = work["Speed"].dropna()
    throttle = work["Throttle"]
    drs_open = pd.to_numeric(work["DRS"], errors="coerce") >= 10
    full_throttle = throttle >= 95
    brake_mask = work["Brake"] >= 0.5
    straight_mask = full_throttle & work["Speed"].ge(150)

    drs_on_speed = pd.to_numeric(work.loc[straight_mask & drs_open, "Speed"], errors="coerce")
    drs_off_speed = pd.to_numeric(work.loc[straight_mask & (~drs_open), "Speed"], errors="coerce")

    return {
        "samples_n": float(len(work)),
        "speed_p95_kph": float(speed.quantile(0.95)) if not speed.empty else np.nan,
        "speed_max_kph": float(speed.max()) if not speed.empty else np.nan,
        "full_throttle_share": float(full_throttle.mean()) if full_throttle.notna().any() else np.nan,
        "brake_share": float(brake_mask.mean()) if brake_mask.notna().any() else np.nan,
        "brake_speed_mean_kph": float(pd.to_numeric(work.loc[brake_mask, "Speed"], errors="coerce").mean()) if bool(brake_mask.any()) else np.nan,
        "drs_open_share": float(drs_open.mean()) if drs_open.notna().any() else np.nan,
        "drs_speed_gain_kph": float(drs_on_speed.mean() - drs_off_speed.mean()) if not drs_on_speed.empty and not drs_off_speed.empty else np.nan,
    }


def _load_session_metrics(raw_dir: Path, year: int, rnd: int, session: str, drivers: Sequence[str]) -> pd.DataFrame:
    rows = []
    for drv in drivers:
        path = raw_dir / f"telemetry_{year}_{rnd}_{session}_{drv}.csv"
        df = read_csv_if_exists(path)
        if df.empty:
            continue
        metrics = _telemetry_metrics(df)
        if not metrics:
            continue
        metrics["Driver"] = str(drv)
        metrics["session_ord"] = float(SESSION_PRIORITY.get(str(session).upper(), 0))
        rows.append(metrics)
    return pd.DataFrame(rows)


def _select_latest_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    for drv, sub in df.groupby("Driver", sort=False):
        pick = sub.sort_values(["session_ord", "samples_n", "speed_p95_kph"], ascending=[False, False, False]).iloc[0]
        rows.append(pick.to_dict())
    return pd.DataFrame(rows)


def _empty_base(drivers: Sequence[str]) -> pd.DataFrame:
    base = pd.DataFrame({"Driver": list(drivers)})
    for col in (
        "tele_eff_pre_session_ord",
        "tele_eff_pre_samples_n",
        "tele_eff_pre_speed_p95_kph",
        "tele_eff_pre_speed_max_kph",
        "tele_eff_pre_full_throttle_share",
        "tele_eff_pre_brake_share",
        "tele_eff_pre_brake_speed_mean_kph",
        "tele_eff_pre_drs_open_share",
        "tele_eff_pre_drs_speed_gain_kph",
        "tele_eff_pre_speed_tm_delta_kph",
        "tele_eff_pre_drs_gain_tm_delta_kph",
    ):
        base[col] = np.nan
    return base


def featurize(ctx: Dict) -> pd.DataFrame:
    raw_dir = Path(ctx.get("raw_dir", "data/raw_csv"))
    year = int(ctx["year"])
    rnd = int(ctx["round"])
    sessions: Sequence[str] = ctx.get("telemetry_sessions", SESSION_ORDER)
    drivers = current_roster(raw_dir, year, rnd, ctx.get("drivers"))
    if not drivers:
        return pd.DataFrame()

    base = _empty_base(drivers)
    frames = []
    for session in sessions:
        df = _load_session_metrics(raw_dir, year, rnd, str(session).upper(), drivers)
        if not df.empty:
            frames.append(df)
    if not frames:
        return base

    metrics = _select_latest_metrics(pd.concat(frames, ignore_index=True, sort=False))
    teams = current_team_map(raw_dir, year, rnd)
    if not teams.empty:
        tmp = metrics[["Driver", "speed_p95_kph", "drs_speed_gain_kph"]].merge(teams, on="Driver", how="left")
        tmp = pairwise_team_delta(tmp, ["speed_p95_kph", "drs_speed_gain_kph"])
        metrics = metrics.merge(
            tmp[["Driver", "speed_p95_kph_tm_delta", "drs_speed_gain_kph_tm_delta"]],
            on="Driver",
            how="left",
        )

    out = base.merge(metrics, on="Driver", how="left")
    out["tele_eff_pre_session_ord"] = out["session_ord"]
    out["tele_eff_pre_samples_n"] = out["samples_n"]
    out["tele_eff_pre_speed_p95_kph"] = out["speed_p95_kph"]
    out["tele_eff_pre_speed_max_kph"] = out["speed_max_kph"]
    out["tele_eff_pre_full_throttle_share"] = out["full_throttle_share"]
    out["tele_eff_pre_brake_share"] = out["brake_share"]
    out["tele_eff_pre_brake_speed_mean_kph"] = out["brake_speed_mean_kph"]
    out["tele_eff_pre_drs_open_share"] = out["drs_open_share"]
    out["tele_eff_pre_drs_speed_gain_kph"] = out["drs_speed_gain_kph"]
    out["tele_eff_pre_speed_tm_delta_kph"] = out.get("speed_p95_kph_tm_delta", np.nan)
    out["tele_eff_pre_drs_gain_tm_delta_kph"] = out.get("drs_speed_gain_kph_tm_delta", np.nan)
    return out[base.columns]
