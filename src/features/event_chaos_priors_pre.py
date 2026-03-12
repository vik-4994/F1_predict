#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import re

import numpy as np
import pandas as pd

from .utils import load_race_control, load_track_status, read_csv_if_exists, to_td, weather_path

__all__ = ["featurize"]


def _asof_mask(races: pd.DataFrame, year: int, rnd: int) -> pd.Series:
    y = pd.to_numeric(races.get("year"), errors="coerce")
    r = pd.to_numeric(races.get("round"), errors="coerce")
    return (y < int(year)) | ((y == int(year)) & (r < int(rnd)))


def _list_prev_races(races: pd.DataFrame, year: int, rnd: int, prev_n: int) -> List[Tuple[int, int]]:
    if races is None or races.empty:
        return []
    past = races.loc[_asof_mask(races, year, rnd), ["year", "round"]].dropna()
    if past.empty:
        return []
    past = past.astype(int).sort_values(["year", "round"]).tail(int(prev_n))
    return list(map(tuple, past.values.tolist()))


def _scan_prev_races(raw_dir: Path, year: int, rnd: int, prev_n: int) -> List[Tuple[int, int]]:
    pairs = set()
    for path in raw_dir.glob("results_*.csv"):
        m = re.search(r"results_(\d{4})_(\d{1,2})(?:_[A-Z]+)?\.csv$", path.name)
        if not m:
            continue
        y, r = int(m.group(1)), int(m.group(2))
        if (y < year) or (y == year and r < rnd):
            pairs.add((y, r))
    out = sorted(pairs)
    return out[-int(prev_n):]


def _slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _round_column(df: pd.DataFrame) -> Optional[str]:
    for cand in ("RoundNumber", "round", "Round"):
        if cand in df.columns:
            return cand
    return None


def _event_slug(raw_dir: Path, y: int, r: int) -> str:
    meta = read_csv_if_exists(raw_dir / f"meta_{y}_{r}.csv")
    if not meta.empty:
        for col in ("EventName", "OfficialEventName", "Location", "CircuitName", "Name"):
            if col in meta.columns and meta[col].notna().any():
                return _slugify(str(meta[col].dropna().iloc[0]))
    sch = read_csv_if_exists(raw_dir / f"schedule_{y}.csv")
    if not sch.empty:
        round_col = _round_column(sch)
        if round_col is not None:
            rr = sch.loc[pd.to_numeric(sch[round_col], errors="coerce") == int(r)]
            if not rr.empty:
                for col in ("EventName", "OfficialEventName", "Location"):
                    if col in rr.columns and rr[col].notna().any():
                        return _slugify(str(rr[col].dropna().iloc[0]))
    return f"{y}_{r}"


def _current_drivers(raw_dir: Path, year: int, rnd: int, drivers: Optional[Iterable[str]]) -> List[str]:
    if drivers:
        vals = [str(d).strip() for d in drivers if str(d).strip()]
        if vals:
            return list(dict.fromkeys(vals))
    for src in (
        f"entrylist_{year}_{rnd}_Q.csv",
        f"entrylist_{year}_{rnd}.csv",
        f"results_{year}_{rnd}_Q.csv",
        f"results_{year}_{rnd}.csv",
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


def _load_session_status(raw_dir: Path, y: int, r: int) -> pd.DataFrame:
    for name in (f"session_status_{y}_{r}.csv", f"session_status_{y}_{r}_R.csv"):
        df = read_csv_if_exists(raw_dir / name)
        if df.empty:
            continue
        out = df.copy()
        for col in ("Status", "Message"):
            if col in out.columns:
                out[col] = out[col].astype(str).str.upper()
        for col in ("Time", "SessionTime"):
            if col in out.columns:
                out[col] = pd.to_timedelta(out[col], errors="coerce")
        if "Utc" in out.columns:
            out["Utc"] = pd.to_datetime(out["Utc"], errors="coerce", utc=True)
        return out
    return pd.DataFrame()


def _contains_any(series: pd.Series, patterns: Iterable[str]) -> bool:
    if series is None or series.empty:
        return False
    txt = series.astype(str).str.upper()
    return bool(txt.apply(lambda x: any(p in x for p in patterns)).any())


def _weather_start_stats(raw_dir: Path, y: int, r: int, start_window_min: int) -> Dict[str, float]:
    df = read_csv_if_exists(weather_path(raw_dir, y, r))
    if df.empty:
        return {"wet_start": np.nan, "rainfall_share_start": np.nan}
    work = df.copy()
    if "Time" in work.columns:
        work["Time"] = pd.to_timedelta(work["Time"], errors="coerce")
        lim = pd.Timedelta(minutes=int(start_window_min))
        sub = work.loc[work["Time"].notna() & (work["Time"] <= lim)].copy()
        if sub.empty:
            sub = work.copy()
    else:
        sub = work.copy()

    rain = pd.Series(dtype=float)
    if "Rainfall" in sub.columns:
        rain = (
            sub["Rainfall"]
            .astype(str)
            .str.strip()
            .str.upper()
            .map({"TRUE": 1.0, "FALSE": 0.0, "1": 1.0, "0": 0.0})
        )
    elif "RainProb" in sub.columns:
        rain = pd.to_numeric(sub["RainProb"], errors="coerce").clip(lower=0.0, upper=1.0)
    if rain.notna().any():
        rainfall_share = float(rain.mean())
        return {
            "wet_start": float(rain.max() > 0),
            "rainfall_share_start": rainfall_share,
        }
    return {"wet_start": np.nan, "rainfall_share_start": np.nan}


def _race_chaos_stats(
    raw_dir: Path,
    y: int,
    r: int,
    *,
    start_window_min: int,
    early_window_min: int,
) -> Dict[str, float]:
    rc = load_race_control(raw_dir, y, r)
    ts = load_track_status(raw_dir, y, r)
    ss = _load_session_status(raw_dir, y, r)
    weather = _weather_start_stats(raw_dir, y, r, start_window_min)

    rc_msg = rc["Message"].astype(str).str.upper() if ("Message" in rc.columns and not rc.empty) else pd.Series(dtype=str)
    rc_flag = rc["Flag"].astype(str).str.upper() if ("Flag" in rc.columns and not rc.empty) else pd.Series(dtype=str)
    rc_cat = rc["Category"].astype(str).str.upper() if ("Category" in rc.columns and not rc.empty) else pd.Series(dtype=str)
    rc_status = rc["Status"].astype(str).str.upper() if ("Status" in rc.columns and not rc.empty) else pd.Series(dtype=str)

    ts_msg = ts["Message"].astype(str).str.upper() if ("Message" in ts.columns and not ts.empty) else pd.Series(dtype=str)
    ts_status = ts["Status"].astype(str).str.upper() if ("Status" in ts.columns and not ts.empty) else pd.Series(dtype=str)

    ss_msg = ss["Message"].astype(str).str.upper() if ("Message" in ss.columns and not ss.empty) else pd.Series(dtype=str)
    ss_status = ss["Status"].astype(str).str.upper() if ("Status" in ss.columns and not ss.empty) else pd.Series(dtype=str)

    had_sc = bool(
        _contains_any(rc_msg, ("SAFETY CAR DEPLOYED", "SAFETY CAR IN THIS LAP", "SAFETY CAR THROUGH THE PIT LANE"))
        or _contains_any(ts_msg, ("SCDEPLOYED",))
        or bool(((rc_cat == "SAFETYCAR") & (rc_status.isin(["DEPLOYED", "IN THIS LAP"]))).any())
    )
    had_vsc = bool(
        _contains_any(rc_msg, ("VIRTUAL SAFETY CAR DEPLOYED", "VIRTUAL SAFETY CAR ENDING"))
        or _contains_any(ts_msg, ("VSCDEPLOYED", "VSCENDING"))
        or _contains_any(ss_msg, ("VIRTUAL SAFETY CAR",))
    )
    had_red_flag = bool(
        _contains_any(rc_msg, ("RED FLAG",))
        or _contains_any(rc_flag, ("RED",))
        or _contains_any(ts_msg, ("RED",))
        or _contains_any(ss_msg, ("RED FLAG",))
        or _contains_any(ss_status, ("ABORTED", "SUSPENDED", "RED"))
    )
    delayed_start = bool(
        _contains_any(rc_msg, ("STARTING PROCEDURE SUSPENDED", "START DELAYED", "FORMATION LAP WILL START AT"))
        or _contains_any(ss_msg, ("START DELAYED", "DELAY", "SUSPEND"))
        or _contains_any(ss_status, ("DELAYED", "SUSPENDED"))
    )
    start_behind_sc = bool(
        _contains_any(
            rc_msg,
            ("RACE WILL START BEHIND THE SAFETY CAR", "FORMATION LAP WILL BE STARTED BEHIND THE SAFETY CAR"),
        )
        or _contains_any(ss_msg, ("START BEHIND THE SAFETY CAR",))
    )

    early_yellow = np.nan
    if not ts.empty and "Time" in ts.columns:
        lim = pd.Timedelta(minutes=int(early_window_min))
        early = ts.loc[ts["Time"].notna() & (ts["Time"] <= lim)].copy()
        if not early.empty:
            early_msg = early["Message"].astype(str).str.upper() if "Message" in early.columns else pd.Series(dtype=str)
            early_status = early["Status"].astype(str).str.upper() if "Status" in early.columns else pd.Series(dtype=str)
            early_yellow = float(
                _contains_any(early_msg, ("YELLOW", "SCDEPLOYED", "VSCDEPLOYED", "RED"))
                or bool(early_status.isin(["2", "4", "5", "6", "7"]).any())
            )

    sc_deploy_n = int(
        rc_msg.str.contains(r"\bSAFETY CAR DEPLOYED\b", regex=True, na=False).sum()
        + ts_msg.eq("SCDEPLOYED").sum()
    )
    vsc_deploy_n = int(
        rc_msg.str.contains(r"\bVIRTUAL SAFETY CAR DEPLOYED\b", regex=True, na=False).sum()
        + ts_msg.eq("VSCDEPLOYED").sum()
    )

    stats = {
        "sc_rate": float(had_sc),
        "vsc_rate": float(had_vsc),
        "red_flag_rate": float(had_red_flag),
        "wet_start_rate": float(weather["wet_start"]) if pd.notna(weather["wet_start"]) else np.nan,
        "delayed_start_rate": float(delayed_start),
        "start_behind_sc_rate": float(start_behind_sc),
        "early_yellow_rate": float(early_yellow) if pd.notna(early_yellow) else np.nan,
        "sc_deploy_mean": float(sc_deploy_n),
        "vsc_deploy_mean": float(vsc_deploy_n),
        "rainfall_share_start": float(weather["rainfall_share_start"]) if pd.notna(weather["rainfall_share_start"]) else np.nan,
    }
    return stats


def _aggregate(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {"n": 0}
    out: Dict[str, float] = {"n": float(len(rows))}
    keys = [k for k in rows[0].keys()]
    for key in keys:
        vals = pd.to_numeric(pd.Series([row.get(key, np.nan) for row in rows]), errors="coerce")
        out[key] = float(vals.mean()) if vals.notna().any() else np.nan
    return out


def _blend(same: Dict[str, float], recent: Dict[str, float], key: str) -> float:
    same_n = float(same.get("n", 0.0) or 0.0)
    recent_n = float(recent.get("n", 0.0) or 0.0)
    same_v = same.get(key, np.nan)
    recent_v = recent.get(key, np.nan)
    if same_n <= 0 and recent_n <= 0:
        return np.nan
    if same_n <= 0 or pd.isna(same_v):
        return float(recent_v)
    if recent_n <= 0 or pd.isna(recent_v):
        return float(same_v)
    alpha = float(np.clip(same_n / 3.0, 0.0, 1.0))
    return float(alpha * float(same_v) + (1.0 - alpha) * float(recent_v))


def featurize(ctx: dict) -> pd.DataFrame:
    raw_dir = Path(ctx.get("raw_dir", "data/raw_csv"))
    year = int(ctx["year"])
    rnd = int(ctx["round"])
    prev_n = int(ctx.get("chaos_prev_n", 10))
    recent_window = int(ctx.get("chaos_recent_window", 8))
    start_window_min = int(ctx.get("chaos_start_window_min", 12))
    early_window_min = int(ctx.get("chaos_early_window_min", 15))

    races = read_csv_if_exists(raw_dir / "races.csv")
    prev = _list_prev_races(races, year, rnd, prev_n=prev_n) if not races.empty else []
    if not prev:
        prev = _scan_prev_races(raw_dir, year, rnd, prev_n)

    cur_slug = _event_slug(raw_dir, year, rnd)
    same = [(y, r) for (y, r) in prev if _event_slug(raw_dir, y, r) == cur_slug]
    recent = prev[-recent_window:]

    recent_rows = [
        _race_chaos_stats(
            raw_dir,
            y,
            r,
            start_window_min=start_window_min,
            early_window_min=early_window_min,
        )
        for (y, r) in recent
    ]
    same_rows = [
        _race_chaos_stats(
            raw_dir,
            y,
            r,
            start_window_min=start_window_min,
            early_window_min=early_window_min,
        )
        for (y, r) in same
    ]

    recent_agg = _aggregate(recent_rows)
    same_agg = _aggregate(same_rows)

    drivers = _current_drivers(raw_dir, year, rnd, ctx.get("drivers"))
    if not drivers:
        return pd.DataFrame()

    out = pd.DataFrame({"Driver": drivers})
    out["chaos_pre_hist_n"] = float(recent_agg.get("n", 0.0) or 0.0)
    out["chaos_pre_same_track_n"] = float(same_agg.get("n", 0.0) or 0.0)

    for key, col in (
        ("sc_rate", "chaos_pre_sc_rate"),
        ("vsc_rate", "chaos_pre_vsc_rate"),
        ("red_flag_rate", "chaos_pre_red_flag_rate"),
        ("wet_start_rate", "chaos_pre_wet_start_rate"),
        ("delayed_start_rate", "chaos_pre_delayed_start_rate"),
        ("start_behind_sc_rate", "chaos_pre_start_behind_sc_rate"),
        ("early_yellow_rate", "chaos_pre_early_yellow_rate"),
        ("sc_deploy_mean", "chaos_pre_sc_deploy_mean"),
        ("vsc_deploy_mean", "chaos_pre_vsc_deploy_mean"),
        ("rainfall_share_start", "chaos_pre_rainfall_share_start"),
    ):
        out[col] = _blend(same_agg, recent_agg, key)

    weights = {
        "chaos_pre_sc_rate": 0.28,
        "chaos_pre_vsc_rate": 0.12,
        "chaos_pre_red_flag_rate": 0.22,
        "chaos_pre_wet_start_rate": 0.16,
        "chaos_pre_start_behind_sc_rate": 0.10,
        "chaos_pre_delayed_start_rate": 0.04,
        "chaos_pre_early_yellow_rate": 0.08,
    }
    chaos_index = pd.Series(0.0, index=out.index, dtype=float)
    any_signal = pd.Series(False, index=out.index)
    for col, w in weights.items():
        vals = pd.to_numeric(out[col], errors="coerce")
        any_signal |= vals.notna()
        chaos_index = chaos_index + vals.fillna(0.0) * float(w)
    out["chaos_pre_index"] = chaos_index.where(any_signal, np.nan)
    return out

