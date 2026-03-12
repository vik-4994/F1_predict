#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .utils import read_csv_if_exists

__all__ = ["featurize"]


SESSION_PRIORITY = {"FP1": 1, "FP2": 2, "FP3": 3}
DRY_MAP = {
    "SOFT": "S",
    "MEDIUM": "M",
    "HARD": "H",
    "S": "S",
    "M": "M",
    "H": "H",
}


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


def _stint_age(df: pd.DataFrame) -> pd.Series:
    if "TyreLife" in df.columns:
        age = pd.to_numeric(df["TyreLife"], errors="coerce")
        if age.notna().any():
            return age
    if "Stint" in df.columns:
        stint = pd.to_numeric(df["Stint"], errors="coerce").astype("Int64")
        tmp = df.copy()
        tmp["_stint"] = stint
        return tmp.groupby(["Driver", "_stint"], sort=False).cumcount() + 1
    return pd.Series(np.nan, index=df.index, dtype=float)


def _dry_compound(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper().map(DRY_MAP)


def _is_green_track_status(series: pd.Series) -> pd.Series:
    txt = series.astype(str).str.strip()
    return ~txt.str.contains(r"[24567]", regex=True, na=False)


def _fit_slope(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 4:
        return np.nan
    xv = pd.to_numeric(x[mask], errors="coerce").to_numpy(dtype=float, copy=False)
    yv = pd.to_numeric(y[mask], errors="coerce").to_numpy(dtype=float, copy=False)
    if len(np.unique(xv)) < 3:
        return np.nan
    med = float(np.nanmedian(yv))
    mad = float(np.nanmedian(np.abs(yv - med)))
    if np.isfinite(mad) and mad > 0:
        keep = np.abs(yv - med) <= 4.0 * 1.4826 * mad
        xv = xv[keep]
        yv = yv[keep]
    if len(xv) < 4 or float(np.var(xv)) <= 1e-9:
        return np.nan
    return float(np.polyfit(xv, yv, deg=1)[0])


def _load_session_laps(raw_dir: Path, year: int, rnd: int, session: str) -> pd.DataFrame:
    df = read_csv_if_exists(raw_dir / f"laps_{year}_{rnd}_{session}.csv")
    if df.empty:
        return df
    work = df.copy()
    if "Driver" not in work.columns:
        for col in ("Abbreviation", "code", "driverRef"):
            if col in work.columns:
                work = work.rename(columns={col: "Driver"})
                break
    if "Driver" not in work.columns:
        return pd.DataFrame()

    if "LapNumber" in work.columns and "lap" not in work.columns:
        work = work.rename(columns={"LapNumber": "lap"})
    if "LapTime" in work.columns:
        lap_src = work["LapTime"]
    elif "milliseconds" in work.columns:
        lap_src = work["milliseconds"]
    else:
        lap_src = pd.Series(np.nan, index=work.index, dtype=float)
    work["lap_sec"] = _to_seconds(lap_src)
    work["compound_norm"] = _dry_compound(work.get("Compound", pd.Series(np.nan, index=work.index)))
    work["tyre_age"] = _stint_age(work)

    for flag_col in ("IsPitIn", "IsPitOut", "PitIn", "PitOut", "InPit", "OutPit"):
        if flag_col not in work.columns:
            work[flag_col] = False
    work["is_pit"] = False
    for flag_col in ("IsPitIn", "IsPitOut", "PitIn", "PitOut", "InPit", "OutPit"):
        work["is_pit"] |= work[flag_col].fillna(False).astype(bool)
    for td_col in ("PitInTime", "PitOutTime"):
        if td_col in work.columns:
            work["is_pit"] |= pd.to_timedelta(work[td_col], errors="coerce").notna()
    if "Stint" not in work.columns:
        compound_key = work["compound_norm"].fillna("UNK")
        stint_break = compound_key.ne(compound_key.groupby(work["Driver"]).shift()) | work["is_pit"]
        work["Stint"] = stint_break.groupby(work["Driver"]).cumsum()

    if "TrackStatus" in work.columns:
        work = work[_is_green_track_status(work["TrackStatus"])].copy()
    work = work[~work["is_pit"]].copy()
    work = work[work["compound_norm"].notna()].copy()
    work = work[work["lap_sec"].notna()].copy()
    work = work[work["tyre_age"].notna() & (pd.to_numeric(work["tyre_age"], errors="coerce") >= 2)].copy()
    work["session"] = str(session).upper()
    return work


def _session_features(laps: pd.DataFrame) -> pd.DataFrame:
    if laps.empty:
        return pd.DataFrame()

    rows = []
    for (drv, stint_id, comp), stint in laps.groupby(["Driver", "Stint", "compound_norm"], dropna=False, sort=False):
        stint = stint.sort_values("tyre_age")
        if len(stint) < 5:
            continue
        pace = pd.to_numeric(stint["lap_sec"], errors="coerce")
        slope = _fit_slope(pd.to_numeric(stint["tyre_age"], errors="coerce"), pace)
        q = pace.quantile([0.25, 0.75]) if pace.notna().sum() >= 2 else pd.Series([np.nan, np.nan], index=[0.25, 0.75])
        best3 = pace.rolling(3, min_periods=3).mean().min()
        rows.append(
            {
                "Driver": str(drv),
                "session": str(stint["session"].iloc[0]),
                "session_ord": int(SESSION_PRIORITY.get(str(stint["session"].iloc[0]), 0)),
                "compound": str(comp),
                "longrun_pace_s": float(pace.median()),
                "longrun_deg_s_per_lap": float(slope) if np.isfinite(slope) else np.nan,
                "longrun_iqr_s": float(q.loc[0.75] - q.loc[0.25]) if q.notna().all() else np.nan,
                "longrun_laps": int(len(stint)),
                "shortrun_best3_s": float(best3) if pd.notna(best3) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _select_driver_rows(stints: pd.DataFrame) -> pd.DataFrame:
    if stints.empty:
        return pd.DataFrame()

    rows = []
    for drv, sub in stints.groupby("Driver", sort=False):
        longrun = sub.sort_values(["longrun_laps", "session_ord", "shortrun_best3_s"], ascending=[False, False, True]).iloc[0]
        shortrun = sub.sort_values(["shortrun_best3_s", "session_ord"], ascending=[True, False]).iloc[0]
        rows.append(
            {
                "Driver": str(drv),
                "prac_pre_dry_runs_n": int(len(sub)),
                "prac_pre_longrun_pace_s": float(longrun["longrun_pace_s"]),
                "prac_pre_longrun_deg_s_per_lap": float(longrun["longrun_deg_s_per_lap"]) if pd.notna(longrun["longrun_deg_s_per_lap"]) else np.nan,
                "prac_pre_longrun_iqr_s": float(longrun["longrun_iqr_s"]) if pd.notna(longrun["longrun_iqr_s"]) else np.nan,
                "prac_pre_longrun_laps": int(longrun["longrun_laps"]),
                "prac_pre_longrun_session_ord": int(longrun["session_ord"]),
                "prac_pre_shortrun_best3_s": float(shortrun["shortrun_best3_s"]) if pd.notna(shortrun["shortrun_best3_s"]) else np.nan,
                "prac_pre_shortrun_session_ord": int(shortrun["session_ord"]),
            }
        )
    return pd.DataFrame(rows)


def featurize(ctx: dict) -> pd.DataFrame:
    raw_dir = Path(ctx.get("raw_dir", "data/raw_csv"))
    year = int(ctx["year"])
    rnd = int(ctx["round"])
    sessions: Sequence[str] = ctx.get("practice_sessions", ("FP3", "FP2", "FP1"))

    frames = []
    for session in sessions:
        laps = _load_session_laps(raw_dir, year, rnd, str(session).upper())
        if laps.empty:
            continue
        feats = _session_features(laps)
        if not feats.empty:
            frames.append(feats)
    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    out = _select_driver_rows(merged)
    if out.empty:
        return out

    drivers = _current_drivers(raw_dir, year, rnd, ctx.get("drivers"))
    if drivers:
        base = pd.DataFrame({"Driver": drivers})
        out = base.merge(out, on="Driver", how="left")
    return out
