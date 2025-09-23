#!/usr/bin/env python3
"""
Tyre priors (PRE; FastF1‑aligned, no leakage).

Works with datasets exported by export_last_two_years.py:
  • races.csv                              – schedule snapshot (year, round, raceId, names)
  • laps_{Y}_{R}.csv                        – per‑lap table (has LapNumber, Stint, Compound, milliseconds)
  • pit_stops_{Y}_{R}.csv                   – derived from laps (lap, duration_ms)
  • stints_{Y}_{R}.csv                      – stint summary (Stint×Compound×laps)
  • results_{Y}_{R}.csv / entrylist_{Y}_{R}.csv – to get current roster (Driver = Abbreviation)
  • meta_{Y}_{R}.csv / schedule_{Y}.csv     – used to derive event slug (same‑track history)

Outputs (one row per current Driver):
  - compound_mix_priors_S/M/H : expected fraction of race laps on S/M/H (same‑track history if available; else recent window)
  - tyre_delta_priors_s_SM    : expected pace delta (S − M) in seconds per lap
  - tyre_delta_priors_s_MH    : expected pace delta (M − H) in seconds per lap
  - expected_deg_S/M/H        : expected degradation slope per compound (field baseline × temp adjustment)

If data are missing, the module still returns a non‑empty frame for the current roster with NaNs in feature columns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import numpy as np
import pandas as pd

try:
    from .utils import read_csv_if_exists
    from .track_onehot import featurize as feat_track
    from .weather_basic import featurize as feat_weather
    from .history_form import featurize as feat_history
except Exception:  # pragma: no cover
    def read_csv_if_exists(p: Path) -> pd.DataFrame:  # type: ignore
        return pd.read_csv(p) if Path(p).exists() else pd.DataFrame()
    def feat_track(ctx: dict) -> pd.DataFrame:  # type: ignore
        return pd.DataFrame()
    def feat_weather(ctx: dict) -> pd.DataFrame:  # type: ignore
        return pd.DataFrame()
    def feat_history(ctx: dict) -> pd.DataFrame:  # type: ignore
        return pd.DataFrame()

__all__ = ["featurize"]

# ----------------------------- helpers -----------------------------

def _asof_mask(races: pd.DataFrame, year: int, rnd: int) -> pd.Series:
    y = pd.to_numeric(races.get("year"), errors="coerce")
    r = pd.to_numeric(races.get("round"), errors="coerce")
    return (y < int(year)) | ((y == int(year)) & (r < int(rnd)))


def _list_prev_races(races: pd.DataFrame, year: int, rnd: int, prevN: int) -> List[Tuple[int,int]]:
    if races is None or races.empty:
        return []
    past = races.loc[_asof_mask(races, year, rnd), ["year","round"]].dropna()
    if past.empty:
        return []
    past = past.astype(int).sort_values(["year","round"]).tail(int(prevN))
    return list(map(tuple, past.values.tolist()))


def _slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _event_slug(raw_dir: Path, y: int, r: int) -> str:
    meta = read_csv_if_exists(raw_dir / f"meta_{y}_{r}.csv")
    if not meta.empty:
        for c in ("EventName","OfficialEventName","Location","CircuitName","Name"):
            if c in meta.columns and meta[c].notna().any():
                return _slugify(str(meta[c].dropna().iloc[0]))
    sch = read_csv_if_exists(raw_dir / f"schedule_{y}.csv")
    if not sch.empty:
        rr = sch.loc[pd.to_numeric(sch.get("RoundNumber") or sch.get("round"), errors="coerce") == int(r)]
        if not rr.empty:
            for c in ("EventName","OfficialEventName","Location"):
                if c in rr.columns and rr[c].notna().any():
                    return _slugify(str(rr[c].dropna().iloc[0]))
    return f"{y}_{r}"


def _ensure_driver(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if "Driver" in df.columns:
        return df
    for c in ("Abbreviation","code","driverRef"):
        if c in df.columns:
            return df.rename(columns={c:"Driver"})
    return df


def _load_laps(raw_dir: Path, y: int, r: int) -> pd.DataFrame:
    df = read_csv_if_exists(raw_dir / f"laps_{y}_{r}.csv")
    if df.empty:
        return df
    df = _ensure_driver(df)
    if "LapNumber" in df.columns and "lap" not in df.columns:
        df = df.rename(columns={"LapNumber":"lap"})
    if "milliseconds" not in df.columns:
        if "LapTime" in df.columns:
            df["milliseconds"] = pd.to_timedelta(df["LapTime"], errors="coerce").dt.total_seconds() * 1000.0
        else:
            df["milliseconds"] = np.nan
    df["lap"] = pd.to_numeric(df.get("lap"), errors="coerce").astype("Int64")
    return df


def _load_pits(raw_dir: Path, y: int, r: int) -> pd.DataFrame:
    for name in (f"pit_stops_{y}_{r}.csv", f"pitstops_{y}_{r}.csv", f"pitStops_{y}_{r}.csv"):
        df = read_csv_if_exists(raw_dir / name)
        if not df.empty:
            break
    else:
        df = pd.DataFrame()
    if df.empty:
        return df
    df = _ensure_driver(df)
    if "lap" not in df.columns and "LapNumber" in df.columns:
        df = df.rename(columns={"LapNumber":"lap"})
    df["lap"] = pd.to_numeric(df.get("lap"), errors="coerce").astype("Int64")
    return df


def _load_stints(raw_dir: Path, y: int, r: int) -> pd.DataFrame:
    df = read_csv_if_exists(raw_dir / f"stints_{y}_{r}.csv")
    if df.empty:
        return df
    df = _ensure_driver(df)
    return df


def _clean_laps_with_compound(laps: pd.DataFrame, pits: pd.DataFrame) -> pd.DataFrame:
    if laps is None or laps.empty:
        return pd.DataFrame(columns=["raceId","Driver","lap","milliseconds","Compound"])  
    df = laps.copy()
    # map compound to S/M/H
    comp_col = None
    for c in ("Compound","compound","Tyre","tyre"):
        if c in df.columns:
            comp_col = c; break
    if comp_col is None:
        return pd.DataFrame(columns=["raceId","Driver","lap","milliseconds","Compound"])  
    m = {"SOFT":"S","MEDIUM":"M","HARD":"H","Soft":"S","Medium":"M","Hard":"H","S":"S","M":"M","H":"H"}
    df["C"] = df[comp_col].astype(str).map(lambda x: m.get(x, np.nan))
    df = df.dropna(subset=["C"]).copy()

    # exclude in/out laps using pit table
    if pits is not None and not pits.empty and {"Driver","lap"}.issubset(pits.columns):
        ex = pits[["Driver","lap"]].dropna().copy()
        ex["lap"] = pd.to_numeric(ex["lap"], errors="coerce").astype("Int64")
        ex = ex.dropna(subset=["lap"]).astype({"lap":int})
        excl = pd.concat([ex.assign(excl=ex["lap"]), ex.assign(excl=ex["lap"]+1)])[ ["Driver","excl"] ]
        df = df.merge(excl, left_on=["Driver","lap"], right_on=["Driver","excl"], how="left")
        df = df[df["excl"].isna()].drop(columns=["excl"])  # keep clean laps

    return df.rename(columns={"C":"Compound"})[ [c for c in ("raceId","Driver","lap","milliseconds","Compound") if c in df.columns] ]


# ----------------------------- core calcs -----------------------------

def _compound_mix_per_race_from_laps(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "Compound" not in df.columns:
        return pd.DataFrame(columns=["raceId","frac_S","frac_M","frac_H"])  
    mix = (df.groupby(["raceId","Compound"])  # race‑level mix across field
             .size().rename("laps").reset_index()
             .pivot(index="raceId", columns="Compound", values="laps").fillna(0))
    mix = mix.div(mix.sum(axis=1), axis=0)
    for c in ("S","M","H"):
        if c not in mix.columns:
            mix[c] = 0.0
    return mix.reset_index().rename(columns={"S":"frac_S","M":"frac_M","H":"frac_H"})


def _compound_mix_per_race_from_stints(st: pd.DataFrame) -> pd.DataFrame:
    if st is None or st.empty:
        return pd.DataFrame(columns=["raceId","frac_S","frac_M","frac_H"])  
    sf = st.copy()
    comp_col = None
    for c in ("Compound","compound","Tyre","tyre"):
        if c in sf.columns:
            comp_col = c; break
    if comp_col is None or "laps" not in sf.columns:
        return pd.DataFrame(columns=["raceId","frac_S","frac_M","frac_H"])  
    m = {"SOFT":"S","MEDIUM":"M","HARD":"H","Soft":"S","Medium":"M","Hard":"H","S":"S","M":"M","H":"H"}
    sf["C"] = sf[comp_col].astype(str).map(lambda x: m.get(x, np.nan))
    sf = sf.dropna(subset=["C","laps"]).copy()
    sf["laps"] = pd.to_numeric(sf["laps"], errors="coerce").fillna(0).astype(float)
    mix = (sf.groupby(["raceId","C"])['laps'].sum().reset_index()
             .pivot(index="raceId", columns="C", values="laps").fillna(0))
    mix = mix.div(mix.sum(axis=1), axis=0)
    for c in ("S","M","H"):
        if c not in mix.columns:
            mix[c] = 0.0
    return mix.reset_index().rename(columns={"S":"frac_S","M":"frac_M","H":"frac_H"})


def _pace_deltas_per_race(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "Compound" not in df.columns:
        return pd.DataFrame(columns=["raceId","d_SM_s","d_MH_s"])  
    med = df.groupby(["raceId","Compound"])['milliseconds'].median().unstack("Compound")
    for c in ("S","M","H"):
        if c not in med.columns:
            med[c] = np.nan
    med = med.reset_index()
    med["d_SM_s"] = (med["S"] - med["M"]) / 1000.0
    med["d_MH_s"] = (med["M"] - med["H"]) / 1000.0
    return med[["raceId","d_SM_s","d_MH_s"]]


def _safe_scalar(df_or_series, col: Optional[str], default: float) -> float:
    try:
        if isinstance(df_or_series, pd.DataFrame):
            s = df_or_series[col] if col is not None and col in df_or_series.columns else None
        elif isinstance(df_or_series, pd.Series):
            s = df_or_series
        else:
            s = None
        v = float(pd.to_numeric(s, errors="coerce").dropna().iloc[0]) if s is not None else float(default)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _current_drivers(raw_dir: Path, year: int, rnd: int) -> List[str]:
    for src in (f"results_{year}_{rnd}.csv", f"results_{year}_{rnd}_R.csv", f"entrylist_{year}_{rnd}_R.csv", f"entrylist_{year}_{rnd}_Q.csv"):
        df = read_csv_if_exists(raw_dir / src)
        if not df.empty:
            for c in ("Abbreviation","Driver","code","driverRef"):
                if c in df.columns and df[c].notna().any():
                    return df[c].astype(str).dropna().drop_duplicates().tolist()
    laps = read_csv_if_exists(raw_dir / f"laps_{year}_{rnd}.csv")
    if not laps.empty:
        for c in ("Abbreviation","Driver"):
            if c in laps.columns:
                return laps[c].astype(str).dropna().drop_duplicates().tolist()
    return []

# ----------------------------- main -----------------------------

def featurize(ctx: dict) -> pd.DataFrame:
    raw_dir = Path(ctx.get("raw_dir", "data/raw_csv"))
    year = int(ctx["year"]); rnd = int(ctx["round"])

    races = read_csv_if_exists(raw_dir / "races.csv")

    # choose history races: prefer same‑track, else last 8 overall strictly before (year, round)
    prev = _list_prev_races(races, year, rnd, prevN=12) if not races.empty else []
    cur_slug = _event_slug(raw_dir, year, rnd)
    same: List[Tuple[int,int]] = []
    if prev:
        for (y,r) in prev:
            if _event_slug(raw_dir, y, r) == cur_slug:
                same.append((y,r))
    use = same if len(same) >= 3 else prev[-8:]

    # build per‑race compound mix and deltas
    mix_rows: List[pd.DataFrame] = []
    delta_rows: List[pd.DataFrame] = []
    for (y,r) in use:
        laps = _load_laps(raw_dir, y, r)
        pits = _load_pits(raw_dir, y, r)
        clean = _clean_laps_with_compound(laps, pits)
        if not clean.empty:
            mix = _compound_mix_per_race_from_laps(clean)
            if not mix.empty:
                mix_rows.append(mix)
            dels = _pace_deltas_per_race(clean)
            if not dels.empty:
                delta_rows.append(dels)
        else:
            st = _load_stints(raw_dir, y, r)
            if not st.empty:
                mix = _compound_mix_per_race_from_stints(st)
                if not mix.empty:
                    mix_rows.append(mix)
            # deltas из стинтов нельзя получить — пропускаем

    if mix_rows:
        mix_all = pd.concat(mix_rows, ignore_index=True)
        frac_S = float(pd.to_numeric(mix_all["frac_S"], errors="coerce").median())
        frac_M = float(pd.to_numeric(mix_all["frac_M"], errors="coerce").median())
        frac_H = float(pd.to_numeric(mix_all["frac_H"], errors="coerce").median())
    else:
        frac_S = frac_M = frac_H = np.nan

    if delta_rows:
        dd = pd.concat(delta_rows, ignore_index=True)
        d_SM = float(pd.to_numeric(dd["d_SM_s"], errors="coerce").median())
        d_MH = float(pd.to_numeric(dd["d_MH_s"], errors="coerce").median())
    else:
        d_SM = d_MH = np.nan

    # degradation baselines (field median) with temperature adjustment
    try:
        w_now = feat_weather({**ctx, "mode": "forecast"})
    except Exception:
        w_now = pd.DataFrame()
    track_temp = _safe_scalar(w_now, "weather_track_temp_C", 25.0)
    if not np.isfinite(track_temp) or track_temp == 0:
        track_temp = _safe_scalar(w_now, "track_temp_C", 25.0)

    hist = pd.DataFrame()
    try:
        hist = feat_history(ctx)
    except Exception:
        hist = pd.DataFrame()

    def _pick_deg(df: pd.DataFrame, keys: List[str]) -> float:
        for k in keys:
            if k in df.columns:
                v = float(pd.to_numeric(df[k], errors="coerce").median())
                if np.isfinite(v):
                    return v
        return np.nan

    if hist is None or hist.empty:
        base_S = base_M = base_H = np.nan
    else:
        base_S = _pick_deg(hist, ["avg_deg_soft_prev3", "deg_slope_hist_S", "deg_soft_prev3", "deg_slope_mean_S"]) 
        base_M = _pick_deg(hist, ["avg_deg_medium_prev3", "deg_slope_hist_M", "deg_medium_prev3", "deg_slope_mean_M"]) 
        base_H = _pick_deg(hist, ["avg_deg_hard_prev3", "deg_slope_hist_H", "deg_hard_prev3", "deg_slope_mean_H"]) 
        if not np.isfinite(base_S) and not np.isfinite(base_M) and not np.isfinite(base_H):
            # fallback to a single overall slope if present
            g = _pick_deg(hist, ["deg_slope_mean", "deg_slope_hist", "deg_prev3"])
            base_S = base_M = base_H = g if np.isfinite(g) else np.nan

    coef = float(ctx.get("deg_temp_coef", 0.015))  # per +1°C vs 25°C
    td = float(track_temp) - 25.0
    exp_S = base_S * (1.0 + coef * td) if np.isfinite(base_S) else np.nan
    exp_M = base_M * (1.0 + coef * td) if np.isfinite(base_M) else np.nan
    exp_H = base_H * (1.0 + coef * td) if np.isfinite(base_H) else np.nan

    # roster for current race
    drivers = _current_drivers(raw_dir, year, rnd)
    if not drivers:
        return pd.DataFrame()  # nothing to broadcast onto

    out = pd.DataFrame({"Driver": drivers})
    out["compound_mix_priors_S"] = frac_S
    out["compound_mix_priors_M"] = frac_M
    out["compound_mix_priors_H"] = frac_H
    out["tyre_delta_priors_s_SM"] = d_SM
    out["tyre_delta_priors_s_MH"] = d_MH
    out["expected_deg_S"] = exp_S
    out["expected_deg_M"] = exp_M
    out["expected_deg_H"] = exp_H

    keep = [
        "Driver",
        "compound_mix_priors_S", "compound_mix_priors_M", "compound_mix_priors_H",
        "tyre_delta_priors_s_SM", "tyre_delta_priors_s_MH",
        "expected_deg_S", "expected_deg_M", "expected_deg_H",
    ]
    for c in keep:
        if c not in out.columns:
            out[c] = np.nan
    return out[keep]


if __name__ == "__main__":
    ctx = {"raw_dir": "data/raw_csv", "year": 2024, "round": 2}
    print(featurize(ctx).head())
