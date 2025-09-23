from __future__ import annotations
"""
Leak‑safe weather features (pre‑race)
------------------------------------
Builds session‑level weather features for a target race BEFORE it happens.

Key guarantees
- Default mode is STRICT pre‑race: uses only forecast files; NEVER falls back to actuals.
- Roster comes from entrylist/results_Q (no reading current race results/laps).
- If no forecast is available, returns driver‑level rows with NaNs (pipeline‑safe, no leakage).

File patterns supported (under raw_dir)
- weather_forecast_{year}_{rnd}.csv          # preferred (pre‑race)
- weather_{year}_{rnd}.csv                    # actual (ONLY if explicitly allowed)
- entrylist_{year}_{rnd}_Q.csv | entrylist_{year}_{rnd}.csv | results_{year}_{rnd}_Q.csv  # roster only
- meta_{year}_{rnd}.csv | meta_{year}_{rnd}_Q.csv  # optional: to trim forecast to session window

Returned columns (prefixed with weather_pre_)
- weather_pre_air_temp_mean, weather_pre_air_temp_std
- weather_pre_track_temp_mean (if available)
- weather_pre_humidity_mean
- weather_pre_wind_kph_mean
- weather_pre_rain_prob_p75 (if present)
- weather_pre_records_n  (number of rows used)

Usage
-----
featurize({
    "raw_dir": "/path/to/csvs",
    "year": 2024,
    "round": 1,
    "session": "R",                   # optional ("R" by default)
    "mode": "forecast",               # "forecast" | "actual" | "auto" (default: forecast)
    "allow_fallback_actual": False,    # if True and mode=="auto", can use actuals (NOT pre‑race safe)
    "drivers": ["VER", "LEC", ...],   # optional; otherwise read entrylist/results_Q
}) -> DataFrame with one row per Driver
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import re

# ---------------------- I/O helpers ----------------------

def _read_csv(p: Path) -> pd.DataFrame:
    try:
        if not p.exists():
            return pd.DataFrame()
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _pick_first_existing(paths: Sequence[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None

# -------------------- Roster detection -------------------

_ROSTER_FILES = (
    "entrylist_{y}_{r}_Q.csv",
    "entrylist_{y}_{r}.csv",
    "results_{y}_{r}_Q.csv",  # safe for roster only
)

_DRIVER_COLS = ("Driver", "Abbreviation", "driverRef", "DriverRef", "DriverCode", "BroadcastName")


def _load_roster(raw_dir: Path, year: int, rnd: int, drivers: Optional[Sequence[str]]) -> List[str]:
    if drivers:
        return list(dict.fromkeys([str(d) for d in drivers]))
    for pat in _ROSTER_FILES:
        df = _read_csv(raw_dir / pat.format(y=year, r=rnd))
        if not df.empty:
            for c in _DRIVER_COLS:
                if c in df.columns:
                    vals = (
                        df[c]
                        .astype(str)
                        .dropna()
                        .str.strip()
                        .replace({"nan": np.nan})
                        .dropna()
                        .drop_duplicates()
                        .tolist()
                    )
                    if vals:
                        return vals
    return []

# ---------------- Session window (optional) ---------------

_META_FILES = (
    "meta_{y}_{r}.csv",
    "meta_{y}_{r}_Q.csv",
)


def _session_window(raw_dir: Path, year: int, rnd: int, session: str) -> Tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Try to get UTC window [start,end] for the session from meta files.
    This is best‑effort and tolerant: returns (None, None) if not available.
    Expected columns (any of): Session/Phase, StartUtc/EndUtc, Start/End, DateStart/DateEnd.
    """
    sess = str(session or "R").upper()
    for pat in _META_FILES:
        df = _read_csv(raw_dir / pat.format(y=year, r=rnd))
        if df.empty:
            continue
        # identify session rows (R/Qualifying/Q/Sprint/FP*). If no marker, use whole file.
        if any(c in df.columns for c in ("Session", "Phase")):
            key = "Session" if "Session" in df.columns else "Phase"
            mask = df[key].astype(str).str.upper().str.startswith(sess[0])  # R/Q/S
            sub = df[mask].copy()
            if sub.empty:
                sub = df.copy()
        else:
            sub = df.copy()
        # try common datetime columns
        for sc, ec in (
            ("StartUtc", "EndUtc"), ("StartUTC", "EndUTC"),
            ("Start", "End"), ("DateStart", "DateEnd"),
        ):
            if sc in sub.columns and ec in sub.columns:
                s = pd.to_datetime(sub[sc], errors="coerce", utc=True)
                e = pd.to_datetime(sub[ec], errors="coerce", utc=True)
                if s.notna().any() and e.notna().any():
                    return s.min(), e.max()
    return None, None

# -------------------- Weather parsing --------------------

_TEMP_COLS = ("AirTemp", "AirTemp_C", "Temperature", "AirTemperature", "temp", "TempC")
_TRACK_TEMP_COLS = ("TrackTemp", "TrackTemp_C", "TrackTemperature")
_HUM_COLS = ("Humidity", "RH", "RelHumidity")
_WIND_COLS = ("WindKph", "Wind_kph", "WindSpeed", "WindSpeedKph", "WindSpeedKMH")
_RAIN_P_COLS = ("RainProb", "PrecipProb", "PrecipitationProb", "rain_probability")
_TIME_COLS = ("Utc", "UTC", "Timestamp", "Time", "DateTime")


def _coerce_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in _TIME_COLS:
        if c in df.columns:
            ts = pd.to_datetime(df[c], errors="coerce", utc=True)
            if ts.notna().any():
                df["_ts"] = ts
                break
    return df


def _select_cols(df: pd.DataFrame, cands: Sequence[str]) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    # case‑insensitive match
    lower = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _aggregate_window(df: pd.DataFrame, t0: pd.Timestamp | None, t1: pd.Timestamp | None) -> Dict[str, float]:
    """Compute aggregates inside [t0, t1] if timestamps are present, else over full table."""
    d = _coerce_time(df)
    if "_ts" in d.columns and (t0 is not None and t1 is not None):
        d = d[(d["_ts"] >= t0) & (d["_ts"] <= t1)].copy()
        if d.empty:  # if filter erased everything, fall back to full
            d = _coerce_time(df)

    out: Dict[str, float] = {}

    def num(colnames: Sequence[str], feat: str, fn) -> None:
        c = _select_cols(d, colnames)
        if c is None:
            out[feat] = np.nan
            return
        v = pd.to_numeric(d[c], errors="coerce")
        out[feat] = float(fn(v)) if v.notna().any() else np.nan

    num(_TEMP_COLS, "weather_pre_air_temp_mean", lambda s: s.mean())
    num(_TEMP_COLS, "weather_pre_air_temp_std", lambda s: s.std(ddof=0))
    # track temp optional
    c_track = _select_cols(d, _TRACK_TEMP_COLS)
    if c_track:
        v = pd.to_numeric(d[c_track], errors="coerce")
        out["weather_pre_track_temp_mean"] = float(v.mean()) if v.notna().any() else np.nan
    else:
        out["weather_pre_track_temp_mean"] = np.nan

    num(_HUM_COLS, "weather_pre_humidity_mean", lambda s: s.mean())
    num(_WIND_COLS, "weather_pre_wind_kph_mean", lambda s: s.mean())

    # rain prob p75 if present
    c_rp = _select_cols(d, _RAIN_P_COLS)
    if c_rp:
        v = pd.to_numeric(d[c_rp], errors="coerce")
        out["weather_pre_rain_prob_p75"] = float(v.quantile(0.75)) if v.notna().any() else np.nan
    else:
        out["weather_pre_rain_prob_p75"] = np.nan

    out["weather_pre_records_n"] = int(len(d))
    return out

# ------------------------ Public API ----------------------

@dataclass
class WeatherOptions:
    mode: str = "forecast"           # "forecast" | "actual" | "auto"
    allow_fallback_actual: bool = False  # only used if mode=="auto"


def featurize(ctx: Dict) -> pd.DataFrame:
    """Leak‑safe weather features repeated per driver.

    Context keys
    ------------
    raw_dir: str | Path
    year: int
    round: int
    session: str = "R" (optional)
    mode: "forecast" | "actual" | "auto"  (default "forecast")
    allow_fallback_actual: bool (default False)
    drivers: Optional[List[str]]
    """
    raw_dir = Path(ctx.get("raw_dir", "."))
    year = int(ctx["year"])  # required
    rnd = int(ctx["round"])  # required
    session = str(ctx.get("session", "R"))

    opt = WeatherOptions(
        mode=str(ctx.get("mode", "forecast")).lower(),
        allow_fallback_actual=bool(ctx.get("allow_fallback_actual", False)),
    )

    # --- choose weather file(s) ---
    forecast_p = raw_dir / f"weather_forecast_{year}_{rnd}.csv"
    actual_p   = raw_dir / f"weather_{year}_{rnd}.csv"

    weather_path: Optional[Path] = None
    if opt.mode == "forecast":
        weather_path = forecast_p if forecast_p.exists() else None
    elif opt.mode == "actual":
        weather_path = actual_p if actual_p.exists() else None
    else:  # auto
        if forecast_p.exists():
            weather_path = forecast_p
        elif opt.allow_fallback_actual and actual_p.exists():
            weather_path = actual_p
        else:
            weather_path = None

    # --- roster ---
    drivers = _load_roster(raw_dir, year, rnd, ctx.get("drivers"))
    base = pd.DataFrame({"Driver": drivers}) if drivers else pd.DataFrame()

    # If no weather data, return neutral rows (NaNs) for roster (or empty DF if no roster)
    if not weather_path:
        if base.empty:
            return base
        feats = {
            "weather_pre_air_temp_mean": np.nan,
            "weather_pre_air_temp_std": np.nan,
            "weather_pre_track_temp_mean": np.nan,
            "weather_pre_humidity_mean": np.nan,
            "weather_pre_wind_kph_mean": np.nan,
            "weather_pre_rain_prob_p75": np.nan,
            "weather_pre_records_n": 0,
        }
        for k, v in feats.items():
            base[k] = v
        return base

    # --- aggregate weather within session window if possible ---
    dfw = _read_csv(weather_path)
    t0, t1 = _session_window(raw_dir, year, rnd, session=session)
    agg = _aggregate_window(dfw, t0, t1)

    if base.empty:
        # No roster → return single row (session‑level), still leak‑safe
        return pd.DataFrame([agg])

    # broadcast to drivers
    for k, v in agg.items():
        base[k] = v
    return base


# ----------------------- CLI wrapper ----------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser("Leak‑safe weather features (pre‑race)")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--session", default="R")
    ap.add_argument("--mode", default="forecast", choices=["forecast", "actual", "auto"])
    ap.add_argument("--allow-fallback-actual", action="store_true")
    ap.add_argument("--drivers-csv", default=None, help="Optional CSV with a 'Driver' column")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    roster = None
    if args.drivers_csv:
        r = _read_csv(Path(args.drivers_csv))
        for c in _DRIVER_COLS:
            if c in r.columns:
                roster = r[c].astype(str).dropna().unique().tolist()
                break

    out = featurize({
        "raw_dir": args.raw_dir,
        "year": args.year,
        "round": args.round,
        "session": args.session,
        "mode": args.mode,
        "allow_fallback_actual": bool(args.allow_fallback_actual),
        "drivers": roster,
    })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Saved weather features to {args.out} (rows={len(out)})")
