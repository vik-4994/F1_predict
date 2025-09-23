#!/usr/bin/env python3
"""
reliability_risk_pre

Estimates per‑driver reliability risk (accident / mechanical / overall DNF)
for the target race using:
  • Historical DNF composition from Ergast‑style master CSVs
    (races.csv, results.csv, status.csv, drivers.csv)
  • Current race driver list from qualifying/results CSVs (or ctx)
  • Weather features from weather_basic (auto: forecast→actual fallback)
  • Simple track/SC/overtake priors from ctx if provided

Returns a DataFrame with columns:
  Driver, reliab_accident_risk, reliab_mech_risk, reliab_dnf_risk, reliab_temp_delta

Notes
- If master CSVs are missing, the module gracefully degrades to neutral priors.
- Weather columns expected from weather_basic: weather_track_temp_C, weather_rain_prob (optional).
- "Driver" is a display code (e.g., "LEC", "HAM"); mapping built from drivers.csv.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple, Optional

import numpy as np
import pandas as pd

# ---- optional import from local utils ----
try:
    from .utils import read_csv_if_exists
except Exception:  # pragma: no cover
    def read_csv_if_exists(p: Path) -> pd.DataFrame:
        try:
            import pandas as _pd
            return _pd.read_csv(p) if p.exists() else _pd.DataFrame()
        except Exception:
            return pd.DataFrame()

# weather_basic (same package)
try:
    from .weather_basic import featurize as feat_weather
except Exception:
    feat_weather = None  # type: ignore


# ---------------------------- helpers ----------------------------

def _pick_scalar(source: dict | pd.DataFrame, keys: Iterable[str], default: float) -> float:
    """Pick the first present key from DataFrame columns or dict and cast to float.
    If a DataFrame with one row is given, use the value from that row.
    """
    if isinstance(source, pd.DataFrame) and not source.empty:
        row = source.iloc[0]
        for k in keys:
            if k in row and pd.notna(row[k]):
                try:
                    return float(row[k])
                except Exception:
                    continue
    elif isinstance(source, dict):
        for k in keys:
            v = source.get(k, None)
            if v is not None and (not (isinstance(v, float) and np.isnan(v))):
                try:
                    return float(v)
                except Exception:
                    continue
    return float(default)


def _status_buckets(status_df: pd.DataFrame) -> Dict[int, str]:
    """Map statusId → bucket: 'finish' | 'accident' | 'mechanical' | 'other'."""
    if status_df.empty:
        return {}
    s = status_df.copy()
    s.columns = [c.lower() for c in s.columns]
    if "statusid" not in s.columns:
        # sometimes the file may already be joined; bail out
        return {}
    s["bucket"] = "other"
    txt = s["status"].astype(str).str.lower()
    acc_kw = ["accident", "collision", "spun off", "crash"]
    mech_kw = [
        "engine", "gear", "hydraul", "elect", "brake", "susp", "clutch",
        "overheat", "driveshaft", "exhaust", "fuel", "turbo", "power unit",
    ]
    s.loc[txt.str.contains("finished") | txt.str.contains("\+\d+ laps"), "bucket"] = "finish"
    s.loc[txt.str.contains("disqualified") | txt.str.contains("excluded"), "bucket"] = "other"
    if len(acc_kw):
        s.loc[np.logical_or.reduce([txt.str.contains(k) for k in acc_kw]), "bucket"] = "accident"
    if len(mech_kw):
        s.loc[np.logical_or.reduce([txt.str.contains(k) for k in mech_kw]), "bucket"] = "mechanical"
    return dict(zip(s["statusid"].astype(int), s["bucket"].astype(str)))


def _gather_current_drivers(raw_dir: Path, year: int, rnd: int, drivers_df: pd.DataFrame, ctx: dict) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """Return (cur, id2code)
    cur: DataFrame with columns [driverId, Driver]
    id2code: mapping driverId → code/driverRef/abbrev
    """
    # build id→code mapping from drivers.csv primarily
    id2code: Dict[int, str] = {}
    if not drivers_df.empty and "driverId" in drivers_df.columns:
        if "code" in drivers_df.columns and drivers_df["code"].notna().any():
            id2code = (
                drivers_df[["driverId", "code"]]
                .dropna().drop_duplicates().set_index("driverId")["code"].astype(str).to_dict()
            )
        elif "driverRef" in drivers_df.columns:
            id2code = (
                drivers_df[["driverId", "driverRef"]]
                .dropna().drop_duplicates().set_index("driverId")["driverRef"].astype(str).to_dict()
            )

    # try to read current race files to get roster
    candidates = [
        raw_dir / f"qualifying_{year}_{rnd}.csv",
        raw_dir / f"results_{year}_{rnd}.csv",
        raw_dir / f"laps_{year}_{rnd}.csv",
    ]
    cur = pd.DataFrame()
    for p in candidates:
        df = read_csv_if_exists(p)
        if df.empty:
            continue
        if "driverId" in df.columns:
            cur = df[["driverId"]].dropna().drop_duplicates()
            break
        for cand in ("Abbreviation", "Driver", "code", "driverRef", "BroadcastName"):
            if cand in df.columns:
                if not id2code:
                    # can't invert without drivers.csv
                    cur = df[[cand]].dropna().drop_duplicates().rename(columns={cand: "Driver"})
                else:
                    inv = {v: k for k, v in id2code.items()}
                    cur = df[[cand]].dropna().drop_duplicates().rename(columns={cand: "Driver"})
                    cur["driverId"] = cur["Driver"].map(inv)
                break
        if not cur.empty:
            break

    # fallback: ctx["drivers"] list
    if cur.empty and "drivers" in ctx:
        seq = pd.Series(ctx["drivers"], name="Driver", dtype=str)
        cur = pd.DataFrame({"Driver": seq})
        if id2code:
            inv = {v: k for k, v in id2code.items()}
            cur["driverId"] = cur["Driver"].map(inv)

    # ensure both columns present
    if "Driver" not in cur.columns and "driverId" in cur.columns and id2code:
        cur["Driver"] = cur["driverId"].map(id2code).astype(str)
    if "driverId" not in cur.columns and "Driver" in cur.columns and id2code:
        inv = {v: k for k, v in id2code.items()}
        cur["driverId"] = cur["Driver"].map(inv)

    cur = cur.drop_duplicates()
    return cur, id2code


@dataclass
class _HistRates:
    acc_prev10: float
    mech_prev10: float
    acc_season: float
    mech_season: float


def _compute_hist_rates(raw_dir: Path, year: int, rnd: int) -> pd.DataFrame:
    """Return per-driver historical rates using Ergast-style masters.
    Output columns: driverId, acc_prev10, mech_prev10, acc_season, mech_season
    """
    races   = read_csv_if_exists(raw_dir / "races.csv")
    results = read_csv_if_exists(raw_dir / "results.csv")
    status  = read_csv_if_exists(raw_dir / "status.csv")

    if races.empty or results.empty or ("statusId" not in results.columns and status.empty):
        return pd.DataFrame(columns=["driverId", "acc_prev10", "mech_prev10", "acc_season", "mech_season"])  # type: ignore

    # join results→races for chronology
    key = "raceId"
    cols = [c for c in ("raceId", "year", "round", "date") if c in races.columns]
    rj = results.merge(races[cols], on="raceId", how="left")

    # attach status bucket
    if "status" in rj.columns:
        # already textual; try to mimic Ergast 'status' text
        st = rj[["status"]].copy()
        st["statusId"] = rj.get("statusId", np.nan)
        status_map = _status_buckets(status if not status.empty else st)
    else:
        status_map = _status_buckets(status)
    if "statusId" in rj.columns and status_map:
        rj["_bucket"] = rj["statusId"].map(status_map)
    else:
        # fallback: treat non-finish positions with numeric 'position' as finish
        rj["_bucket"] = np.where(
            rj.get("positionText", "").astype(str).str.lower().eq("r"), "other", "finish"
        )

    # mark accident / mechanical
    rj["is_accident"]  = (rj["_bucket"] == "accident").astype(int)
    rj["is_mech"]      = (rj["_bucket"] == "mechanical").astype(int)
    rj["is_finish"]    = (rj["_bucket"] == "finish").astype(int)

    # order by date then round for rolling windows
    if "date" in rj.columns:
        rj["_ord"] = pd.to_datetime(rj["date"], errors="coerce")
    else:
        rj["_ord"] = rj["year"].astype(str) + "-" + rj["round"].astype(str)

    rj = rj.sort_values(["driverId", "_ord"])  # ascending

    # previous 10 appearances per driver (exclusive of current season's target round)
    mask_past = (rj["year"] < year) | ((rj["year"] == year) & (rj["round"] < rnd))
    past = rj.loc[mask_past].copy()

    def _roll_prev10(g: pd.DataFrame) -> pd.DataFrame:
        w = 10
        acc = g["is_accident"].rolling(w, min_periods=1).mean().shift(1)
        mech = g["is_mech"].rolling(w, min_periods=1).mean().shift(1)
        out = pd.DataFrame({"acc_prev10": acc, "mech_prev10": mech})
        out["raceId"] = g["raceId"].values
        out["driverId"] = g["driverId"].values
        return out

    prev10 = past.groupby("driverId", group_keys=False).apply(_roll_prev10)

    # season to date in current year
    cur_season = rj[rj["year"] == year].copy()
    cur_season = cur_season[cur_season["round"] < rnd]
    by_drv = (
        cur_season.groupby("driverId")[ ["is_accident", "is_mech"] ].mean().reset_index()
        .rename(columns={"is_accident": "acc_season", "is_mech": "mech_season"})
    )

    out = (
        prev10.merge(by_drv, on=["driverId"], how="left")
        [["driverId", "acc_prev10", "mech_prev10", "acc_season", "mech_season"]]
        .drop_duplicates("driverId")
    )
    return out


# ------------------------------ main ------------------------------

def featurize(ctx: dict) -> pd.DataFrame:
    raw_dir = Path(ctx.get("raw_dir", "data/raw"))
    year = int(ctx.get("year"))
    rnd  = int(ctx.get("round"))

    # read masters
    races   = read_csv_if_exists(raw_dir / "races.csv")
    results = read_csv_if_exists(raw_dir / "results.csv")
    status  = read_csv_if_exists(raw_dir / "status.csv")
    drivers = read_csv_if_exists(raw_dir / "drivers.csv")

    # weather (auto: forecast → actual)
    w = pd.DataFrame()
    if feat_weather is not None:
        try:
            w = feat_weather({**ctx, "mode": "auto"})
        except Exception:
            w = pd.DataFrame()

    # priors from ctx (safe defaults)
    sc_prob = float(ctx.get("track_sc_rate", ctx.get("sc_prob", 0.30)))
    overtake_difficulty = float(ctx.get("track_overtake_difficulty", 0.50))
    rain_prob = _pick_scalar(w, ("weather_rain_prob", "rain_prob"), 0.0)
    track_temp = _pick_scalar(w, ("weather_track_temp_C", "track_temp_C"), 25.0)
    temp_delta = (track_temp - 25.0) / 10.0
    chaos_index_forecast = float(np.clip(sc_prob * overtake_difficulty * rain_prob, 0.0, 1.0))

    # weights
    rw: Dict[str, float] = {
        "acc_chaos_mult": 0.75,
        "mech_temp_mult": 0.25,
        "mech_chaos_mult": 0.15,
    }
    # allow overrides
    rw.update(ctx.get("reliab_weights", {}))

    # historical base rates
    hist = _compute_hist_rates(raw_dir, year, rnd)

    # current roster
    cur, id2code = _gather_current_drivers(raw_dir, year, rnd, drivers, ctx)

    if cur.empty:
        return pd.DataFrame()

    # merge bases
    out = cur.merge(hist, on="driverId", how="left") if "driverId" in cur.columns else cur.copy()

    # fill neutral priors if no history
    for c in ("acc_prev10", "mech_prev10", "acc_season", "mech_season"):
        if c not in out.columns:
            out[c] = np.nan
    out[["acc_prev10", "mech_prev10"]] = out[["acc_prev10", "mech_prev10"]].fillna(0.05)
    out[["acc_season", "mech_season"]] = out[["acc_season", "mech_season"]].fillna(0.03)

    base_acc = 0.6 * out["acc_prev10"].astype(float) + 0.4 * out["acc_season"].astype(float)
    base_mech = 0.6 * out["mech_prev10"].astype(float) + 0.4 * out["mech_season"].astype(float)

    # adjustments
    acc = base_acc * (1.0 + rw["acc_chaos_mult"] * chaos_index_forecast)
    mech = base_mech * (1.0 + rw["mech_temp_mult"] * max(0.0, temp_delta) + rw["mech_chaos_mult"] * chaos_index_forecast)

    acc = np.clip(acc, 0.0, 0.80)
    mech = np.clip(mech, 0.0, 0.80)
    dnf = 1.0 - (1.0 - acc) * (1.0 - mech)

    out_cols = pd.DataFrame({
        "Driver": out.get("Driver", pd.Series(dtype=str)).astype(str) if "Driver" in out.columns else out.get("driverId", pd.Series(dtype=str)).map(id2code).astype(str),
        "reliab_accident_risk": acc.astype(float),
        "reliab_mech_risk": mech.astype(float),
        "reliab_dnf_risk": dnf.astype(float),
        "reliab_temp_delta": float(temp_delta),
    })

    # ensure Driver present
    if "Driver" not in out_cols.columns or out_cols["Driver"].isna().all():
        if "driverId" in out.columns and id2code:
            out_cols["Driver"] = out["driverId"].map(id2code).astype(str)

    # drop duplicates & tidy
    out_cols = out_cols.drop_duplicates(subset=["Driver"]).reset_index(drop=True)
    return out_cols


if __name__ == "__main__":
    # quick manual test (won't run in production pipeline)
    df = featurize({"raw_dir": "data/raw", "year": 2024, "round": 10})
    print(df.head())
