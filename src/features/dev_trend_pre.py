#!/usr/bin/env python3
"""
Pace development trends (PRE; FastF1‑aligned, no leakage).

Works with datasets exported by export_last_two_years.py:
  • races.csv                         – schedule snapshot (year, round)
  • laps_{Y}_{R}.csv                  – per‑lap times (LapNumber/LapTime or milliseconds)
  • pit_stops_{Y}_{R}.csv             – per‑stop (lap, duration_ms) — optional for clean‑lap filtering
  • results_{Y}_{R}.csv / entrylist_{Y}_{R}_{Q|R}.csv – map Driver(=Abbreviation) → TeamName for that race

Outputs (one row per current Driver):
  - driver_trend             : LS slope of driver's per‑race pace_z over last K races
  - team_dev_trend           : LS slope of team's per‑race mean pace_z over last K races (broadcast to drivers)
  - stability_delta_vs_tm    : robust dispersion (IQR/1.35) of (driver pace_z − team pace_z) over last W races

Notes:
  - pace_z computed per race by z‑scoring *driver* trimmed‑median clean‑lap time (lower ms → higher z)
  - clean laps exclude in‑lap (L) and out‑lap (L+1) if pit table is available; otherwise use all laps
  - never returns empty if текущий ростер найден (выдаёт каркас с NaN‑ами)

Tunables via ctx:
  - dev_trend_window (K, default=6)
  - stability_window (W, default=8)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import re

try:
    from .utils import read_csv_if_exists
except Exception:  # pragma: no cover
    def read_csv_if_exists(p: Path) -> pd.DataFrame:  # type: ignore
        import pandas as _pd
        return _pd.read_csv(p) if Path(p).exists() else _pd.DataFrame()

__all__ = ["featurize"]

# ----------------------------- helpers -----------------------------

def _parse_fn_triplet(name: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"_(\d{4})_(\d{1,2})\.csv$", name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _asof_mask(races: pd.DataFrame, year: int, rnd: int) -> pd.Series:
    y = pd.to_numeric(races.get("year"), errors="coerce")
    r = pd.to_numeric(races.get("round"), errors="coerce")
    return (y < int(year)) | ((y == int(year)) & (r < int(rnd)))


def _order_key(y: int, r: int) -> int:
    return int(y) * 1000 + int(r)


def _to_ms(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().any():
        return s.astype(float)
    # try parse time strings like mm:ss.mmm
    try:
        return pd.to_timedelta(series, errors="coerce").dt.total_seconds() * 1000.0
    except Exception:
        return pd.Series(np.nan, index=series.index)


def _trimmed_median_ms(a: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    if x.size == 0:
        return np.nan
    q05, q95 = np.nanpercentile(x, [5, 95])
    x = x[(x >= q05) & (x <= q95)]
    return float(np.nanmedian(x if x.size else a))


def _robust_std_iqr(a: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    if x.size == 0:
        return np.nan
    q75, q25 = np.nanpercentile(x, [75, 25])
    iqr = q75 - q25
    return float(iqr / 1.35) if iqr > 0 else float(np.nanstd(x, ddof=0))


def _lin_slope_last_k(vals: List[float], k: int) -> float:
    x = np.array([v for v in vals if np.isfinite(v)], dtype=float)
    if x.size < 3:
        return np.nan
    if x.size > k:
        x = x[-k:]
    t = np.arange(x.size, dtype=float)
    a, _b = np.polyfit(t, x, 1)
    return float(a)


def _ensure_driver(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if "Driver" in df.columns:
        return df
    for c in ("Abbreviation", "code", "driverRef", "DriverId", "driverId"):
        if c in df.columns:
            return df.rename(columns={c: "Driver"})
    return df

# ----------------------------- loaders -----------------------------

def _load_laps(raw_dir: Path, y: int, r: int) -> pd.DataFrame:
    df = read_csv_if_exists(raw_dir / f"laps_{y}_{r}.csv")
    if df.empty:
        return df
    df = _ensure_driver(df)
    if "LapNumber" in df.columns and "lap" not in df.columns:
        df = df.rename(columns={"LapNumber": "lap"})
    if "milliseconds" not in df.columns:
        if "LapTimeSec" in df.columns:
            df["milliseconds"] = pd.to_numeric(df["LapTimeSec"], errors="coerce") * 1000.0
        elif "LapTime" in df.columns:
            df["milliseconds"] = pd.to_timedelta(df["LapTime"], errors="coerce").dt.total_seconds() * 1000.0
        else:
            df["milliseconds"] = np.nan
    df["lap"] = pd.to_numeric(df.get("lap"), errors="coerce").astype("Int64")
    return df


def _load_pits(raw_dir: Path, y: int, r: int) -> pd.DataFrame:
    for name in (f"pit_stops_{y}_{r}.csv", f"pitstops_{y}_{r}.csv", f"pitStops_{y}_{r}.csv"):
        ps = read_csv_if_exists(raw_dir / name)
        if not ps.empty:
            break
    else:
        ps = pd.DataFrame()
    if ps.empty:
        return ps
    ps = _ensure_driver(ps)
    if "lap" not in ps.columns and "LapNumber" in ps.columns:
        ps = ps.rename(columns={"LapNumber": "lap"})
    ps["lap"] = pd.to_numeric(ps.get("lap"), errors="coerce").astype("Int64")
    return ps


def _driver_team_map(raw_dir: Path, y: int, r: int) -> pd.DataFrame:
    # prefer results for that race, fallback to entrylist
    res = read_csv_if_exists(raw_dir / f"results_{y}_{r}.csv")
    if res.empty:
        res = read_csv_if_exists(raw_dir / f"results_{y}_{r}_R.csv")
    ent = read_csv_if_exists(raw_dir / f"entrylist_{y}_{r}_R.csv")
    if ent.empty:
        ent = read_csv_if_exists(raw_dir / f"entrylist_{y}_{r}_Q.csv")

    if not res.empty:
        dcol = next((c for c in ("Abbreviation","Driver","code","driverRef") if c in res.columns), None)
        tcol = next((c for c in ("TeamName","Team","Constructor","constructorRef") if c in res.columns), None)
        if dcol and tcol:
            df = res[[dcol, tcol]].dropna().rename(columns={dcol: "Driver", tcol: "Team"})
            if not df.empty:
                return df.drop_duplicates()
    if not ent.empty:
        dcol = next((c for c in ("Abbreviation","Driver","code","driverRef") if c in ent.columns), None)
        tcol = next((c for c in ("TeamName","Team") if c in ent.columns), None)
        if dcol and tcol:
            return ent[[dcol, tcol]].dropna().rename(columns={dcol: "Driver", tcol: "Team"}).drop_duplicates()
    return pd.DataFrame(columns=["Driver","Team"])


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

# ----------------------------- per‑race paces -----------------------------

def _per_race_pace_z(raw_dir: Path, y: int, r: int) -> pd.DataFrame:
    laps = _load_laps(raw_dir, y, r)
    if laps.empty:
        return pd.DataFrame(columns=["Driver","pace_z"])  

    # mark in/out laps using pit table if available
    pits = _load_pits(raw_dir, y, r)
    if not pits.empty and {"Driver","lap"}.issubset(pits.columns):
        ex = pits[["Driver","lap"]].dropna().copy()
        ex["lap"] = pd.to_numeric(ex["lap"], errors="coerce").astype("Int64")
        ex = ex.dropna(subset=["lap"]).astype({"lap": int})
        excl = pd.concat([ex.assign(excl=ex["lap"]), ex.assign(excl=ex["lap"]+1)])[ ["Driver","excl"] ]
        laps = laps.merge(excl, left_on=["Driver","lap"], right_on=["Driver","excl"], how="left")
        laps = laps[laps["excl"].isna()].drop(columns=["excl"])  # clean laps

    # per‑driver trimmed median
    grp = (laps.dropna(subset=["milliseconds"]) 
               .groupby("Driver", as_index=False)["milliseconds"]
               .agg(med_clean_ms=_trimmed_median_ms))
    if grp.empty:
        return pd.DataFrame(columns=["Driver","pace_z"])  

    mu = float(pd.to_numeric(grp["med_clean_ms"], errors="coerce").mean())
    sd = float(pd.to_numeric(grp["med_clean_ms"], errors="coerce").std())
    if not np.isfinite(sd) or sd <= 0:
        grp["pace_z"] = 0.0
    else:
        grp["pace_z"] = -(grp["med_clean_ms"] - mu) / sd

    return grp[["Driver","pace_z"]]

# ----------------------------- main -----------------------------

def featurize(ctx: Dict) -> pd.DataFrame:
    raw_dir = Path(ctx.get("raw_dir", "data/raw_csv"))
    year = int(ctx.get("year")); rnd = int(ctx.get("round"))

    # choose history races strictly before (year, round)
    races = read_csv_if_exists(raw_dir / "races.csv")
    prev: List[Tuple[int,int]] = []
    if not races.empty and {"year","round"}.issubset(races.columns):
        rs = races.loc[_asof_mask(races, year, rnd), ["year","round"]].dropna().astype(int)
        prev = list(map(tuple, rs.sort_values(["year","round"]).values.tolist()))
    else:
        # infer from file names
        keys = []
        for p in raw_dir.glob("laps_*.csv"):
            pr = _parse_fn_triplet(p.name)
            if pr and ((pr[0] < year) or (pr[0] == year and pr[1] < rnd)):
                keys.append(pr)
        prev = sorted(set(keys))

    # keep recent window up to Kmax for stability & trend calcs
    K = int(ctx.get("dev_trend_window", 6))
    W = int(ctx.get("stability_window", 8))
    if prev:
        prev = prev[-max(K, W, 3):]  # need at least few races if available

    # build pace_z per race and team means
    per_race: List[pd.DataFrame] = []
    team_means: List[pd.DataFrame] = []
    for (y, r) in prev:
        dpace = _per_race_pace_z(raw_dir, y, r)
        if dpace.empty:
            continue
        dpace["_ord"] = _order_key(y, r)
        dpace["_yr"] = int(y); dpace["_rd"] = int(r)
        # team mapping for that race
        d2t = _driver_team_map(raw_dir, y, r)
        if not d2t.empty:
            dpace = dpace.merge(d2t, on="Driver", how="left")
            tmean = dpace.dropna(subset=["Team"]).groupby("Team", as_index=False)["pace_z"].mean().rename(columns={"pace_z":"team_pace_z"})
            tmean["_ord"] = _order_key(y, r)
            team_means.append(tmean)
        per_race.append(dpace)

    if not per_race:
        # no history → return skeleton for current drivers
        roster = _current_drivers(raw_dir, year, rnd)
        if not roster:
            return pd.DataFrame()
        out = pd.DataFrame({"Driver": roster})
        out["driver_trend"] = np.nan
        out["team_dev_trend"] = np.nan
        out["stability_delta_vs_tm"] = np.nan
        return out

    all_dr = pd.concat(per_race, ignore_index=True)
    all_tm = pd.concat(team_means, ignore_index=True) if team_means else pd.DataFrame(columns=["Team","team_pace_z","_ord"])

    # driver trend
    dtrend_rows = []
    for drv, g in all_dr.sort_values(["Driver","_ord"]).groupby("Driver"):
        slope = _lin_slope_last_k(g["pace_z"].tolist(), K)
        dtrend_rows.append({"Driver": drv, "driver_trend": slope})
    d_trend = pd.DataFrame(dtrend_rows)

    # team dev trend
    t_trend = pd.DataFrame(columns=["Team","team_dev_trend"])
    if not all_tm.empty:
        rows = []
        for tm, g in all_tm.sort_values(["Team","_ord"]).groupby("Team"):
            slope = _lin_slope_last_k(g["team_pace_z"].tolist(), K)
            rows.append({"Team": tm, "team_dev_trend": slope})
        t_trend = pd.DataFrame(rows)

    # stability of delta vs team
    stab_rows = []
    if "Team" in all_dr.columns and not all_tm.empty:
        # attach team mean per same race order
        tm_map = all_tm.set_index(["Team","_ord"]) ["team_pace_z"].to_dict()
        deltas = []
        for _, row in all_dr.iterrows():
            tm = row.get("Team")
            if pd.isna(tm):
                continue
            tp = tm_map.get((tm, row["_ord"]))
            if tp is None or not np.isfinite(tp):
                continue
            deltas.append({"Driver": row["Driver"], "_ord": row["_ord"], "delta": float(row["pace_z"]) - float(tp)})
        if deltas:
            dd = pd.DataFrame(deltas).sort_values(["Driver","_ord"]) 
            for drv, g in dd.groupby("Driver"):
                vals = g["delta"].tail(W)
                stab_rows.append({"Driver": drv, "stability_delta_vs_tm": _robust_std_iqr(vals)})
    stab = pd.DataFrame(stab_rows)

    # current roster for broadcasting team trend
    roster = _current_drivers(raw_dir, year, rnd)
    if not roster:
        # as a fallback, use all drivers seen in history
        roster = all_dr["Driver"].astype(str).dropna().drop_duplicates().tolist()

    out = pd.DataFrame({"Driver": roster})
    out = out.merge(d_trend, on="Driver", how="left")

    # join current team mapping to broadcast team trend
    d2t_now = _driver_team_map(raw_dir, year, rnd)
    if not d2t_now.empty and not t_trend.empty:
        out = out.merge(d2t_now, on="Driver", how="left").merge(t_trend, on="Team", how="left")
    else:
        out["team_dev_trend"] = np.nan

    out = out.merge(stab, on="Driver", how="left")

    keep = ["Driver","driver_trend","team_dev_trend","stability_delta_vs_tm"]
    for c in keep:
        if c not in out.columns:
            out[c] = np.nan
    return out[keep]


if __name__ == "__main__":
    ctx = {"raw_dir": "data/raw_csv", "year": 2024, "round": 3}
    print(featurize(ctx).head())
