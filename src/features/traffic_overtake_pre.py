from __future__ import annotations
"""
traffic_overtake_pre.py

Pre‑race features from *historical races only* (strictly earlier than the current race):
  - lap1_gain_avg_prev10        : average (grid_pos - pos_after_lap1)
  - lap1_incident_rate_prev10   : share of past races with an incident on lap 1
  - traffic_penalty_s_prev10    : avg (dirty - clean) lap‑time delta, dirty if gap_ahead < 1.5s
  - net_pass_index_prev10       : avg net overtakes per 100 green laps (excludes pit laps & SC/VSC)

Robust to missing files/columns. If a metric cannot be computed with available data,
returns NaN for that metric. Uses only past races; no leakage from the current weekend.
"""
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math
import numpy as np
import pandas as pd
import re

# Prefer project utils; provide minimal fallbacks if unavailable
try:
    from .utils import read_csv_if_exists, ensure_driver_index, load_laps_enriched
except Exception:  # pragma: no cover
    def read_csv_if_exists(p: Path, **kwargs) -> pd.DataFrame:  # noqa: D401
        return pd.read_csv(p, **kwargs) if Path(p).exists() else pd.DataFrame()
    def ensure_driver_index(drivers: pd.Series, features: Dict[str, float]) -> pd.DataFrame:
        df = pd.DataFrame({"Driver": pd.Series(drivers, name="Driver").astype(str)})
        for k, v in features.items():
            df[k] = v if np.isscalar(v) else pd.Series(v).reindex(df.index).values
        return df
    def load_laps_enriched(raw_dir: Path, year: int, rnd: int) -> pd.DataFrame:
        for patt in (f"laps_{year}_{rnd}.csv", f"lap_times_{year}_{rnd}.csv"):
            df = read_csv_if_exists(Path(raw_dir) / patt)
            if not df.empty:
                return df
        return pd.DataFrame()

__all__ = ["featurize"]

# -------------------------- helpers --------------------------

def _list_prev_races(raw_dir: Path, year: int, rnd: int, k: int) -> List[Tuple[int,int]]:
    races = read_csv_if_exists(raw_dir / "races.csv")
    if not races.empty and {"year","round"}.issubset(races.columns):
        df = races.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values(["date","year","round"]).reset_index(drop=True)
        else:
            df = df.sort_values(["year","round"]).reset_index(drop=True)
        prev = df[(df["year"].astype(int) < year) | ((df["year"].astype(int) == year) & (df["round"].astype(int) < rnd))]
        tail = prev.tail(k)
        return list(zip(tail["year"].astype(int), tail["round"].astype(int)))
    # fallback from filenames
    cand = []
    for p in raw_dir.glob("results_*_*.csv"):
        m = re.match(r"results_(\d{4})_(\d{1,2})\.csv$", p.name)
        if m:
            y, r = int(m.group(1)), int(m.group(2))
            if (y < year) or (y == year and r < rnd):
                cand.append((y, r))
    cand.sort()
    return cand[-k:]


def _col(df: pd.DataFrame, options: Sequence[str]) -> Optional[str]:
    for c in options:
        if c in df.columns:
            return c
    return None


def _lap_col(df: pd.DataFrame) -> Optional[str]:
    return _col(df, ["Lap","lap","LapNumber","lap_number"])  # int


def _pos_col(df: pd.DataFrame) -> Optional[str]:
    return _col(df, ["Position","position","Pos","pos","PositionOrder"])  # int-like


def _drv_col(df: pd.DataFrame) -> Optional[str]:
    return _col(df, ["Driver","driver","driverId","DriverId","name"])  # string id/name


def _time_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in ["LapTimeSeconds","lap_time_s","LapTime","lap_time","LapTimeMs","laptime","time"] if c in df.columns]


def _gap_ahead_col(df: pd.DataFrame) -> Optional[str]:
    return _col(df, ["GapToAhead","gap_to_ahead","IntervalToCarAhead","interval_to_ahead","GapAhead","gapAhead","Interval","interval","GapToFront"])  # seconds


def _pit_flag(df: pd.DataFrame) -> Optional[str]:
    return _col(df, ["IsPitLap","is_pit","Pit","pit","in_pit"])  # bool


def _sc_flag(df: pd.DataFrame) -> Optional[str]:
    return _col(df, ["IsSC","is_sc","SC","sc","IsVSC","is_vsc","VSC","vsc","YellowPhase"])  # bool


def _grid_map(raw_dir: Path, y: int, r: int) -> Dict[str, float]:
    # prefer qualifying; fallback to results.grid
    q = read_csv_if_exists(raw_dir / f"qualifying_{y}_{r}.csv")
    if not q.empty:
        d = _drv_col(q) or "Driver"
        p = _col(q, ["Position","position","pos"]) or None
        if p is None:
            # fallback rank by order
            q = q.copy(); q["__rk"] = np.arange(1, len(q)+1)
            return {str(a): float(b) for a,b in zip(q[d].astype(str), q["__rk"]) }
        return {str(a): float(b) for a,b in zip(q[d].astype(str), pd.to_numeric(q[p], errors="coerce"))}
    res = read_csv_if_exists(raw_dir / f"results_{y}_{r}.csv")
    if not res.empty and ("grid" in res.columns):
        d = _drv_col(res) or "Driver"
        return {str(a): float(b) for a,b in zip(res[d].astype(str), pd.to_numeric(res["grid"], errors="coerce"))}
    return {}


def _to_seconds_any(x: pd.Series) -> pd.Series:
    s = pd.to_timedelta(x, errors="coerce")
    if s.notna().any():
        return s.dt.total_seconds()
    xn = pd.to_numeric(x, errors="coerce")
    if xn.notna().any():
        if (xn > 1000).mean() > 0.5:
            xn = xn / 1000.0
        return xn
    return pd.Series(np.nan, index=x.index)

# --------------------- per-race metrics ---------------------

def _race_lap1(raw_dir: Path, y: int, r: int) -> pd.DataFrame:
    """Return per-driver lap1_gain and lap1_incident (0/1)."""
    laps = load_laps_enriched(raw_dir, y, r)
    if laps.empty:
        return pd.DataFrame(columns=["Driver","lap1_gain","lap1_incident"]) 

    lapc = _lap_col(laps); posc = _pos_col(laps); dcol = _drv_col(laps)
    if any(v is None for v in (lapc,posc,dcol)):
        return pd.DataFrame(columns=["Driver","lap1_gain","lap1_incident"]) 

    grid = _grid_map(raw_dir, y, r)
    if not grid:
        return pd.DataFrame(columns=["Driver","lap1_gain","lap1_incident"]) 

    # pos after lap 1: last record with lap==1 per driver
    tmp = laps[pd.to_numeric(laps[lapc], errors="coerce") == 1].copy()
    if tmp.empty:
        return pd.DataFrame(columns=["Driver","lap1_gain","lap1_incident"]) 
    tmp = tmp.sort_values([dcol, lapc]).groupby(dcol).tail(1)
    tmp["pos1"] = pd.to_numeric(tmp[posc], errors="coerce")
    tmp["Driver"] = tmp[dcol].astype(str)
    tmp["grid"] = tmp["Driver"].map(grid)
    tmp = tmp.dropna(subset=["grid","pos1"]) 
    if tmp.empty:
        return pd.DataFrame(columns=["Driver","lap1_gain","lap1_incident"]) 

    tmp["lap1_gain"] = tmp["grid"] - tmp["pos1"]

    # try race_events for incidents
    inc = pd.Series(0.0, index=tmp.index)
    ev = read_csv_if_exists(raw_dir / f"race_events_{y}_{r}.csv")
    if not ev.empty:
        L = _lap_col(ev); who = _drv_col(ev); ec = _col(ev, ["Event","Type","event","type","Flag","flag","Description"]) 
        if L and who and ec:
            mask1 = pd.to_numeric(ev[L], errors="coerce") == 1
            if mask1.any():
                bad = ev.loc[mask1, ec].astype(str).str.upper().str.contains(r"INCIDENT|CRASH|COLLISION|SPIN|DNF|RETIRE|CONTACT|OFF|WALL")
                violators = ev.loc[mask1 & bad, who].astype(str).unique().tolist()
                inc = tmp["Driver"].isin(violators).astype(float)
    if inc.eq(0).all():
        # fallback: big loss or pit on lap1
        pitc = _pit_flag(laps)
        lost = (tmp["grid"] - tmp["pos1"]).fillna(0) <= -3
        if pitc is not None:
            pit1 = laps[(pd.to_numeric(laps[lapc], errors="coerce")==1) & laps[pitc].astype(bool)][dcol].astype(str).unique().tolist()
            inc = (lost | tmp["Driver"].isin(pit1)).astype(float)
        else:
            inc = lost.astype(float)

    return tmp[["Driver","lap1_gain"]].assign(lap1_incident=inc.values)


def _race_traffic_penalty(raw_dir: Path, y: int, r: int, dirty_gap_s: float = 1.5) -> pd.DataFrame:
    """Return per-driver traffic_penalty_s for a race using gap-to-ahead if available."""
    laps = load_laps_enriched(raw_dir, y, r)
    if laps.empty:
        return pd.DataFrame(columns=["Driver","traffic_penalty_s"]) 

    lapc = _lap_col(laps); dcol = _drv_col(laps)
    gapc = _gap_ahead_col(laps)
    if any(v is None for v in (lapc,dcol)) or gapc is None:
        return pd.DataFrame(columns=["Driver","traffic_penalty_s"]) 

    # choose lap time column and convert to seconds
    ltcands = _time_cols(laps)
    if not ltcands:
        return pd.DataFrame(columns=["Driver","traffic_penalty_s"]) 
    lt = ltcands[0]
    lt_s = _to_seconds_any(laps[lt])

    df = laps.copy()
    df["Driver"] = df[dcol].astype(str)
    df["__lap"] = pd.to_numeric(df[lapc], errors="coerce")
    df["gap_ahead_s"] = pd.to_numeric(laps[gapc], errors="coerce")
    df["lap_time_s"] = pd.to_numeric(lt_s, errors="coerce")

    # exclude pit/SC if available
    mask = pd.Series(True, index=df.index)
    pitc = _pit_flag(laps); scc = _sc_flag(laps)
    if pitc is not None:
        mask &= ~laps[pitc].astype(bool)
    if scc is not None:
        mask &= ~laps[scc].astype(bool)

    # reasonable lap time bounds to filter outliers
    mask &= df["lap_time_s"].between(40, 200)

    df = df[mask]
    if df.empty:
        return pd.DataFrame(columns=["Driver","traffic_penalty_s"]) 

    df["is_dirty"] = df["gap_ahead_s"] < dirty_gap_s
    # need both regimes per driver
    g = df.groupby("Driver")
    dirty = g.apply(lambda x: float(np.nanmean(x.loc[x["is_dirty"], "lap_time_s"])) if x["is_dirty"].any() else np.nan)
    clean = g.apply(lambda x: float(np.nanmean(x.loc[~x["is_dirty"], "lap_time_s"])) if (~x["is_dirty"]).any() else np.nan)
    pen = (dirty - clean).rename("traffic_penalty_s")
    out = pen.reset_index()
    return out


def _race_net_pass_index(raw_dir: Path, y: int, r: int) -> pd.DataFrame:
    """Return per-driver net_pass (normalized later per 100 green laps)."""
    laps = load_laps_enriched(raw_dir, y, r)
    if laps.empty:
        return pd.DataFrame(columns=["Driver","net_pass","green_laps"]) 

    lapc = _lap_col(laps); posc = _pos_col(laps); dcol = _drv_col(laps)
    if any(v is None for v in (lapc,posc,dcol)):
        return pd.DataFrame(columns=["Driver","net_pass","green_laps"]) 

    df = laps.copy().sort_values([dcol, lapc])
    df["Driver"] = df[dcol].astype(str)
    df["__lap"] = pd.to_numeric(df[lapc], errors="coerce")
    df["__pos"] = pd.to_numeric(df[posc], errors="coerce")

    # exclude pit and SC/VSC laps if columns exist
    mask = pd.Series(True, index=df.index)
    pitc = _pit_flag(laps); scc = _sc_flag(laps)
    if pitc is not None:
        mask &= ~laps[pitc].astype(bool)
    if scc is not None:
        mask &= ~laps[scc].astype(bool)

    df = df[mask]
    if df.empty:
        return pd.DataFrame(columns=["Driver","net_pass","green_laps"]) 

    def _per_driver(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("__lap").copy()
        d = g["__pos"].diff()  # +1 = lost a place; -1 = gained
        gains = np.maximum(-d, 0)
        losses = np.maximum(d, 0)
        net = float(np.nansum(gains) - np.nansum(losses))
        green = float((g["__lap"].diff().fillna(0) > 0).sum())
        return pd.Series({"net_pass": net, "green_laps": green})

    # pandas 2.2+: исключаем группирующие колонки из apply
    agg = (df.groupby("Driver", dropna=False)
            .apply(_per_driver, include_groups=False)
            .reset_index())

    return agg

def _safe_roster(raw_dir: Path, year: int, rnd: int, explicit):
    if explicit:
        return pd.Series(explicit, name="Driver", dtype=str)
    # 1) entrylist по приоритету
    for name in (f"entrylist_{year}_{rnd}_Q.csv",
                 f"entrylist_{year}_{rnd}.csv",
                 f"results_{year}_{rnd}_Q.csv"):  # допускаем только Q
        df = read_csv_if_exists(raw_dir / name)
        if not df.empty:
            col = _drv_col(df)
            if col:
                return df[col].astype(str).drop_duplicates().rename("Driver")
    # 2) резерв — объединить всех пилотов из последних K прошедших гонок
    pool = []
    for (y, r) in _list_prev_races(raw_dir, year, rnd, k=10):
        res = read_csv_if_exists(raw_dir / f"results_{y}_{r}.csv")
        col = _drv_col(res)
        if not res.empty and col:
            pool.append(res[col].astype(str))
    return (pd.concat(pool).drop_duplicates().rename("Driver")
            if pool else pd.Series([], name="Driver", dtype=str))


# ------------------------- main API -------------------------

def featurize(ctx: Dict, lookback: int = 10) -> pd.DataFrame:
    """Compute pre‑race traffic/start/overtake features from last K past races.

    ctx: {'raw_dir': str|Path, 'year': int, 'round': int, 'drivers'?: list[str]}
    """
    raw_dir = Path(ctx["raw_dir"]) if not isinstance(ctx.get("raw_dir"), Path) else ctx["raw_dir"]
    year = int(ctx["year"]); rnd = int(ctx["round"])  

    # Resolve driver index to return
    drivers = _safe_roster(raw_dir, year, rnd, ctx.get("drivers"))
    if drivers.empty:
        return pd.DataFrame()

    prev = _list_prev_races(raw_dir, year, rnd, lookback)
    if not prev:
        return ensure_driver_index(drivers, {
            "lap1_gain_avg_prev10": np.nan,
            "lap1_incident_rate_prev10": np.nan,
            "traffic_penalty_s_prev10": np.nan,
            "net_pass_index_prev10": np.nan,
        })

    L1, TR, NP = [], [], []
    for (y, r) in prev:
        a = _race_lap1(raw_dir, y, r)
        if not a.empty:
            L1.append(a)
        b = _race_traffic_penalty(raw_dir, y, r)
        if not b.empty:
            TR.append(b)
        c = _race_net_pass_index(raw_dir, y, r)
        if not c.empty:
            NP.append(c)

    def _avg(df_list: List[pd.DataFrame], key_in: str, key_out: str) -> pd.DataFrame:
        if not df_list:
            return pd.DataFrame({"Driver": drivers, key_out: np.nan})
        df = pd.concat(df_list, ignore_index=True)
        s = df.groupby("Driver")[key_in].mean()
        return pd.DataFrame({"Driver": s.index.astype(str), key_out: s.values})

    # averages across races
    l1_gain = _avg(L1, "lap1_gain", "lap1_gain_avg_prev10")
    l1_inc  = _avg(L1, "lap1_incident", "lap1_incident_rate_prev10")

    traf_p = _avg(TR, "traffic_penalty_s", "traffic_penalty_s_prev10")

    if NP:
        df_np = pd.concat(NP, ignore_index=True)
        df_np["net_per100"] = (df_np["net_pass"] / df_np["green_laps"]).replace([np.inf, -np.inf], np.nan) * 100.0
        net_pi = df_np.groupby("Driver")["net_per100"].mean().reset_index().rename(columns={"net_per100":"net_pass_index_prev10"})
    else:
        net_pi = pd.DataFrame({"Driver": drivers, "net_pass_index_prev10": np.nan})

    # merge & reindex to requested drivers order
    out = (
        pd.DataFrame({"Driver": drivers})
        .merge(l1_gain, on="Driver", how="left")
        .merge(l1_inc, on="Driver", how="left")
        .merge(traf_p, on="Driver", how="left")
        .merge(net_pi, on="Driver", how="left")
    )

    # ensure numeric types
    for c in ["lap1_gain_avg_prev10","lap1_incident_rate_prev10","traffic_penalty_s_prev10","net_pass_index_prev10"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out
