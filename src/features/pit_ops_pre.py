#!/usr/bin/env python3
"""
Pit operations priors (PRE; FastF1‑aligned, no leakage).

Replaces Ergast‑style dependencies (driverId/constructorId, pitStops.csv, lap_times.csv)
with FastF1 exporter layout used in this repo:
  • races.csv                                  – schedule snapshot (year, round)
  • results_{Y}_{R}.csv / entrylist_{Y}_{R}.csv – map Driver(=Abbreviation) → TeamName (per race)
  • pit_stops_{Y}_{R}.csv                       – per stop with 'duration_ms' (optional on some events)
  • laps_{Y}_{R}.csv                            – per lap with 'milliseconds' or 'LapTime'

Output (one row per current Driver):
  - pitcrew_time_team_p50_s   : team's median pit time (s) over history window
  - pitcrew_time_team_p90_s   : team's 90th percentile pit time (s)
  - slowstop_risk_team        : P(stop > field p90) over window
  - undercut_gain_hist_s      : median undercut gain (s) from same‑track history (fallback: recent window)
  - overcut_gain_hist_s       : median overcut gain (s) from same‑track history (fallback: recent window)

If history files are missing, returns a non‑empty skeleton for current drivers with NaNs in feature columns.
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

def _asof_mask(races: pd.DataFrame, year: int, rnd: int) -> pd.Series:
    y = pd.to_numeric(races.get('year'), errors='coerce')
    r = pd.to_numeric(races.get('round'), errors='coerce')
    return (y < int(year)) | ((y == int(year)) & (r < int(rnd)))


def _list_prev_races(races: pd.DataFrame, year: int, rnd: int, prevN: int) -> List[Tuple[int,int]]:
    if races is None or races.empty:
        return []
    past = races.loc[_asof_mask(races, year, rnd), ['year','round']].dropna()
    if past.empty:
        return []
    past = past.astype(int).sort_values(['year','round']).tail(int(prevN))
    return list(map(tuple, past.values.tolist()))


def _slugify(s: str) -> str:
    s = (s or '').strip().lower()
    s = re.sub(r'[^\w]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s or 'unknown'


def _event_slug(raw_dir: Path, y: int, r: int) -> str:
    meta = read_csv_if_exists(raw_dir / f'meta_{y}_{r}.csv')
    if not meta.empty:
        for c in ('EventName','OfficialEventName','Location','CircuitName','Name'):
            if c in meta.columns and meta[c].notna().any():
                return _slugify(str(meta[c].dropna().iloc[0]))
    sc = read_csv_if_exists(raw_dir / f'schedule_{y}.csv')
    if not sc.empty:
        rr = sc.loc[pd.to_numeric(sc.get('RoundNumber') or sc.get('round'), errors='coerce') == int(r)]
        if not rr.empty:
            for c in ('EventName','OfficialEventName','Location'):
                if c in rr.columns and rr[c].notna().any():
                    return _slugify(str(rr[c].dropna().iloc[0]))
    return f'{y}_{r}'


def _ensure_driver(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if 'Driver' in df.columns:
        return df
    for c in ('Abbreviation','DriverCode','code','driverRef','DriverId','driverId'):
        if c in df.columns:
            return df.rename(columns={c:'Driver'})
    return df


def _load_pit_stops(raw_dir: Path, y: int, r: int) -> pd.DataFrame:
    for name in (f'pit_stops_{y}_{r}.csv', f'pitstops_{y}_{r}.csv', f'pitStops_{y}_{r}.csv'):
        df = read_csv_if_exists(raw_dir / name)
        if not df.empty:
            break
    else:
        df = pd.DataFrame()
    if df.empty:
        return df
    df = _ensure_driver(df)
    if 'lap' not in df.columns and 'LapNumber' in df.columns:
        df = df.rename(columns={'LapNumber':'lap'})
    if 'duration_ms' not in df.columns and 'milliseconds' in df.columns:
        df = df.rename(columns={'milliseconds':'duration_ms'})
    df['lap'] = pd.to_numeric(df.get('lap'), errors='coerce').astype('Int64')
    df['duration_ms'] = pd.to_numeric(df.get('duration_ms'), errors='coerce')
    return df


def _load_laps(raw_dir: Path, y: int, r: int) -> pd.DataFrame:
    df = read_csv_if_exists(raw_dir / f'laps_{y}_{r}.csv')
    if df.empty:
        return df
    df = _ensure_driver(df)
    if 'LapNumber' in df.columns and 'lap' not in df.columns:
        df = df.rename(columns={'LapNumber':'lap'})
    if 'milliseconds' not in df.columns:
        if 'LapTime' in df.columns:
            df['milliseconds'] = pd.to_timedelta(df['LapTime'], errors='coerce').dt.total_seconds() * 1000.0
        elif 'LapTimeSec' in df.columns:
            df['milliseconds'] = pd.to_numeric(df['LapTimeSec'], errors='coerce') * 1000.0
        else:
            df['milliseconds'] = np.nan
    df['lap'] = pd.to_numeric(df.get('lap'), errors='coerce').astype('Int64')
    return df


def _read_session_results(raw_dir: Path, y: int, r: int) -> pd.DataFrame:
    df = read_csv_if_exists(raw_dir / f'results_{y}_{r}.csv')
    if df.empty:
        df = read_csv_if_exists(raw_dir / f'results_{y}_{r}_R.csv')
    return df


def _read_entrylist(raw_dir: Path, y: int, r: int) -> pd.DataFrame:
    for ses in ('R','Q'):
        df = read_csv_if_exists(raw_dir / f'entrylist_{y}_{r}_{ses}.csv')
        if not df.empty:
            return df
    return pd.DataFrame()


def _driver_team_map(raw_dir: Path, y: int, r: int) -> pd.DataFrame:
    res = _read_session_results(raw_dir, y, r)
    ent = _read_entrylist(raw_dir, y, r)
    if not res.empty:
        dcol = next((c for c in ('Abbreviation','Driver','code','driverRef') if c in res.columns), None)
        tcol = next((c for c in ('TeamName','Team','Constructor','constructorRef') if c in res.columns), None)
        if dcol and tcol:
            df = res[[dcol, tcol]].dropna().rename(columns={dcol:'Driver', tcol:'Team'})
            if not df.empty:
                return df.drop_duplicates()
    if not ent.empty:
        dcol = next((c for c in ('Abbreviation','Driver','code','driverRef') if c in ent.columns), None)
        tcol = next((c for c in ('TeamName','Team') if c in ent.columns), None)
        if dcol and tcol:
            return ent[[dcol, tcol]].dropna().rename(columns={dcol:'Driver', tcol:'Team'}).drop_duplicates()
    return pd.DataFrame(columns=['Driver','Team'])


def _current_drivers(raw_dir: Path, year: int, rnd: int) -> List[str]:
    for src in (f'results_{year}_{rnd}.csv', f'results_{year}_{rnd}_R.csv', f'entrylist_{year}_{rnd}_R.csv', f'entrylist_{year}_{rnd}_Q.csv'):
        df = read_csv_if_exists(raw_dir / src)
        if not df.empty:
            for c in ('Abbreviation','Driver','code','driverRef'):
                if c in df.columns and df[c].notna().any():
                    return df[c].astype(str).dropna().drop_duplicates().tolist()
    laps = read_csv_if_exists(raw_dir / f'laps_{year}_{rnd}.csv')
    if not laps.empty:
        for c in ('Abbreviation','Driver'):
            if c in laps.columns:
                return laps[c].astype(str).dropna().drop_duplicates().tolist()
    return []

# ----------------------------- core calcs -----------------------------

def _team_pitcrew_priors(raw_dir: Path, races: pd.DataFrame, year: int, rnd: int, prevN: int = 10) -> pd.DataFrame:
    prev = _list_prev_races(races, year, rnd, prevN)
    if not prev:
        return pd.DataFrame(columns=['Team','pitcrew_time_team_p50_s','pitcrew_time_team_p90_s','slowstop_risk_team'])

    samples: List[pd.DataFrame] = []
    for (y, r) in prev:
        ps = _load_pit_stops(raw_dir, y, r)
        if ps.empty:
            continue
        d2t = _driver_team_map(raw_dir, y, r)
        if d2t.empty:
            continue
        df = ps.merge(d2t, on='Driver', how='left').dropna(subset=['Team','duration_ms'])
        if df.empty:
            continue
        df['duration_ms'] = pd.to_numeric(df['duration_ms'], errors='coerce')
        df = df[(df['duration_ms'] >= 1500) & (df['duration_ms'] <= 25000)]
        if df.empty:
            continue
        df['year'] = int(y); df['round'] = int(r)
        samples.append(df[['year','round','Team','duration_ms']])

    if not samples:
        return pd.DataFrame(columns=['Team','pitcrew_time_team_p50_s','pitcrew_time_team_p90_s','slowstop_risk_team'])

    pit = pd.concat(samples, ignore_index=True)
    # trim outliers
    qlo, qhi = pit['duration_ms'].quantile([0.01, 0.99])
    pit = pit[(pit['duration_ms'] >= qlo) & (pit['duration_ms'] <= qhi)]

    field_p90 = float(pit['duration_ms'].quantile(0.90)) if not pit.empty else np.nan

    agg = (
        pit.groupby('Team')['duration_ms']
           .agg(p50=lambda s: float(np.median(s)) / 1000.0,
                p90=lambda s: float(np.quantile(s, 0.90)) / 1000.0)
           .reset_index()
           .rename(columns={'p50':'pitcrew_time_team_p50_s','p90':'pitcrew_time_team_p90_s'})
    )
    if np.isfinite(field_p90):
        risk = (pit.assign(is_slow=pit['duration_ms'] > field_p90)
                    .groupby('Team')['is_slow'].mean()
                    .rename('slowstop_risk_team').reset_index())
    else:
        risk = agg[['Team']].assign(slowstop_risk_team=np.nan)

    return agg.merge(risk, on='Team', how='left')


def _undercut_overcut_one_race(raw_dir: Path, y: int, r: int) -> Optional[Tuple[float,float]]:
    laps = _load_laps(raw_dir, y, r)
    ps   = _load_pit_stops(raw_dir, y, r)
    if laps.empty or ps.empty:
        return None

    lt = laps[['Driver','lap','milliseconds']].dropna(subset=['lap','milliseconds']).copy()
    lt['milliseconds'] = pd.to_numeric(lt['milliseconds'], errors='coerce')

    # mark pitted laps
    pp = ps[['Driver','lap']].dropna().drop_duplicates().copy()
    pp['pitted'] = True
    lt = lt.merge(pp, on=['Driver','lap'], how='left')
    lt['pitted'] = lt['pitted'].fillna(False)

    # median non‑pit reference per lap
    ref = (lt.loc[~lt['pitted']].groupby('lap', as_index=False)['milliseconds'].median().rename(columns={'milliseconds':'ref_ms'}))

    inlap  = pp.merge(lt, on=['Driver','lap'], how='left').rename(columns={'milliseconds':'in_ms'})
    inlap  = inlap.merge(ref, on='lap', how='left').rename(columns={'ref_ms':'in_ref_ms'})

    outlap = pp[['Driver','lap']].copy(); outlap['lap'] = outlap['lap'] + 1
    outlap = outlap.merge(lt, on=['Driver','lap'], how='left').rename(columns={'milliseconds':'out_ms'})
    outlap = outlap.merge(ref, on='lap', how='left').rename(columns={'ref_ms':'out_ref_ms'})

    io = inlap.merge(outlap, on=['Driver','lap'], how='left')
    io['in_pen_ms']  = pd.to_numeric(io['in_ms'], errors='coerce')  - pd.to_numeric(io['in_ref_ms'], errors='coerce')
    io['out_pen_ms'] = pd.to_numeric(io['out_ms'], errors='coerce') - pd.to_numeric(io['out_ref_ms'], errors='coerce')

    io = io.dropna(subset=['in_pen_ms','out_pen_ms'])
    io = io[(io['in_pen_ms'].abs() <= 60000) & (io['out_pen_ms'].abs() <= 60000)]
    if io.empty:
        return None

    race_med = io.assign(
        undercut_gain_ms=lambda d: d['in_pen_ms'] - d['out_pen_ms'],
        overcut_gain_ms=lambda d: d['out_pen_ms'],
    ).agg({'undercut_gain_ms':'median','overcut_gain_ms':'median'})

    return float(race_med['undercut_gain_ms'])/1000.0, float(race_med['overcut_gain_ms'])/1000.0


def _undercut_overcut_priors(raw_dir: Path, races: pd.DataFrame, year: int, rnd: int, prevN: int = 10) -> Tuple[float,float]:
    prev = _list_prev_races(races, year, rnd, prevN)
    if not prev:
        return (np.nan, np.nan)
    cur_slug = _event_slug(raw_dir, year, rnd)
    same: List[Tuple[int,int]] = []
    for (y,r) in prev:
        if _event_slug(raw_dir, y, r) == cur_slug:
            same.append((y,r))
    use = same if len(same) >= 2 else prev  # if too few same‑track, fall back to general window

    vals_u: List[float] = []
    vals_o: List[float] = []
    for (y,r) in use:
        res = _undercut_overcut_one_race(raw_dir, y, r)
        if res is None:
            continue
        u, o = res
        if np.isfinite(u):
            vals_u.append(u)
        if np.isfinite(o):
            vals_o.append(o)
    u_med = float(np.median(vals_u)) if vals_u else np.nan
    o_med = float(np.median(vals_o)) if vals_o else np.nan
    return (u_med, o_med)

# ----------------------------- main -----------------------------

def featurize(ctx: Dict) -> pd.DataFrame:
    """Build pit‑ops priors for the current event. Returns DF keyed by 'Driver'."""
    raw_dir = Path(ctx.get('raw_dir', 'data/raw_csv'))
    year = int(ctx['year']); rnd = int(ctx['round'])

    races = read_csv_if_exists(raw_dir / 'races.csv')
    drivers = _current_drivers(raw_dir, year, rnd)
    if not drivers:
        return pd.DataFrame()  # nothing to broadcast onto

    # Team priors (pitcrew time & slowstop risk)
    team_pr = _team_pitcrew_priors(raw_dir, races, year, rnd, prevN=int(ctx.get('pit_prevN', 10))) if not races.empty else pd.DataFrame()

    # Undercut/overcut priors (same‑track if possible)
    u_med, o_med = _undercut_overcut_priors(raw_dir, races, year, rnd, prevN=int(ctx.get('pit_prevN', 10))) if not races.empty else (np.nan, np.nan)

    # Current mapping Driver→Team to broadcast team priors
    d2t = _driver_team_map(raw_dir, year, rnd)
    out = pd.DataFrame({'Driver': drivers})
    if not d2t.empty and not team_pr.empty:
        out = out.merge(d2t, on='Driver', how='left').merge(team_pr, on='Team', how='left')
    else:
        out['pitcrew_time_team_p50_s'] = np.nan
        out['pitcrew_time_team_p90_s'] = np.nan
        out['slowstop_risk_team'] = np.nan

    out['undercut_gain_hist_s'] = u_med
    out['overcut_gain_hist_s']  = o_med

    keep = ['Driver','pitcrew_time_team_p50_s','pitcrew_time_team_p90_s','slowstop_risk_team','undercut_gain_hist_s','overcut_gain_hist_s']
    for c in keep:
        if c not in out.columns:
            out[c] = np.nan
    return out[keep]


if __name__ == '__main__':
    ctx = {"raw_dir": "data/raw_csv", "year": 2024, "round": 3}
    print(featurize(ctx).head())
