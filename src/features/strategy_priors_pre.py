#!/usr/bin/env python3
"""
Strategy priors (PRE; FastF1‑aligned, no leakage) — v2

Key changes vs previous patch:
  • Never returns empty: even if нет истории (1‑й этап) или нет pit/stints — отдаём каркас по текущему составу пилотов.
  • Если не удаётся собрать Team‑маппинг для текущей гонки, всё равно отдаём фичи по Driver (double‑stack‑колонки = NaN).
  • Больше источников для текущего состава: results/entrylist/track/telemetry/laps.

Exports per current Driver:
  Driver,
  expected_stop_count,
  first_stint_len_exp,
  undercut_window_width,
  double_stack_risk,
  double_stack_same_lap
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import re

try:
    from .utils import read_csv_if_exists
    from .track_onehot import featurize as feat_track
    from .weather_basic import featurize as feat_weather
except Exception:  # pragma: no cover
    def read_csv_if_exists(p: Path) -> pd.DataFrame:  # type: ignore
        return pd.read_csv(p) if Path(p).exists() else pd.DataFrame()
    def feat_track(ctx: dict) -> pd.DataFrame:  # type: ignore
        return pd.DataFrame()
    def feat_weather(ctx: dict) -> pd.DataFrame:  # type: ignore
        return pd.DataFrame()

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


def _safe_scalar(obj, col: Optional[str], default: float) -> float:
    try:
        if isinstance(obj, pd.DataFrame):
            s = obj[col] if col is not None and col in obj.columns else None
        elif isinstance(obj, pd.Series):
            s = obj
        else:
            s = None
        v = float(pd.to_numeric(s, errors='coerce').dropna().iloc[0]) if s is not None else float(default)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


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


def _load_laps(raw_dir: Path, y: int, r: int) -> pd.DataFrame:
    df = read_csv_if_exists(raw_dir / f'laps_{y}_{r}.csv')
    if df.empty:
        return df
    if 'LapNumber' in df.columns and 'lap' not in df.columns:
        df = df.rename(columns={'LapNumber':'lap'})
    for dcol in ('Driver','Abbreviation','code','driverRef'):
        if dcol in df.columns:
            df = df.rename(columns={dcol:'Driver'})
            break
    if 'milliseconds' not in df.columns and 'LapTime' in df.columns:
        df['milliseconds'] = pd.to_timedelta(df['LapTime'], errors='coerce').dt.total_seconds() * 1000.0
    df['lap'] = pd.to_numeric(df.get('lap'), errors='coerce').astype('Int64')
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
    for dcol in ('Driver','Abbreviation','code','driverRef'):
        if dcol in df.columns:
            df = df.rename(columns={dcol:'Driver'})
            break
    if 'lap' not in df.columns and 'LapNumber' in df.columns:
        df = df.rename(columns={'LapNumber':'lap'})
    if 'duration_ms' not in df.columns and 'milliseconds' in df.columns:
        df = df.rename(columns={'milliseconds':'duration_ms'})
    df['lap'] = pd.to_numeric(df.get('lap'), errors='coerce').astype('Int64')
    df['duration_ms'] = pd.to_numeric(df.get('duration_ms'), errors='coerce')
    return df


def _load_stints(raw_dir: Path, y: int, r: int) -> pd.DataFrame:
    df = read_csv_if_exists(raw_dir / f'stints_{y}_{r}.csv')
    if df.empty:
        return df
    for dcol in ('Driver','Abbreviation','code','driverRef'):
        if dcol in df.columns:
            df = df.rename(columns={dcol:'Driver'})
            break
    return df

# ----------------------------- core calcs -----------------------------

def _first_stop_by_driver(pit: pd.DataFrame, min_first_lap: int) -> pd.DataFrame:
    if pit is None or pit.empty or 'Driver' not in pit.columns:
        return pd.DataFrame(columns=['Driver','first_stop_lap'])
    pp = pit.dropna(subset=['lap']).copy()
    pp['lap'] = pd.to_numeric(pp['lap'], errors='coerce').astype('Int64')
    pp = pp.dropna(subset=['lap'])
    if pp.empty:
        return pd.DataFrame(columns=['Driver','first_stop_lap'])
    first = pp.groupby('Driver', as_index=False)['lap'].min().rename(columns={'lap':'first_stop_lap'})
    first = first.loc[first['first_stop_lap'] >= int(min_first_lap)].reset_index(drop=True)
    return first


def _stops_from_stints(st: pd.DataFrame) -> pd.DataFrame:
    if st is None or st.empty or 'Driver' not in st.columns or 'Stint' not in st.columns:
        return pd.DataFrame(columns=['Driver','stops'])
    mx = st.groupby('Driver', as_index=False)['Stint'].max().rename(columns={'Stint':'stops'})
    mx['stops'] = mx['stops'].astype(float) - 1.0
    return mx


def _stops_from_pits(pit: pd.DataFrame) -> pd.DataFrame:
    if pit is None or pit.empty:
        return pd.DataFrame(columns=['Driver','stops'])
    cc = pit.dropna(subset=['Driver']).groupby('Driver', as_index=False)['lap'].nunique().rename(columns={'lap':'stops'})
    cc['stops'] = pd.to_numeric(cc['stops'], errors='coerce').astype(float)
    return cc


def _cluster_median_stops(raw_dir: Path, prev: List[Tuple[int,int]]) -> float:
    vals: List[float] = []
    for (y,r) in prev:
        st = _load_stints(raw_dir, y, r)
        if not st.empty:
            dd = _stops_from_stints(st)
        else:
            pit = _load_pit_stops(raw_dir, y, r)
            dd = _stops_from_pits(pit)
        if dd.empty:
            continue
        m = pd.to_numeric(dd['stops'], errors='coerce').median()
        if pd.notna(m):
            vals.append(float(m))
    return float(np.median(vals)) if vals else np.nan


def _cluster_first_stint_len(raw_dir: Path, prev: List[Tuple[int,int]]) -> float:
    vals: List[float] = []
    for (y,r) in prev:
        st = _load_stints(raw_dir, y, r)
        if st.empty or 'Stint' not in st.columns or 'laps' not in st.columns:
            continue
        s1 = st.loc[st['Stint'] == 1]
        if s1.empty:
            continue
        m = pd.to_numeric(s1['laps'], errors='coerce').median()
        if pd.notna(m):
            vals.append(float(m))
    return float(np.median(vals)) if vals else np.nan


def _per_lap_undercut_gain(raw_dir: Path, y: int, r: int) -> pd.DataFrame:
    laps = _load_laps(raw_dir, y, r)
    pit = _load_pit_stops(raw_dir, y, r)
    if laps.empty or pit.empty:
        return pd.DataFrame(columns=['lap','undercut_gain_ms'])

    lt = laps[['Driver','lap','milliseconds']].dropna(subset=['lap','milliseconds']).copy()
    lt['milliseconds'] = pd.to_numeric(lt['milliseconds'], errors='coerce')

    ps = pit[['Driver','lap']].dropna().drop_duplicates().copy()
    ps['pitted'] = True

    lt = lt.merge(ps, on=['Driver','lap'], how='left')
    lt['pitted'] = lt['pitted'].fillna(False)

    med_nonpit = (
        lt.loc[~lt['pitted']]
          .groupby('lap', as_index=False)['milliseconds'].median()
          .rename(columns={'milliseconds':'ref_ms'})
    )

    inlap = ps.merge(lt, on=['Driver','lap'], how='left').rename(columns={'milliseconds':'in_ms'})
    inlap = inlap.merge(med_nonpit, on='lap', how='left').rename(columns={'ref_ms':'in_ref_ms'})

    outlap = ps.copy(); outlap['lap'] = outlap['lap'] + 1
    outlap = outlap.merge(lt, on=['Driver','lap'], how='left').rename(columns={'milliseconds':'out_ms'})
    outlap = outlap.merge(med_nonpit, on='lap', how='left').rename(columns={'ref_ms':'out_ref_ms'})

    io = inlap.merge(outlap, on=['Driver','lap'], how='left')
    io['undercut_gain_ms'] = -(
        (pd.to_numeric(io['in_ms'], errors='coerce') - pd.to_numeric(io['in_ref_ms'], errors='coerce')).fillna(0)
        + (pd.to_numeric(io['out_ms'], errors='coerce') - pd.to_numeric(io['out_ref_ms'], errors='coerce')).fillna(0)
    )

    gain = io.groupby('lap', as_index=False)['undercut_gain_ms'].median()
    return gain.dropna(subset=['undercut_gain_ms'])


def _window_width_from_gains(gain: pd.DataFrame, thresh_ms: float) -> float:
    if gain is None or gain.empty or 'lap' not in gain.columns:
        return np.nan
    g = gain.sort_values('lap').copy()
    ok = (pd.to_numeric(g['undercut_gain_ms'], errors='coerce') > float(thresh_ms)).astype(int).to_numpy()
    if ok.size == 0:
        return np.nan
    best = cur = 0
    for v in ok:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return float(best) if best > 0 else np.nan

# ----------------------------- roster helpers -----------------------------

def _current_drivers(raw_dir: Path, year: int, rnd: int) -> List[str]:
    # results / entrylist
    for src in (f'results_{year}_{rnd}.csv', f'results_{year}_{rnd}_R.csv', f'entrylist_{year}_{rnd}_R.csv', f'entrylist_{year}_{rnd}_Q.csv'):
        df = read_csv_if_exists(raw_dir / src)
        if not df.empty:
            for c in ('Abbreviation','Driver','code','driverRef'):
                if c in df.columns and df[c].notna().any():
                    return df[c].astype(str).dropna().drop_duplicates().tolist()
    # track_onehot/telemetry outputs
    for aux in ('track_onehot.csv', 'telemetry_summary.csv'):
        df = read_csv_if_exists(raw_dir / aux)
        if not df.empty and 'Driver' in df.columns:
            return df['Driver'].astype(str).dropna().drop_duplicates().tolist()
    # laps fallback
    laps = read_csv_if_exists(raw_dir / f'laps_{year}_{rnd}.csv')
    if not laps.empty:
        for c in ('Abbreviation','Driver'):
            if c in laps.columns:
                return laps[c].astype(str).dropna().drop_duplicates().tolist()
    return []


def _double_stack_risks(raw_dir: Path, y: int, r: int, stack_window: int, min_first_stop_lap: int) -> pd.DataFrame:
    pit = _load_pit_stops(raw_dir, y, r)
    if pit.empty:
        return pd.DataFrame(columns=['Team','double_stack_risk','double_stack_same_lap'])

    d2t = _driver_team_map(raw_dir, y, r)
    if d2t.empty or 'Driver' not in pit.columns:
        return pd.DataFrame(columns=['Team','double_stack_risk','double_stack_same_lap'])

    # первый пит каждого пилота (после разумного минимума)
    first = pit.dropna(subset=['Driver','lap']).copy()
    first['lap'] = pd.to_numeric(first['lap'], errors='coerce').astype('Int64')
    first = first.dropna(subset=['lap'])
    first = first.groupby('Driver', as_index=False)['lap'].min().rename(columns={'lap':'first_stop_lap'})
    first = first[first['first_stop_lap'] >= int(min_first_stop_lap)]

    df = first.merge(d2t, on='Driver', how='left').dropna(subset=['Team'])
    if df.empty:
        return pd.DataFrame(columns=['Team','double_stack_risk','double_stack_same_lap'])

    # по командам: вероятность, что два пилота остановились в один круг (same_lap)
    # и в пределах ±stack_window кругов (stacked)
    rows = []
    for tm, g in df.groupby('Team'):
        laps = sorted(pd.to_numeric(g['first_stop_lap'], errors='coerce').dropna().astype(int).tolist())
        if len(laps) < 2:
            rows.append({'Team': tm, 'double_stack_risk': 0.0, 'double_stack_same_lap': 0.0})
            continue
        a, b = laps[0], laps[1]
        same = float(a == b)
        near = float(abs(a - b) <= int(stack_window))
        rows.append({'Team': tm, 'double_stack_risk': near, 'double_stack_same_lap': same})
    return pd.DataFrame(rows)


# ----------------------------- main -----------------------------

def featurize(ctx: dict) -> pd.DataFrame:
    raw_dir = Path(ctx.get('raw_dir', 'data/raw_csv'))
    year = int(ctx['year']); rnd = int(ctx['round'])

    races = read_csv_if_exists(raw_dir / 'races.csv')
    # Even if races.csv is missing/empty — produce defaults for current drivers
    drivers_now = _current_drivers(raw_dir, year, rnd)

    W: Dict[str, float] = {
        'prevN': 8,
        'stack_window': 1,
        'min_first_stop_lap': 5,
        'undercut_thresh_ms': 250,
        'w_temp_count': 0.15,
        'w_temp_stint': 0.10,
        'w_sc_count': 0.50,
        'w_sc_stint': 0.50,
    }
    W.update(ctx.get('strategy_weights', {}))

    prev: List[Tuple[int,int]] = _list_prev_races(races, year, rnd, int(W['prevN'])) if not races.empty else []

    # --- Baselines (use soft defaults if history is missing) ---
    cluster_stops_median = _cluster_median_stops(raw_dir, prev) if prev else np.nan
    cluster_first_med    = _cluster_first_stint_len(raw_dir, prev) if prev else np.nan

    if not np.isfinite(cluster_stops_median):
        cluster_stops_median = 2.0
    if not np.isfinite(cluster_first_med):
        cluster_first_med = 15.0

    # Undercut window width (median across prev races)
    win_vals: List[float] = []
    if prev:
        for (y, r) in prev:
            g = _per_lap_undercut_gain(raw_dir, y, r)
            w = _window_width_from_gains(g, float(W['undercut_thresh_ms']))
            if pd.notna(w):
                win_vals.append(float(w))
    undercut_window_width = float(np.median(win_vals)) if win_vals else np.nan

    # Track/weather scalars
    try:
        tr = feat_track(ctx)
    except Exception:
        tr = pd.DataFrame()
    try:
        wthr = feat_weather({**ctx, 'mode': 'forecast'})
    except Exception:
        wthr = pd.DataFrame()

    sc_prob = _safe_scalar(tr, 'track_sc_prob', 0.25)
    track_temp = _safe_scalar(wthr, 'weather_track_temp_C', 25.0)
    if not np.isfinite(track_temp) or track_temp == 0:
        track_temp = _safe_scalar(wthr, 'track_temp_C', 25.0)

    temp_delta = (float(track_temp) - 25.0) / 10.0

    expected_stop_count = cluster_stops_median + W['w_temp_count']*temp_delta + W['w_sc_count']*sc_prob
    first_stint_len_exp  = cluster_first_med  * (1.0 - W['w_temp_stint']*temp_delta) - W['w_sc_stint']*sc_prob*5.0

    expected_stop_count = float(np.clip(expected_stop_count, 1.0, 4.0))
    first_stint_len_exp = float(np.clip(first_stint_len_exp, 5.0, 30.0))

    # Team risks (double‑stack) — may be unavailable if нет pit_stops
    team_risks = pd.DataFrame(columns=['Team','double_stack_risk','double_stack_same_lap'])
    if prev:
        all_risks: List[pd.DataFrame] = []
        for (y, r) in prev:
            rr = _double_stack_risks(raw_dir, y, r, int(W['stack_window']), int(W['min_first_stop_lap']))
            if not rr.empty:
                all_risks.append(rr)
        if all_risks:
            risks = pd.concat(all_risks, ignore_index=True)
            team_risks = risks.groupby('Team', as_index=False).agg(
                double_stack_risk=('double_stack_risk','mean'),
                double_stack_same_lap=('double_stack_same_lap','mean'),
            )

    # Current team mapping (optional)
    d2t_now = _driver_team_map(raw_dir, year, rnd)

    # --- Build output even if mapping is absent ---
    if not d2t_now.empty:
        out = d2t_now.merge(team_risks, on='Team', how='left')
        drivers = out['Driver'].astype(str).dropna().drop_duplicates().tolist()
    else:
        # fallback roster only
        drivers = drivers_now
        out = pd.DataFrame({'Driver': drivers})
        # join team risks невозможно — оставим NaN
        out['double_stack_risk'] = np.nan
        out['double_stack_same_lap'] = np.nan

    out['expected_stop_count']   = expected_stop_count
    out['first_stint_len_exp']   = first_stint_len_exp
    out['undercut_window_width'] = undercut_window_width

    keep = ['Driver','expected_stop_count','first_stint_len_exp','undercut_window_width','double_stack_risk','double_stack_same_lap']
    for c in keep:
        if c not in out.columns:
            out[c] = np.nan

    # ensure only current roster rows
    if drivers_now:
        out = pd.DataFrame({'Driver': drivers_now}).merge(out[keep], on='Driver', how='left')

    return out[keep]


if __name__ == '__main__':
    ctx = {"raw_dir": "data/raw_csv", "year": 2024, "round": 2}
    print(featurize(ctx).head())
