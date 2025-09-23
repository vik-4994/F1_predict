#!/usr/bin/env python3
"""
Pit‑crew & operational risks (PRE; FastF1‑aligned) — v2

Key change vs v1: **никогда не отдаём empty**, даже если нет истории или
нет маппинга команд — вернём каркас по текущему ростеру с NaN‑ами.

Works with datasets exported by export_last_two_years.py:
  • races.csv                              – schedule snapshot (year, round)
  • pit_stops_{Y}_{R}.csv                  – derived from laps (duration_ms, lap)
  • results_{Y}_{R}.csv / entrylist_{Y}_{R}.csv – to map Driver→TeamName for that race

Outputs (one row per Driver, broadcasted from their current Team):
  - pitcrew_iqr_team        : IQR of the team's pit‑stop durations (ms) over prev ≤ N races
  - slowstop_risk_team      : P(duration_ms > field_p90) over prev ≤ N races
  - slowstop_risk_cond_SC   : slowstop_risk_team adjusted by track SC risk & undercut window width

Optional dependencies:
  - strategy_priors_pre.featurize(ctx) → column 'undercut_window_width' (scalar)
  - track_onehot.featurize(ctx)        → column 'track_sc_prob' (scalar)

Tuning via ctx['pit_ops_risk_weights']:
{
  'trim_q_low': 0.01,
  'trim_q_high': 0.99,
  'ms_min': 1500,
  'ms_max': 25000,
  'prevN': 10,
  'sc_weight': 0.60,
  'window_weight': 0.25,
  'sc_baseline': 0.25
}
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import re

try:
    from .utils import read_csv_if_exists
    from .track_onehot import featurize as feat_track
except Exception:  # pragma: no cover
    def read_csv_if_exists(p: Path) -> pd.DataFrame:  # type: ignore
        return pd.read_csv(p) if Path(p).exists() else pd.DataFrame()
    def feat_track(ctx: dict) -> pd.DataFrame:  # type: ignore
        return pd.DataFrame()

# optional, guarded in code
try:
    from .strategy_priors_pre import featurize as feat_strategy
except Exception:  # pragma: no cover
    feat_strategy = None  # type: ignore

__all__ = ["featurize"]

# -----------------------------
# Helpers
# -----------------------------

def _asof_mask(races: pd.DataFrame, year: int, rnd: int) -> pd.Series:
    y = pd.to_numeric(races.get("year"), errors="coerce")
    r = pd.to_numeric(races.get("round"), errors="coerce")
    return (y < int(year)) | ((y == int(year)) & (r < int(rnd)))


def _list_prev_races(races: pd.DataFrame, year: int, rnd: int, prevN: int) -> List[Tuple[int, int]]:
    if races is None or races.empty:
        return []
    past = races.loc[_asof_mask(races, year, rnd), ["year", "round"]].dropna()
    if past.empty:
        return []
    past = past.astype(int).sort_values(["year", "round"]).tail(int(prevN))
    return list(map(tuple, past.values.tolist()))


def _read_results_for_race(raw_dir: Path, y: int, r: int) -> pd.DataFrame:
    df = read_csv_if_exists(raw_dir / f"results_{y}_{r}.csv")
    if df.empty:
        df = read_csv_if_exists(raw_dir / f"results_{y}_{r}_R.csv")
    return df


def _read_entrylist_for_race(raw_dir: Path, y: int, r: int) -> pd.DataFrame:
    for ses in ("R", "Q"):
        df = read_csv_if_exists(raw_dir / f"entrylist_{y}_{r}_{ses}.csv")
        if not df.empty:
            return df
    return pd.DataFrame()


def _driver_team_map(raw_dir: Path, y: int, r: int) -> pd.DataFrame:
    """Return rows [Driver, Team] for that race.
    Driver is Abbreviation (preferred) or Driver code; Team is TeamName or Team.
    """
    res = _read_results_for_race(raw_dir, y, r)
    ent = _read_entrylist_for_race(raw_dir, y, r)

    # Try from results first
    if not res.empty:
        drv_col = None
        for c in ("Abbreviation", "Driver", "code", "driverRef"):
            if c in res.columns and res[c].notna().any():
                drv_col = c; break
        team_col = None
        for c in ("TeamName", "Team", "Constructor", "constructorRef"):
            if c in res.columns and res[c].notna().any():
                team_col = c; break
        if drv_col and team_col:
            df = res[[drv_col, team_col]].dropna().rename(columns={drv_col: "Driver", team_col: "Team"}).drop_duplicates()
            if not df.empty:
                return df

    # Fallback to entrylist
    if not ent.empty:
        drv_col = None
        for c in ("Abbreviation", "Driver", "code", "driverRef"):
            if c in ent.columns and ent[c].notna().any():
                drv_col = c; break
        team_col = None
        for c in ("TeamName", "Team"):
            if c in ent.columns and ent[c].notna().any():
                team_col = c; break
        if drv_col and team_col:
            df = ent[[drv_col, team_col]].dropna().rename(columns={drv_col: "Driver", team_col: "Team"}).drop_duplicates()
            return df

    return pd.DataFrame(columns=["Driver", "Team"])  # empty


def _current_drivers(raw_dir: Path, year: int, rnd: int) -> List[str]:
    # Prefer current race results/entrylist
    for src in (f"results_{year}_{rnd}.csv", f"results_{year}_{rnd}_R.csv", f"entrylist_{year}_{rnd}_R.csv", f"entrylist_{year}_{rnd}_Q.csv"):
        df = read_csv_if_exists(raw_dir / src)
        if not df.empty:
            for c in ("Abbreviation", "Driver", "code", "driverRef"):
                if c in df.columns and df[c].notna().any():
                    return df[c].astype(str).dropna().drop_duplicates().tolist()
    # Fallback to laps
    laps = read_csv_if_exists(raw_dir / f"laps_{year}_{rnd}.csv")
    if not laps.empty:
        for c in ("Abbreviation", "Driver"):
            if c in laps.columns:
                return laps[c].astype(str).dropna().drop_duplicates().tolist()
    return []


def _load_pitstops_for_race(raw_dir: Path, y: int, r: int) -> pd.DataFrame:
    """Load pit stops derived by the exporter. Columns: duration_ms, Driver/DriverNumber if available."""
    # Preferred filename per exporter
    for name in (f"pit_stops_{y}_{r}.csv", f"pitstops_{y}_{r}.csv", f"pitStops_{y}_{r}.csv"):
        df = read_csv_if_exists(raw_dir / name)
        if not df.empty:
            break
    else:
        df = pd.DataFrame()
    if df.empty:
        return df

    # Normalize duration column
    dur = None
    for c in ("duration_ms", "milliseconds", "pit_ms"):
        if c in df.columns:
            dur = pd.to_numeric(df[c], errors="coerce"); break
    if dur is None and "duration" in df.columns:
        # duration in seconds (string or float)
        sec = pd.to_numeric(df["duration"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        dur = sec * 1000.0
    if dur is None:
        return pd.DataFrame()

    out = pd.DataFrame({"duration_ms": dur.astype(float)})
    # attach driver if present (Abbreviation preferred)
    for c in ("Abbreviation", "Driver", "code", "driverRef"):
        if c in df.columns:
            out["Driver"] = df[c].astype(str)
            break
    # attach DriverNumber if present (can help with merges if Driver absent)
    if "DriverNumber" in df.columns and "Driver" not in out.columns:
        out["DriverNumber"] = pd.to_numeric(df["DriverNumber"], errors="coerce").astype("Int64")
    return out.dropna(subset=["duration_ms"]).reset_index(drop=True)


# -----------------------------
# Main featurizer
# -----------------------------

def featurize(ctx: dict) -> pd.DataFrame:
    raw_dir = Path(ctx.get("raw_dir", "data/raw_csv"))
    year = int(ctx["year"])  # required
    rnd = int(ctx["round"])  # required

    races = read_csv_if_exists(raw_dir / "races.csv")

    # Weights & trims
    W: Dict[str, float] = {
        'trim_q_low': 0.01,
        'trim_q_high': 0.99,
        'ms_min': 1500,
        'ms_max': 25000,
        'prevN': 10,
        'sc_weight': 0.60,
        'window_weight': 0.25,
        'sc_baseline': 0.25,
    }
    W.update(ctx.get('pit_ops_risk_weights', {}))

    prev = _list_prev_races(races, year, rnd, int(W['prevN'])) if not races.empty else []

    # Collect pit samples for previous races with team labels
    samples: List[pd.DataFrame] = []
    for (y, r) in prev:
        ps = _load_pitstops_for_race(raw_dir, y, r)
        if ps.empty:
            continue
        # map Driver -> Team for that race
        d2t = _driver_team_map(raw_dir, y, r)
        if d2t.empty:
            continue
        # prefer join by Driver (Abbreviation), else by DriverNumber if available
        if 'Driver' in ps.columns and 'Driver' in d2t.columns:
            df = ps.merge(d2t, on='Driver', how='left')
        elif 'DriverNumber' in ps.columns and 'Driver' not in ps.columns and 'DriverNumber' in d2t.columns:
            df = ps.merge(d2t, on='DriverNumber', how='left')
        else:
            # cannot map to team; skip this race's data
            continue
        df = df.dropna(subset=['duration_ms', 'Team'])
        if df.empty:
            continue
        df['year'] = int(y); df['round'] = int(r)
        samples.append(df[['year','round','Team','duration_ms']])

    agg = pd.DataFrame(columns=['Team','p25','p75','pitcrew_iqr_team','slowstop_risk_team'])
    if samples:
        pit = pd.concat(samples, ignore_index=True)
        if not pit.empty:
            # Trim and sanity bounds on durations
            qlo = float(pit['duration_ms'].quantile(float(W['trim_q_low'])))
            qhi = float(pit['duration_ms'].quantile(float(W['trim_q_high'])))
            low = max(float(W['ms_min']), qlo)
            high = min(float(W['ms_max']), qhi)
            pit = pit[(pit['duration_ms'] >= low) & (pit['duration_ms'] <= high)]

            # Field p90 across all teams
            field_p90 = float(pit['duration_ms'].quantile(0.90)) if not pit.empty else np.nan

            # Team aggregates
            if not pit.empty:
                agg = (
                    pit.groupby('Team')['duration_ms']
                       .agg(p25=lambda s: float(s.quantile(0.25)),
                            p75=lambda s: float(s.quantile(0.75)),
                            slowstop_risk_team=lambda s: float(np.mean(s > field_p90) if np.isfinite(field_p90) else np.nan))
                       .reset_index()
                )
                agg['pitcrew_iqr_team'] = agg['p75'] - agg['p25']

    # SC probability from track_onehot
    tr = pd.DataFrame()
    try:
        tr = feat_track(ctx)
    except Exception:
        tr = pd.DataFrame()
    if tr is None or tr.empty or 'track_sc_prob' not in tr.columns:
        sc_prob = float(W['sc_baseline'])
    else:
        try:
            sc_prob = float(pd.to_numeric(tr['track_sc_prob'], errors='coerce').dropna().iloc[0])
        except Exception:
            sc_prob = float(W['sc_baseline'])
    sc_prob = float(np.clip(sc_prob, 0.0, 1.0))

    # Optional undercut window width
    uw_norm = 0.0
    if feat_strategy is not None:
        try:
            sp = feat_strategy(ctx)
            if sp is not None and not sp.empty and 'undercut_window_width' in sp.columns:
                uw = float(pd.to_numeric(sp['undercut_window_width'], errors='coerce').median())
                uw_norm = float(np.clip(uw / 5.0, 0.0, 1.0))
        except Exception:
            uw_norm = 0.0

    sc_uplift = 1.0 + float(W['sc_weight']) * (sc_prob - float(W['sc_baseline']))
    win_uplift = 1.0 + float(W['window_weight']) * uw_norm

    if not agg.empty and 'slowstop_risk_team' in agg.columns:
        agg['slowstop_risk_cond_SC'] = (agg['slowstop_risk_team'] * sc_uplift * win_uplift).clip(0.0, 1.0)
    else:
        agg['slowstop_risk_cond_SC'] = np.nan

    # Broadcast by current lineup (Driver → Team mapping for current race)
    d2t_now = _driver_team_map(raw_dir, year, rnd)
    roster = _current_drivers(raw_dir, year, rnd)

    # Build output skeleton first (to avoid empty)
    out = pd.DataFrame({'Driver': roster}) if roster else pd.DataFrame(columns=['Driver'])

    if not d2t_now.empty and not agg.empty:
        out = d2t_now.merge(agg[['Team','pitcrew_iqr_team','slowstop_risk_team','slowstop_risk_cond_SC']], on='Team', how='left')[['Driver','pitcrew_iqr_team','slowstop_risk_team','slowstop_risk_cond_SC']]
    else:
        for c in ('pitcrew_iqr_team','slowstop_risk_team','slowstop_risk_cond_SC'):
            out[c] = np.nan

    # ensure final columns
    keep = ['Driver', 'pitcrew_iqr_team', 'slowstop_risk_team', 'slowstop_risk_cond_SC']
    for c in keep:
        if c not in out.columns:
            out[c] = np.nan

    # if even roster is missing, return truly empty (nothing to broadcast onto)
    if out.empty:
        return pd.DataFrame(columns=keep)

    return out[keep]


if __name__ == '__main__':
    ctx = {"raw_dir": "data/raw_csv", "year": 2024, "round": 2}
    print(featurize(ctx).head())
