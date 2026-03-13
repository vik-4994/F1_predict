from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .features import featurize_pre
from .frame_utils import sanitize_frame_columns

_ROSTER_FILES = (
    "entrylist_{y}_{r}_Q.csv",
    "entrylist_{y}_{r}.csv",
    "results_{y}_{r}_Q.csv",
    "results_{y}_{r}.csv",
)
_DRIVER_COLS = ("Abbreviation", "Driver", "code", "driverRef", "BroadcastName")
_TRACK_COLS = ("EventName", "OfficialEventName", "Location", "CircuitName", "Name")
FUTURE_SAFE_MODULES = [
    "track_onehot",
    "track_profile",
    "driver_track_cluster_pre",
    "weather_basic",
    "event_chaos_priors_pre",
    "history_form",
    "telemetry_history_pre",
    "quali_priors_pre",
    "tyre_priors_pre",
    "dev_trend_pre",
    "reliability_risk_pre",
    "traffic_overtake_pre",
    "driver_team_priors_pre",
]
_OBSERVED_FILE_PATTERNS = (
    "results_{y}_{r}.csv",
    "results_{y}_{r}_Q.csv",
    "results_{y}_{r}_SQ.csv",
    "results_{y}_{r}_S.csv",
    "laps_{y}_{r}.csv",
    "laps_{y}_{r}_Q.csv",
    "laps_{y}_{r}_SQ.csv",
    "laps_{y}_{r}_S.csv",
    "laps_{y}_{r}_FP1.csv",
    "laps_{y}_{r}_FP2.csv",
    "laps_{y}_{r}_FP3.csv",
    "weather_{y}_{r}.csv",
    "weather_{y}_{r}_Q.csv",
    "weather_{y}_{r}_SQ.csv",
    "weather_{y}_{r}_S.csv",
    "weather_{y}_{r}_FP1.csv",
    "weather_{y}_{r}_FP2.csv",
    "weather_{y}_{r}_FP3.csv",
    "session_status_{y}_{r}.csv",
    "track_status_{y}_{r}.csv",
    "race_control_{y}_{r}.csv",
)


def _dedupe_strings(values: Iterable[object]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item or item.lower() == "nan" or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _extract_roster(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return []
    for col in _DRIVER_COLS:
        if col in df.columns and df[col].notna().any():
            return _dedupe_strings(df[col].tolist())
    return []


def _scan_known_roster_rounds(raw_dir: Path) -> List[Tuple[int, int]]:
    rounds: set[Tuple[int, int]] = set()
    for pattern in ("entrylist_*.csv", "results_*.csv"):
        for path in raw_dir.glob(pattern):
            match = re.search(r"_(\d{4})_(\d{1,2})(?:_[A-Za-z0-9]+)?\.csv$", path.name)
            if match:
                rounds.add((int(match.group(1)), int(match.group(2))))
    return sorted(rounds)


def load_roster_drivers(
    raw_dir: Path | str,
    year: int,
    rnd: int,
    drivers: Optional[Sequence[str]] = None,
) -> List[str]:
    raw_dir = Path(raw_dir)
    if drivers:
        return _dedupe_strings(drivers)

    for pattern in _ROSTER_FILES:
        vals = _extract_roster(_read_csv(raw_dir / pattern.format(y=year, r=rnd)))
        if vals:
            return vals
    return []


def load_latest_known_roster_drivers(
    raw_dir: Path | str,
    year: int,
    rnd: int,
    drivers: Optional[Sequence[str]] = None,
) -> List[str]:
    raw_dir = Path(raw_dir)
    current = load_roster_drivers(raw_dir, year, rnd, drivers)
    if current:
        return current
    if drivers:
        return _dedupe_strings(drivers)

    rounds = _scan_known_roster_rounds(raw_dir)
    usable = [(y, r) for (y, r) in rounds if (y < int(year)) or (y == int(year) and r < int(rnd))]
    usable.sort(key=lambda item: (item[0], item[1]), reverse=True)
    for y, r in usable:
        vals = load_roster_drivers(raw_dir, y, r, None)
        if vals:
            return vals
    return []


def has_observed_weekend_data(raw_dir: Path | str, year: int, rnd: int) -> bool:
    raw_dir = Path(raw_dir)
    return any((raw_dir / pattern.format(y=year, r=rnd)).exists() for pattern in _OBSERVED_FILE_PATTERNS)


def resolve_scenario_mode(
    raw_dir: Path | str,
    year: int,
    rnd: int,
    scenario_mode: str = "auto",
) -> str:
    mode = str(scenario_mode or "auto").strip().lower()
    if mode not in {"auto", "observed", "future"}:
        raise ValueError(f"Unsupported scenario mode: {scenario_mode}")
    if mode != "auto":
        return mode
    return "observed" if has_observed_weekend_data(raw_dir, year, rnd) else "future"


def resolve_official_track_name(raw_dir: Path | str, year: int, rnd: int) -> Optional[str]:
    raw_dir = Path(raw_dir)

    for name in (f"meta_{year}_{rnd}.csv", f"meta_{year}_{rnd}_Q.csv"):
        path = raw_dir / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        for col in _TRACK_COLS:
            if col in df.columns and df[col].notna().any():
                value = str(df[col].dropna().iloc[0]).strip()
                if value:
                    return value

    schedule_path = raw_dir / f"schedule_{year}.csv"
    if schedule_path.exists():
        sched = pd.read_csv(schedule_path)
        if not sched.empty:
            round_col = "round" if "round" in sched.columns else ("RoundNumber" if "RoundNumber" in sched.columns else None)
            if round_col is not None:
                rounds = pd.to_numeric(sched[round_col], errors="coerce")
                row = sched.loc[rounds == int(rnd)]
                if not row.empty:
                    for col in _TRACK_COLS:
                        if col in row.columns and row[col].notna().any():
                            value = str(row[col].dropna().iloc[0]).strip()
                            if value:
                                return value
    return None


def ordered_driver_slice(df: pd.DataFrame, drivers: Sequence[str]) -> pd.DataFrame:
    order = {drv: idx for idx, drv in enumerate(drivers)}
    out = df[df["Driver"].astype(str).isin(drivers)].copy()
    out["__order__"] = out["Driver"].astype(str).map(order)
    out = out.sort_values("__order__", kind="mergesort").drop_duplicates("Driver", keep="first")
    return out.drop(columns=["__order__"]).reset_index(drop=True)


def build_scenario_features(
    raw_dir: Path | str,
    sim_year: int,
    sim_round: int,
    *,
    track: Optional[str] = None,
    drivers: Optional[Sequence[str]] = None,
    mode: str = "auto",
    scenario_mode: str = "auto",
    allow_fallback_actual: bool = True,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Optional[str], List[str]]:
    raw_dir = Path(raw_dir)
    resolved_scenario_mode = resolve_scenario_mode(raw_dir, sim_year, sim_round, scenario_mode)
    if resolved_scenario_mode == "future":
        roster = load_latest_known_roster_drivers(raw_dir, sim_year, sim_round, drivers)
        modules = list(FUTURE_SAFE_MODULES)
    else:
        roster = load_roster_drivers(raw_dir, sim_year, sim_round, drivers)
        modules = None
    track_name = (str(track).strip() if track else "") or resolve_official_track_name(raw_dir, sim_year, sim_round)

    ctx = {
        "raw_dir": raw_dir,
        "year": int(sim_year),
        "round": int(sim_round),
        "mode": str(mode),
        "scenario_mode": resolved_scenario_mode,
        "allow_fallback_actual": bool(allow_fallback_actual),
        "verbose": bool(verbose),
    }
    if track_name:
        ctx["track"] = track_name
    if roster:
        ctx["drivers"] = roster
        ctx["roster"] = roster

    df = sanitize_frame_columns(featurize_pre(ctx, modules=modules))
    if df.empty:
        raise RuntimeError(f"Failed to rebuild scenario features from raw data ({resolved_scenario_mode} mode)")
    if "Driver" not in df.columns:
        raise RuntimeError("Scenario features are missing 'Driver'")

    if roster:
        df = ordered_driver_slice(df, roster)
        if df.empty:
            raise RuntimeError("Requested drivers are missing from rebuilt scenario features")
    else:
        df = df.drop_duplicates("Driver", keep="first").reset_index(drop=True)

    if "year" not in df.columns:
        df["year"] = np.int32(sim_year)
    else:
        df.loc[:, "year"] = np.int32(sim_year)
    if "round" not in df.columns:
        df["round"] = np.int32(sim_round)
    else:
        df.loc[:, "round"] = np.int32(sim_round)

    final_roster = _dedupe_strings(df["Driver"].tolist())
    return df, track_name, final_roster


__all__ = [
    "build_scenario_features",
    "FUTURE_SAFE_MODULES",
    "has_observed_weekend_data",
    "load_latest_known_roster_drivers",
    "load_roster_drivers",
    "ordered_driver_slice",
    "resolve_scenario_mode",
    "resolve_official_track_name",
]
