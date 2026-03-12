from __future__ import annotations

from pathlib import Path
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
        df = pd.read_csv(raw_dir / pattern.format(y=year, r=rnd)) if (raw_dir / pattern.format(y=year, r=rnd)).exists() else pd.DataFrame()
        if df.empty:
            continue
        for col in _DRIVER_COLS:
            if col in df.columns and df[col].notna().any():
                vals = _dedupe_strings(df[col].tolist())
                if vals:
                    return vals
    return []


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
    allow_fallback_actual: bool = True,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Optional[str], List[str]]:
    raw_dir = Path(raw_dir)
    roster = load_roster_drivers(raw_dir, sim_year, sim_round, drivers)
    track_name = (str(track).strip() if track else "") or resolve_official_track_name(raw_dir, sim_year, sim_round)

    ctx = {
        "raw_dir": raw_dir,
        "year": int(sim_year),
        "round": int(sim_round),
        "mode": str(mode),
        "allow_fallback_actual": bool(allow_fallback_actual),
        "verbose": bool(verbose),
    }
    if track_name:
        ctx["track"] = track_name
    if roster:
        ctx["drivers"] = roster
        ctx["roster"] = roster

    df = sanitize_frame_columns(featurize_pre(ctx))
    if df.empty:
        raise RuntimeError("Failed to rebuild scenario features from raw data")
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
    "load_roster_drivers",
    "ordered_driver_slice",
    "resolve_official_track_name",
]
