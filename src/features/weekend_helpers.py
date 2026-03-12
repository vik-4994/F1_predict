from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .utils import read_csv_if_exists

__all__ = [
    "current_roster",
    "current_team_map",
    "pairwise_team_delta",
    "to_seconds",
]

_DRIVER_COLS = ("Abbreviation", "Driver", "code", "driverRef", "BroadcastName")
_TEAM_COLS = ("TeamName", "Team", "Constructor", "ConstructorName")
_ROSTER_FILES = (
    "results_{year}_{rnd}_Q.csv",
    "entrylist_{year}_{rnd}_Q.csv",
    "results_{year}_{rnd}.csv",
    "entrylist_{year}_{rnd}.csv",
)


def current_roster(raw_dir: Path, year: int, rnd: int, drivers: Optional[Iterable[str]] = None) -> List[str]:
    if drivers:
        vals = [str(d).strip() for d in drivers if str(d).strip()]
        if vals:
            return list(dict.fromkeys(vals))

    for pattern in _ROSTER_FILES:
        df = read_csv_if_exists(raw_dir / pattern.format(year=year, rnd=rnd))
        if df.empty:
            continue
        for col in _DRIVER_COLS:
            if col in df.columns and df[col].notna().any():
                vals = df[col].astype(str).dropna().drop_duplicates().tolist()
                if vals:
                    return vals
    return []


def current_team_map(raw_dir: Path, year: int, rnd: int) -> pd.DataFrame:
    for pattern in _ROSTER_FILES:
        df = read_csv_if_exists(raw_dir / pattern.format(year=year, rnd=rnd))
        if df.empty:
            continue
        dcol = next((c for c in _DRIVER_COLS if c in df.columns), None)
        tcol = next((c for c in _TEAM_COLS if c in df.columns), None)
        if dcol and tcol:
            out = pd.DataFrame({"Driver": df[dcol].astype(str), "Team": df[tcol].astype(str)})
            out = out.dropna(subset=["Driver"]).drop_duplicates("Driver")
            if not out.empty:
                return out
    return pd.DataFrame(columns=["Driver", "Team"])


def to_seconds(series: pd.Series) -> pd.Series:
    try:
        td = pd.to_timedelta(series, errors="coerce")
        if td.notna().any():
            sec = td.dt.total_seconds()
            if sec.notna().mean() > 0.5:
                return sec
    except Exception:
        pass
    num = pd.to_numeric(series, errors="coerce")
    med = num.replace([np.inf, -np.inf], np.nan).median()
    if pd.notna(med) and med > 1e3:
        return num / 1000.0
    return num


def pairwise_team_delta(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    if df.empty or "Team" not in df.columns:
        return df

    out = df.copy()
    team_size = out.groupby("Team")["Driver"].transform("nunique")
    out["__tm_has_pair__"] = (team_size == 2).astype(float)

    mate = out[["Driver", "Team", *cols]].copy()
    mate = mate.rename(columns={"Driver": "Teammate", **{c: f"{c}__mate" for c in cols}})
    paired = out.merge(mate, on="Team", how="left")
    paired = paired[paired["Driver"] != paired["Teammate"]].copy()
    if paired.empty:
        for col in cols:
            out[f"{col}_tm_delta"] = np.nan
        out["tm_has_pair"] = out["__tm_has_pair__"]
        return out.drop(columns=["__tm_has_pair__"])

    keep = ["Driver", "Team", "__tm_has_pair__"]
    for col in cols:
        paired[f"{col}_tm_delta"] = pd.to_numeric(paired[col], errors="coerce") - pd.to_numeric(
            paired[f"{col}__mate"],
            errors="coerce",
        )
        keep.append(f"{col}_tm_delta")

    reduced = paired[keep].drop_duplicates("Driver")
    reduced = reduced.rename(columns={"__tm_has_pair__": "tm_has_pair"})
    out = out.drop(columns=["__tm_has_pair__"], errors="ignore").merge(reduced, on=["Driver", "Team"], how="left")
    return out
