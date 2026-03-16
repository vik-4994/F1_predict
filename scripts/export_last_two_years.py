#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export all FastF1 datasets needed for our pre-race features.

What it exports (into --out-dir, default: data/raw_csv):
Per race (year, round) and per requested session (default: R Q S SQ FP1 FP2 FP3):
  • laps_{Y}_{R}[_SES].csv        – full Laps table (+ 'milliseconds' numeric)
  • weather_{Y}_{R}[_SES].csv     – session weather data
  • results_{Y}_{R}[_SES].csv     – FastF1 session results (for reference/QA)
  • meta_{Y}_{R}[_SES].csv        – metadata (event name, location, date)
  • race_ctrl_{Y}_{R}.csv         – Race Control messages (R only, if available)
  • track_status_{Y}_{R}.csv      – Track Status log (R only, if available)
  • entrylist_{Y}_{R}[_SES].csv   – driver list (Abbreviation, DriverNumber, TeamName)
  • stints_{Y}_{R}[_SES].csv      – stint summary from Laps (Stint×Compound×laps)
  • telemetry_{Y}_{R}[_SES]_{ABBR}.csv  – car telemetry per driver (downsampled)
  • position_{Y}_{R}[_SES]_{ABBR}.csv   – XY position time series per driver (if available)

Additionally aggregates:
  • schedule_{Y}.csv              – season schedule snapshot
  • weather.csv                   – per-race median Air/Track temps across exported seasons

Notes:
- We do NOT fabricate Ergast-style 'driverId' / 'constructorId'. Our feature loaders will
  map by Abbreviation/DriverNumber using existing Ergast CSVs where needed.
- Timedelta columns are stringified for safer CSV IO; an extra 'milliseconds' column is added to Laps.

Usage examples:
  python scripts/export_fastf1_datasets.py
  python scripts/export_fastf1_datasets.py --years 2024 2025 --sessions R Q S --telemetry-stride 5 --max-workers 4 --skip-existing
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Iterable, Dict, List, Tuple

import pandas as pd
import numpy as np

try:
    import fastf1  # type: ignore[import-not-found]
    from fastf1 import _api as ffapi  # type: ignore[import-not-found]
    _FASTF1_IMPORT_ERROR: Optional[ModuleNotFoundError] = None
except ModuleNotFoundError as exc:
    _FASTF1_IMPORT_ERROR = exc

    class _MissingCache:
        @staticmethod
        def enable_cache(*_args: Any, **_kwargs: Any) -> None:
            raise ModuleNotFoundError(
                "fastf1 is required for cache-enabled export workflows. "
                "Install the project dependencies with `pip install -r requirements.txt`."
            ) from _FASTF1_IMPORT_ERROR

    class _MissingFastF1:
        __version__ = ""
        Cache = _MissingCache

        @staticmethod
        def get_session(*_args: Any, **_kwargs: Any) -> Any:
            raise ModuleNotFoundError(
                "fastf1 is required for session export workflows. "
                "Install the project dependencies with `pip install -r requirements.txt`."
            ) from _FASTF1_IMPORT_ERROR

        @staticmethod
        def get_event_schedule(*_args: Any, **_kwargs: Any) -> Any:
            raise ModuleNotFoundError(
                "fastf1 is required for schedule export workflows. "
                "Install the project dependencies with `pip install -r requirements.txt`."
            ) from _FASTF1_IMPORT_ERROR

    class _MissingFastF1Api:
        @staticmethod
        def driver_info(*_args: Any, **_kwargs: Any) -> Any:
            raise ModuleNotFoundError("fastf1 is required for low-level export API access.") from _FASTF1_IMPORT_ERROR

        @staticmethod
        def session_status_data(*_args: Any, **_kwargs: Any) -> Any:
            raise ModuleNotFoundError("fastf1 is required for low-level export API access.") from _FASTF1_IMPORT_ERROR

        @staticmethod
        def _extended_timing_data(*_args: Any, **_kwargs: Any) -> Any:
            raise ModuleNotFoundError("fastf1 is required for low-level export API access.") from _FASTF1_IMPORT_ERROR

        @staticmethod
        def timing_app_data(*_args: Any, **_kwargs: Any) -> Any:
            raise ModuleNotFoundError("fastf1 is required for low-level export API access.") from _FASTF1_IMPORT_ERROR

    fastf1 = _MissingFastF1()
    ffapi = _MissingFastF1Api()


def _require_fastf1() -> None:
    if _FASTF1_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "fastf1 is not installed. Install the project dependencies with "
            "`pip install -r requirements.txt` before running raw-data export commands."
        ) from _FASTF1_IMPORT_ERROR


                               
            
                               

def setup_dirs(out_dir: Path, cache_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))


def _stringify_timedeltas(df: pd.DataFrame) -> pd.DataFrame:
    """Convert any timedelta-like columns to string to avoid CSV issues."""
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_timedelta64_dtype(df[c]):
            df[c] = df[c].astype("string")
    return df


def safe_to_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    _stringify_timedeltas(df).to_csv(path, index=False)


def _add_race_id(df: pd.DataFrame, year: int, rnd: int) -> pd.DataFrame:
    """Attach a synthetic raceId (year*1000+round) if absent."""
    rid = int(year) * 1000 + int(rnd)
    if "raceId" not in df.columns:
        df = df.copy()
        df.insert(0, "raceId", rid)
    return df


                               
                   
                               

def laps_with_ms(laps: pd.DataFrame) -> pd.DataFrame:
    """Ensure a numeric 'milliseconds' column is present (from LapTime)."""
    df = laps.copy()
    if "milliseconds" not in df.columns:
        if "LapTime" in df.columns:
                                                                          
            ms = pd.to_timedelta(df["LapTime"], errors="coerce").dt.total_seconds() * 1000.0
            df["milliseconds"] = ms.astype(float)
        else:
                                            
            if "Time" in df.columns and pd.api.types.is_timedelta64_dtype(df["Time"]):
                df["milliseconds"] = df["Time"].dt.total_seconds() * 1000.0
            else:
                df["milliseconds"] = np.nan
    return df


def derive_pitstops_from_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Derive pit stops from Laps (PitInTime/PitOutTime). Returns raceId, Driver, DriverNumber, lap, duration_ms."""
    if laps is None or laps.empty:
        return pd.DataFrame(columns=["raceId", "Driver", "DriverNumber", "lap", "duration_ms"])
    df = laps.copy()
    cols = set(df.columns)
                                                       
    if not {"LapNumber"}.issubset(cols):
        return pd.DataFrame(columns=["raceId", "Driver", "DriverNumber", "lap", "duration_ms"])
    df["lap"] = pd.to_numeric(df["LapNumber"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["lap"]).astype({"lap": int})
                                                                
    has_pitin = "PitInTime" in cols
    has_pitout = "PitOutTime" in cols
    if not has_pitin and not has_pitout:
                                                                                                              
        return pd.DataFrame(columns=["raceId", "Driver", "DriverNumber", "lap", "duration_ms"])
    pit = df.loc[df["PitInTime"].notna()] if has_pitin else pd.DataFrame(columns=df.columns)
    if has_pitin and has_pitout:
        dur = (pd.to_timedelta(pit["PitOutTime"], errors="coerce") - pd.to_timedelta(pit["PitInTime"], errors="coerce")).dt.total_seconds() * 1000.0
    else:
        dur = pd.Series(np.nan, index=pit.index, dtype=float)
    out = pit[["raceId", "Driver", "DriverNumber", "lap"]].copy()
    out["duration_ms"] = dur.astype(float)
    return out.reset_index(drop=True)


def stints_from_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Aggregate stints per driver using Stint/Compound columns in Laps."""
    if laps is None or laps.empty:
        return pd.DataFrame(columns=["raceId", "Driver", "DriverNumber", "Stint", "Compound", "laps"])
    cols = set(laps.columns)
    if not {"Stint", "Compound", "Driver", "DriverNumber"}.issubset(cols):
        return pd.DataFrame(columns=["raceId", "Driver", "DriverNumber", "Stint", "Compound", "laps"])
    df = laps[["raceId", "Driver", "DriverNumber", "Stint", "Compound"]].copy()
    df["laps"] = 1
    return df.groupby(["raceId", "Driver", "DriverNumber", "Stint", "Compound"])['laps'].sum().reset_index()


def entrylist_from_session(session) -> pd.DataFrame:
    rows = []
    for drv in session.drivers:
        info = session.get_driver(drv)                                                                       
        rows.append({
            "raceId": int(session.event["EventDate"].year) * 1000 + int(session.event["RoundNumber"]),
            "DriverNumber": info.get("DriverNumber"),
            "Abbreviation": info.get("Abbreviation"),
            "BroadcastName": info.get("BroadcastName"),
            "TeamName": info.get("TeamName"),
            "FirstName": info.get("FirstName"),
            "LastName": info.get("LastName"),
        })
    return pd.DataFrame(rows)


def schedule_for_year(year: int) -> pd.DataFrame:
    try:
        sched = fastf1.get_event_schedule(year)
        keep = [
            c
            for c in [
                "EventName",
                "OfficialEventName",
                "Location",
                "Country",
                "EventDate",
                "EventFormat",
                "F1ApiSupport",
                "RoundNumber",
                "Session1",
                "Session1Date",
                "Session1DateUtc",
                "Session2",
                "Session2Date",
                "Session2DateUtc",
                "Session3",
                "Session3Date",
                "Session3DateUtc",
                "Session4",
                "Session4Date",
                "Session4DateUtc",
                "Session5",
                "Session5Date",
                "Session5DateUtc",
            ]
            if c in sched.columns
        ]
        df = sched[keep].copy().reset_index(drop=True)
                                                  
        if "RoundNumber" in df.columns:
            df.insert(0, "raceId", [int(year) * 1000 + int(r) for r in df["RoundNumber"]])
        else:
            df.insert(0, "raceId", np.arange(1, df.shape[0]+1) + int(year) * 1000)
        df.insert(1, "year", int(year))
        df.rename(columns={"RoundNumber": "round"}, inplace=True)
        return df
    except Exception as e:
        return pd.DataFrame()


_SESSION_ALIASES: Dict[str, set[str]] = {
    "FP1": {"FP1", "Practice 1"},
    "FP2": {"FP2", "Practice 2"},
    "FP3": {"FP3", "Practice 3"},
    "Q": {"Q", "Qualifying"},
    "R": {"R", "Race"},
    "S": {"S", "Sprint"},
    "SQ": {"SQ", "Sprint Qualifying"},
    "SS": {"SS", "Sprint Shootout"},
}


def _utc_naive(ts_like) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(ts_like)
    except Exception:
        return pd.NaT
    if pd.isna(ts):
        return pd.NaT
    if ts.tzinfo is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return ts


def _session_date_from_schedule_row(row: pd.Series, session_code: str) -> pd.Timestamp:
    want = str(session_code).upper()
    aliases = _SESSION_ALIASES.get(want, {want})
    for idx in range(1, 6):
        name = str(row.get(f"Session{idx}", "")).strip()
        if not name:
            continue
        if name in aliases or name.upper() == want:
            for col in (f"Session{idx}DateUtc", f"Session{idx}Date"):
                if col in row.index:
                    ts = _utc_naive(row.get(col))
                    if pd.notna(ts):
                        return ts
    if want == "R":
        return _utc_naive(row.get("EventDate"))
    return pd.NaT


def resolve_rounds_from_schedule(
    schedule: pd.DataFrame,
    sessions: Iterable[str],
    *,
    completed_only: bool = False,
    latest_only: bool = False,
    lookback_rounds: Optional[int] = None,
    now_utc: Optional[pd.Timestamp] = None,
) -> List[int]:
    if schedule is None or schedule.empty:
        return []

    now = _utc_naive(now_utc or datetime.now(timezone.utc))
    round_col = "RoundNumber" if "RoundNumber" in schedule.columns else "round"
    if round_col not in schedule.columns:
        return []

    rounds: List[int] = []
    sessions = [str(s).upper() for s in sessions]
    work = schedule.copy()
    work[round_col] = pd.to_numeric(work[round_col], errors="coerce")
    work = work[work[round_col] > 0].sort_values(round_col)

    for _, row in work.iterrows():
        rnd = int(row[round_col])
        if not completed_only:
            rounds.append(rnd)
            continue
        session_dates = [_session_date_from_schedule_row(row, ses) for ses in sessions]
        session_dates = [ts for ts in session_dates if pd.notna(ts)]
        if not session_dates:
            event_ts = _utc_naive(row.get("EventDate"))
            if pd.notna(event_ts) and event_ts <= now:
                rounds.append(rnd)
            continue
        if all(ts <= now for ts in session_dates):
            rounds.append(rnd)

    if lookback_rounds is not None and lookback_rounds > 0:
        rounds = rounds[-int(lookback_rounds):]
    if latest_only and rounds:
        rounds = [rounds[-1]]
    return rounds


def completed_rounds_for_year(
    year: int,
    sessions: Iterable[str],
    *,
    latest_only: bool = False,
    lookback_rounds: Optional[int] = None,
) -> List[int]:
    try:
        sched = fastf1.get_event_schedule(year)
    except Exception:
        return []
    return resolve_rounds_from_schedule(
        sched,
        sessions,
        completed_only=True,
        latest_only=latest_only,
        lookback_rounds=lookback_rounds,
    )


def _session_codes_from_schedule_row(row: pd.Series) -> List[str]:
    codes: List[str] = []
    for idx in range(1, 6):
        code = _normalize_session_code(row.get(f"Session{idx}", ""))
        if code:
            codes.append(code)
    return codes


def _normalize_session_code(name: object) -> Optional[str]:
    text = str(name or "").strip()
    if not text:
        return None
    upper = text.upper()
    for code, aliases in _SESSION_ALIASES.items():
        if text in aliases or upper in {alias.upper() for alias in aliases}:
            return code
    return None


def _session_anchor_path(out_dir: Path, year: int, rnd: int, session: str) -> Path:
    suffix = "" if session == "R" else f"_{session}"
    return out_dir / f"laps_{year}_{rnd}{suffix}.csv"


def missing_rounds_from_schedule(
    schedule: pd.DataFrame,
    out_dir: Path,
    sessions: Iterable[str],
    *,
    completed_only: bool = True,
    now_utc: Optional[pd.Timestamp] = None,
) -> List[int]:
    sessions = [str(s).upper() for s in sessions]
    rounds = resolve_rounds_from_schedule(
        schedule,
        sessions,
        completed_only=completed_only,
        latest_only=False,
        lookback_rounds=None,
        now_utc=now_utc,
    )
    if not rounds:
        return []

    round_col = "RoundNumber" if "RoundNumber" in schedule.columns else "round"
    missing: List[int] = []
    for rnd in rounds:
        row = schedule.loc[pd.to_numeric(schedule[round_col], errors="coerce") == int(rnd)]
        if row.empty:
            continue
        available_codes = set(_session_codes_from_schedule_row(row.iloc[0]))
        requested = [code for code in sessions if code in available_codes]
        if not requested:
            continue
        event_year = row.iloc[0].get("year", row.iloc[0].get("Year", np.nan))
        if pd.isna(event_year) and "raceId" in row.columns:
            race_id = pd.to_numeric(row.iloc[0].get("raceId"), errors="coerce")
            if pd.notna(race_id):
                event_year = int(race_id) // 1000
        event_year = int(event_year) if pd.notna(event_year) else 0
        if any(not _session_anchor_path(out_dir, event_year, int(rnd), ses).exists() for ses in requested):
            missing.append(int(rnd))
    return sorted(set(missing))


def missing_completed_rounds_for_year(
    out_dir: Path,
    year: int,
    sessions: Iterable[str],
    *,
    now_utc: Optional[pd.Timestamp] = None,
) -> List[int]:
    try:
        sched = fastf1.get_event_schedule(year)
    except Exception:
        return []
    sched = sched.copy()
    if "year" not in sched.columns:
        sched["year"] = int(year)
    return missing_rounds_from_schedule(sched, out_dir, sessions, completed_only=True, now_utc=now_utc)


def per_race_weather_summary(year: int, rnd: int, ses: str, weather_df: pd.DataFrame) -> Dict[str, float]:
    """Median temps for weather.csv aggregate."""
    ww = weather_df.copy()
    out = {
        "raceId": int(year) * 1000 + int(rnd),
        "year": int(year),
        "round": int(rnd),
        "session": ses
    }
    for col, new in [("AirTemp", "air_temp_C"), ("TrackTemp", "track_temp_C"), ("Humidity", "humidity"), ("WindSpeed", "wind_speed_ms")]:
        if col in ww.columns:
            try:
                v = pd.to_numeric(ww[col], errors="coerce").median()
                out[new] = float(v) if pd.notna(v) else np.nan
            except Exception:
                out[new] = np.nan
        else:
            out[new] = np.nan
    return out


def _fallback_driver_info(api_path: str) -> pd.DataFrame:
    try:
        raw = ffapi.driver_info(api_path)
    except Exception:
        return pd.DataFrame(columns=["DriverNumber", "Abbreviation", "BroadcastName", "TeamName", "FirstName", "LastName"])
    rows = []
    for key, entry in (raw or {}).items():
        rows.append(
            {
                "DriverNumber": str(entry.get("RacingNumber") or key),
                "Abbreviation": entry.get("Tla"),
                "BroadcastName": entry.get("BroadcastName"),
                "TeamName": entry.get("TeamName"),
                "FirstName": entry.get("FirstName"),
                "LastName": entry.get("LastName"),
            }
        )
    return pd.DataFrame(rows)


def _fallback_session_start_time(api_path: str) -> Optional[pd.Timedelta]:
    try:
        status = ffapi.session_status_data(api_path)
    except Exception:
        return pd.NaT
    if not status:
        return pd.NaT
    try:
        df = pd.DataFrame(status)
    except Exception:
        return pd.NaT
    if df.empty or "Status" not in df.columns or "Time" not in df.columns:
        return pd.NaT
    started = df.loc[df["Status"].astype(str) == "Started", "Time"]
    if started.empty:
        return pd.NaT
    return started.iloc[0]


def _fallback_build_laps(api_path: str, session_name: str) -> pd.DataFrame:
    laps_data, _stream, _splits = ffapi._extended_timing_data(api_path)
    if laps_data is None or laps_data.empty:
        return pd.DataFrame()

    app_data = ffapi.timing_app_data(api_path)
    app_data = app_data.reset_index(drop=True) if isinstance(app_data, pd.DataFrame) else pd.DataFrame(app_data)
    d_info = _fallback_driver_info(api_path)
    session_start_time = _fallback_session_start_time(api_path)

    base = laps_data.copy().reset_index(drop=True)
    if app_data is None or app_data.empty:
        base["Compound"] = ""
        base["TyreLife"] = np.nan
        base["Stint"] = 1
        base["FreshTyre"] = False
    else:
        merged_parts = []
        for drv, d1 in base.groupby("Driver", sort=False):
            d2 = app_data.loc[app_data["Driver"].astype(str) == str(drv)].copy()
            d1 = d1.sort_values("Time").reset_index(drop=True)
            if d2.empty:
                d1["Compound"] = ""
                d1["TyreLife"] = np.nan
                d1["Stint"] = 1
                d1["FreshTyre"] = False
            else:
                d2 = d2.sort_values("Time").reset_index(drop=True)
                merged = pd.merge_asof(d1, d2, on="Time", by="Driver", direction="backward")
                merged["Compound"] = merged["Compound"].fillna("")
                merged["TyreLife"] = pd.to_numeric(merged.get("StartLaps"), errors="coerce")
                merged["Stint"] = pd.to_numeric(merged.get("Stint"), errors="coerce").fillna(0).astype(int) + 1
                merged["FreshTyre"] = merged.get("New", False).fillna(False).astype(bool)
                d1 = merged
            merged_parts.append(d1)
        base = pd.concat(merged_parts, ignore_index=True, sort=False) if merged_parts else base.iloc[0:0].copy()

    base = base.rename(columns={"Driver": "DriverNumber", "NumberOfLaps": "LapNumber"})
    base["DriverNumber"] = base["DriverNumber"].astype(str)
    if not d_info.empty:
        d_info["DriverNumber"] = d_info["DriverNumber"].astype(str)
        base = base.merge(d_info, on="DriverNumber", how="left")
    else:
        base["Abbreviation"] = base["DriverNumber"]
        base["TeamName"] = np.nan

    base["Driver"] = base.get("Abbreviation", base["DriverNumber"]).fillna(base["DriverNumber"]).astype(str)
    base["Team"] = base.get("TeamName", np.nan)
    base["LapStartTime"] = (
        base.sort_values(["DriverNumber", "Time"], kind="mergesort")
        .groupby("DriverNumber", sort=False)["Time"]
        .shift(1)
    )
    if pd.notna(session_start_time):
        is_race_like = str(session_name).strip().lower() in {"race", "sprint", "qualifying", "sprint qualifying", "sprint shootout"}
        if is_race_like:
            first_mask = base["LapStartTime"].isna()
            base.loc[first_mask, "LapStartTime"] = session_start_time
    base["Position"] = pd.to_numeric(base.get("Position"), errors="coerce")
    base["IsAccurate"] = base["LapTime"].notna()
    base["TrackStatus"] = ""
    keep_cols = [
        "Driver",
        "DriverNumber",
        "Team",
        "LapNumber",
        "LapTime",
        "Time",
        "LapStartTime",
        "Stint",
        "Compound",
        "TyreLife",
        "FreshTyre",
        "Position",
        "PitOutTime",
        "PitInTime",
        "Sector1Time",
        "Sector2Time",
        "Sector3Time",
        "Sector1SessionTime",
        "Sector2SessionTime",
        "Sector3SessionTime",
        "SpeedI1",
        "SpeedI2",
        "SpeedFL",
        "SpeedST",
        "IsPersonalBest",
        "IsAccurate",
        "TrackStatus",
    ]
    for col in keep_cols:
        if col not in base.columns:
            base[col] = np.nan
    return base[keep_cols].reset_index(drop=True)


def _is_unavailable_session_error(exc: Exception) -> bool:
    msg = str(exc).strip().lower()
    patterns = (
        "invalid session",
        "session does not exist",
        "session is not part of this event",
        "event does not have",
        "no session",
        "unknown session",
    )
    return any(p in msg for p in patterns)


                               
              
                               

def export_one_session(year: int, rnd: int, ses: str, out_dir: Path, telemetry_stride: int, driver_limit: Optional[int]) -> Dict[str, object]:
    tag = f"{year} R{rnd:02d} {ses}"
    t0 = time.time()
    try:
        warnings_list: List[str] = []
        failed_step: Optional[str] = None

        def _capture_optional(name: str, fn) -> None:
            try:
                fn()
            except Exception as exc:
                warnings_list.append(f"{name}: {exc}")

        def _optional_frame(name: str, getter) -> pd.DataFrame:
            try:
                df = getter()
            except Exception as exc:
                warnings_list.append(f"{name}: {exc}")
                return pd.DataFrame()
            if df is None:
                return pd.DataFrame()
            return df.reset_index(drop=True) if isinstance(df, pd.DataFrame) else pd.DataFrame(df).reset_index(drop=True)

        def _optional_value(name: str, getter, default=""):
            try:
                value = getter()
            except Exception as exc:
                warnings_list.append(f"{name}: {exc}")
                return default
            return default if value is None else value

        session = fastf1.get_session(year, rnd, ses)
        # Car telemetry is fetched per-driver later; preloading it here is both slower
        # and has been brittle on freshly published weekends.
        session.load(laps=True, telemetry=False, weather=True, messages=True)

        suffix = "" if ses == "R" else f"_{ses}"

        meta = {
            "raceId": int(year) * 1000 + int(rnd),
            "Year": year, "Round": rnd, "Session": ses,
            "EventName": session.event.get("EventName", ""),
            "OfficialEventName": session.event.get("OfficialEventName", ""),
            "Location": session.event.get("Location", ""),
            "Country": session.event.get("Country", ""),
            "EventDate": str(session.event.get("EventDate", "")),
            "EventFormat": session.event.get("EventFormat", ""),
            "F1ApiSupport": bool(getattr(session, "f1_api_support", False)),
            "SessionName": _optional_value("session.name", lambda: session.name, ses),
            "SessionDate": str(_optional_value("session.date", lambda: session.date, "")),
            "SessionStartTime": str(_optional_value("session.session_start_time", lambda: session.session_start_time, "")),
            "T0Date": str(_optional_value("session.t0_date", lambda: session.t0_date, "")),
            "ApiPath": _optional_value("session.api_path", lambda: session.api_path, ""),
        }
        meta_df = pd.DataFrame([meta])
        safe_to_csv(meta_df, out_dir / f"meta_{year}_{rnd}{suffix}.csv")

        def _save_entrylist() -> None:
            ent = entrylist_from_session(session)
            if not ent.empty:
                safe_to_csv(ent, out_dir / f"entrylist_{year}_{rnd}{suffix}.csv")
        _capture_optional("entrylist", _save_entrylist)

        def _save_session_info() -> None:
            info = getattr(session, "session_info", None)
            if info is None:
                return
            if isinstance(info, dict):
                sdf = pd.DataFrame([info])
            else:
                sdf = pd.DataFrame([dict(info)])
            if not sdf.empty:
                sdf = _add_race_id(sdf, year, rnd)
                safe_to_csv(sdf, out_dir / f"session_info_{year}_{rnd}{suffix}.csv")
        _capture_optional("session_info", _save_session_info)

        weather = _optional_frame("weather_data", lambda: session.weather_data)
        if not weather.empty:
            weather = _add_race_id(weather, year, rnd)
            safe_to_csv(weather, out_dir / f"weather_{year}_{rnd}{suffix}.csv")

        results = _optional_frame("results", lambda: session.results)
        if not results.empty:
            results = _add_race_id(results, year, rnd)
            safe_to_csv(results, out_dir / f"results_{year}_{rnd}{suffix}.csv")

        try:
            laps = session.laps.reset_index(drop=True)
            laps = _add_race_id(laps, year, rnd)
            laps = laps_with_ms(laps)
            safe_to_csv(laps, out_dir / f"laps_{year}_{rnd}{suffix}.csv")
        except Exception as exc:
            warnings_list.append(f"laps: {exc}")
            try:
                laps = _fallback_build_laps(session.api_path, getattr(session, "name", ses))
                if not laps.empty:
                    laps = _add_race_id(laps, year, rnd)
                    laps = laps_with_ms(laps)
                    safe_to_csv(laps, out_dir / f"laps_{year}_{rnd}{suffix}.csv")
                    warnings_list.append("laps_fallback_api: used _extended_timing_data + timing_app_data")
                else:
                    failed_step = "laps"
            except Exception as fallback_exc:
                failed_step = "laps"
                warnings_list.append(f"laps_fallback_api: {fallback_exc}")
                laps = pd.DataFrame()

        def _save_session_status() -> None:
            ss = session.session_status.reset_index(drop=True)
            if not ss.empty:
                ss = _add_race_id(ss, year, rnd)
                safe_to_csv(ss, out_dir / f"session_status_{year}_{rnd}{suffix}.csv")
        _capture_optional("session_status", _save_session_status)

                          
        if ses == "R":
            def _save_race_ctrl() -> None:
                rcm = session.race_control_messages.reset_index(drop=True)
                rcm = _add_race_id(rcm, year, rnd)
                safe_to_csv(rcm, out_dir / f"race_ctrl_{year}_{rnd}.csv")
            _capture_optional("race_ctrl", _save_race_ctrl)

            def _save_track_status() -> None:
                ts = session.track_status.reset_index(drop=True)
                ts = _add_race_id(ts, year, rnd)
                safe_to_csv(ts, out_dir / f"track_status_{year}_{rnd}.csv")
            _capture_optional("track_status", _save_track_status)

        tel_cnt = 0
        pos_cnt = 0
        pit_cnt = 0
        if not laps.empty:
            def _save_stints() -> None:
                st = stints_from_laps(laps)
                if not st.empty:
                    safe_to_csv(st, out_dir / f"stints_{year}_{rnd}{suffix}.csv")
            _capture_optional("stints", _save_stints)

            drivers: List[int] = list(session.drivers)
            if driver_limit is not None:
                drivers = drivers[:driver_limit]

            for drv in drivers:
                info = session.get_driver(drv)
                abbr = info.get("Abbreviation", str(drv))

                def _save_telemetry() -> None:
                    nonlocal tel_cnt
                    car = session.laps.pick_driver(drv).get_car_data()
                    if car is not None and len(car):
                        tdf = car.reset_index(drop=True)
                        tdf = _add_race_id(tdf, year, rnd)
                        if telemetry_stride and telemetry_stride > 1:
                            tdf = tdf.iloc[::telemetry_stride].reset_index(drop=True)
                        safe_to_csv(tdf, out_dir / f"telemetry_{year}_{rnd}{suffix}_{abbr}.csv")
                        tel_cnt += 1
                _capture_optional(f"telemetry:{abbr}", _save_telemetry)

                def _save_position() -> None:
                    nonlocal pos_cnt
                    pos = session.laps.pick_driver(drv).get_pos_data()
                    if pos is not None and len(pos):
                        pdf = pos.reset_index(drop=True)
                        pdf = _add_race_id(pdf, year, rnd)
                        safe_to_csv(pdf, out_dir / f"position_{year}_{rnd}{suffix}_{abbr}.csv")
                        pos_cnt += 1
                _capture_optional(f"position:{abbr}", _save_position)

        if ses == "R" and not laps.empty:
            def _save_pit_stops() -> None:
                nonlocal pit_cnt
                pits = derive_pitstops_from_laps(laps)
                if not pits.empty:
                    safe_to_csv(pits, out_dir / f"pit_stops_{year}_{rnd}.csv")
                    pit_cnt = pits.shape[0]
            _capture_optional("pit_stops", _save_pit_stops)

        if failed_step == "laps":
            status = f"partial_missing_laps{':' + str(len(warnings_list)) if warnings_list else ''}"
        else:
            status = "ok" if not warnings_list else f"ok_with_warnings:{len(warnings_list)}"
        return {
            "race": tag,
            "year": int(year),
            "round": int(rnd),
            "session": str(ses),
            "status": status,
            "failed_step": failed_step,
            "drivers_tel": tel_cnt,
            "drivers_pos": pos_cnt,
            "pit_rows": pit_cnt,
            "warning_count": len(warnings_list),
            "warnings": warnings_list[:5],
            "secs": time.time() - t0,
        }

    except Exception as e:
        if _is_unavailable_session_error(e):
            return {
                "race": tag,
                "year": int(year),
                "round": int(rnd),
                "session": str(ses),
                "status": "skipped_unavailable_session",
                "failed_step": None,
                "drivers_tel": 0,
                "drivers_pos": 0,
                "pit_rows": 0,
                "warning_count": 0,
                "warnings": [],
                "secs": time.time() - t0,
            }
        return {
            "race": tag,
            "year": int(year),
            "round": int(rnd),
            "session": str(ses),
            "status": f"error: {e}",
            "failed_step": failed_step,
            "drivers_tel": 0,
            "drivers_pos": 0,
            "pit_rows": 0,
            "warning_count": len(warnings_list) if 'warnings_list' in locals() else 0,
            "warnings": warnings_list[:5] if 'warnings_list' in locals() else [],
            "secs": time.time() - t0,
        }


def get_max_round(year: int) -> int:
    try:
        sched = fastf1.get_event_schedule(year)
        return int(sched["RoundNumber"].max())
    except Exception:
                                
        return 25


def export_season(
    year: int,
    sessions: Iterable[str],
    out_dir: Path,
    telemetry_stride: int,
    max_workers: int,
    driver_limit: Optional[int],
    skip_existing: bool,
    rounds: Optional[Iterable[int]] = None,
) -> List[Dict[str, object]]:
    round_list = sorted(set(int(r) for r in rounds)) if rounds is not None else list(range(1, get_max_round(year) + 1))
    if not round_list:
        return []
    tasks = []
    logs = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for rnd in round_list:
            for ses in sessions:
                if skip_existing:
                                                                         
                    suffix = "" if ses == "R" else f"_{ses}"
                    if (out_dir / f"laps_{year}_{rnd}{suffix}.csv").exists():
                        continue
                tasks.append(ex.submit(export_one_session, year, rnd, ses, out_dir, telemetry_stride, driver_limit))
        for fut in as_completed(tasks):
            res = fut.result()
            logs.append(res)
            warn_tail = f", warn:{res.get('warning_count', 0)}" if res.get("warning_count", 0) else ""
            print(f"• {res['race']}: {res['status']} (tel:{res['drivers_tel']}, pos:{res['drivers_pos']}, pit:{res['pit_rows']}{warn_tail}, {res['secs']:.1f}s)")
    return logs


def run_export(
    *,
    out_dir: Path,
    cache_dir: Path,
    years: Iterable[int],
    sessions: Iterable[str],
    telemetry_stride: int,
    max_workers: int,
    driver_limit: Optional[int],
    skip_existing: bool,
    completed_only: bool,
    latest_only: bool,
    lookback_rounds: Optional[int],
) -> List[Dict[str, object]]:
    _require_fastf1()
    years = sorted(set(int(y) for y in years))
    sessions = [str(s).upper() for s in sessions]
    setup_dirs(out_dir, cache_dir)

    logs: List[Dict[str, object]] = []
    print(f"== Exporting years: {', '.join(map(str, years))} ==")
    for y in years:
        rounds = None
        if completed_only or latest_only or lookback_rounds:
            rounds = completed_rounds_for_year(
                y,
                sessions,
                latest_only=latest_only,
                lookback_rounds=lookback_rounds,
            )
        print(f"\n=== Season {y} ===")
        if rounds is not None:
            if rounds:
                print(f"Planned rounds: {', '.join(map(str, rounds))}")
            else:
                print("No completed rounds match the requested filter.")
                continue
        logs.extend(
            export_season(
                year=y,
                sessions=sessions,
                out_dir=out_dir,
                telemetry_stride=telemetry_stride,
                max_workers=max_workers,
                driver_limit=driver_limit,
                skip_existing=skip_existing,
                rounds=rounds,
            )
        )

    export_schedule(years, out_dir)
    aggregate_weather(out_dir, years, sessions)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fastf1_version": getattr(fastf1, "__version__", ""),
        "years": years,
        "sessions": sessions,
        "completed_only": bool(completed_only),
        "latest_only": bool(latest_only),
        "lookback_rounds": int(lookback_rounds) if lookback_rounds else None,
        "results": logs,
    }
    (out_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return logs


def aggregate_weather(out_dir: Path, years: Iterable[int], sessions: Iterable[str]):
    rows = []
    for y in years:
        rounds = get_max_round(y)
        for rnd in range(1, rounds + 1):
            for ses in sessions:
                suffix = "" if ses == "R" else f"_{ses}"
                p = out_dir / f"weather_{y}_{rnd}{suffix}.csv"
                if not p.exists():
                    continue
                try:
                    w = pd.read_csv(p)
                    rows.append(per_race_weather_summary(y, rnd, ses, w))
                except Exception:
                    pass
    if rows:
        df = pd.DataFrame(rows)
                                                                     
        df["_pref"] = df["session"].apply(lambda s: 0 if s == "R" else 1)
        df = df.sort_values(["raceId", "_pref"]).drop_duplicates("raceId").drop(columns=["_pref","session"])
        safe_to_csv(df, out_dir / "weather.csv")


def export_schedule(years: Iterable[int], out_dir: Path):
    all_rows = []
    for y in years:
        sc = schedule_for_year(y)
        if not sc.empty:
            all_rows.append(sc)
            safe_to_csv(sc, out_dir / f"schedule_{y}.csv")
    if all_rows:
        df = pd.concat(all_rows, ignore_index=True)
        safe_to_csv(df, out_dir / "races.csv")                                                     


                               
     
                               

def main():
    ap = argparse.ArgumentParser(description="Export FastF1 datasets for pre-race features.")
    ap.add_argument("--out-dir", type=Path, default=Path("data/raw_csv"), help="Where to save CSV files.")
    ap.add_argument("--cache-dir", type=Path, default=Path("data/fastf1_cache"), help="FastF1 cache directory.")
    ap.add_argument("--years", nargs="+", type=int, default=None, help="Years to export, e.g. --years 2024 2025. Default: last two years.")
    ap.add_argument("--last-n-seasons", type=int, default=2, help="If --years not set, export this many most-recent seasons.")
    ap.add_argument("--sessions", nargs="+", default=["R", "Q", "S", "SQ", "FP1", "FP2", "FP3"], help="Session codes to export (e.g., R Q FP1 FP2 FP3 S SQ).")
    ap.add_argument("--telemetry-stride", type=int, default=1, help="Downsample telemetry rows (1=no downsample, 5=every 5th row).")
    ap.add_argument("--max-workers", type=int, default=4, help="Parallel workers.")
    ap.add_argument("--driver-limit", type=int, default=None, help="Limit number of drivers per session (debug).")
    ap.add_argument("--skip-existing", action="store_true", help="Skip sessions that already have laps CSV.")
    ap.add_argument("--completed-only", action="store_true", help="Export only rounds whose requested sessions are already completed.")
    ap.add_argument("--latest-only", action="store_true", help="Export only the latest completed round per requested year.")
    ap.add_argument("--lookback-rounds", type=int, default=None, help="Restrict export to the last N completed rounds per year.")
    args = ap.parse_args()

                   
    if args.years:
        years = sorted(set(args.years))
    else:
        from datetime import date
        cur = date.today().year
                                          
        years = list(range(cur - args.last_n_seasons + 1, cur + 1))

    run_export(
        out_dir=args.out_dir,
        cache_dir=args.cache_dir,
        years=years,
        sessions=args.sessions,
        telemetry_stride=args.telemetry_stride,
        max_workers=args.max_workers,
        driver_limit=args.driver_limit,
        skip_existing=args.skip_existing,
        completed_only=args.completed_only or args.latest_only or bool(args.lookback_rounds),
        latest_only=args.latest_only,
        lookback_rounds=args.lookback_rounds,
    )

    print("\n✅ Done. Files saved in", args.out_dir)
    print("   Next: run scripts.build_features with --raw-dir pointing to this folder.")

if __name__ == "__main__":
    main()
