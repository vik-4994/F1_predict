#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export all FastF1 datasets needed for our pre-race features.

What it exports (into --out-dir, default: data/raw_csv):
Per race (year, round) and per requested session (default: R and Q):
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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Iterable, Dict, List, Tuple

import pandas as pd
import numpy as np
import fastf1


# -----------------------------
# IO helpers
# -----------------------------

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


# -----------------------------
# Transform helpers
# -----------------------------

def laps_with_ms(laps: pd.DataFrame) -> pd.DataFrame:
    """Ensure a numeric 'milliseconds' column is present (from LapTime)."""
    df = laps.copy()
    if "milliseconds" not in df.columns:
        if "LapTime" in df.columns:
            # LapTime is a pandas Timedelta; convert to float milliseconds
            ms = pd.to_timedelta(df["LapTime"], errors="coerce").dt.total_seconds() * 1000.0
            df["milliseconds"] = ms.astype(float)
        else:
            # try 'Time' (rare) or leave NaN
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
    # FastF1 Laps have LapNumber, PitInTime, PitOutTime
    if not {"LapNumber"}.issubset(cols):
        return pd.DataFrame(columns=["raceId", "Driver", "DriverNumber", "lap", "duration_ms"])
    df["lap"] = pd.to_numeric(df["LapNumber"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["lap"]).astype({"lap": int})
    # Detect pit entries where PitInTime is not null on this lap
    has_pitin = "PitInTime" in cols
    has_pitout = "PitOutTime" in cols
    if not has_pitin and not has_pitout:
        # Some data only flags 'IsPitOut' on outlap; fallback: mark laps where 'PitOutTime' on this lap exists
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
        info = session.get_driver(drv)  # dict with Abbreviation, DriverNumber, BroadcastName, TeamName, etc.
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
        # keep essential columns
        keep = [c for c in ["EventName", "OfficialEventName", "Location", "Country", "EventDate", "RoundNumber"] if c in sched.columns]
        df = sched[keep].copy().reset_index(drop=True)
        # add synthetic raceId = year*1000 + round
        if "RoundNumber" in df.columns:
            df.insert(0, "raceId", [int(year) * 1000 + int(r) for r in df["RoundNumber"]])
        else:
            df.insert(0, "raceId", np.arange(1, df.shape[0]+1) + int(year) * 1000)
        df.insert(1, "year", int(year))
        df.rename(columns={"RoundNumber": "round"}, inplace=True)
        return df
    except Exception as e:
        return pd.DataFrame()


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


# -----------------------------
# Export logic
# -----------------------------

def export_one_session(year: int, rnd: int, ses: str, out_dir: Path, telemetry_stride: int, driver_limit: Optional[int]) -> Dict[str, object]:
    tag = f"{year} R{rnd:02d} {ses}"
    t0 = time.time()
    try:
        session = fastf1.get_session(year, rnd, ses)
        session.load(laps=True, telemetry=True, weather=True)

        # Base dataframes
        laps = session.laps.reset_index(drop=True)
        laps = _add_race_id(laps, year, rnd)
        laps = laps_with_ms(laps)

        weather = session.weather_data.reset_index(drop=True)
        weather = _add_race_id(weather, year, rnd)

        results = session.results.reset_index(drop=True)
        results = _add_race_id(results, year, rnd)

        # meta
        meta = {
            "raceId": int(year) * 1000 + int(rnd),
            "Year": year, "Round": rnd, "Session": ses,
            "EventName": session.event.get("EventName", ""),
            "Location": session.event.get("Location", ""),
            "Country": session.event.get("Country", ""),
            "EventDate": str(session.event.get("EventDate", "")),
        }
        meta_df = pd.DataFrame([meta])

        # Save core CSVs
        suffix = "" if ses == "R" else f"_{ses}"
        safe_to_csv(laps, out_dir / f"laps_{year}_{rnd}{suffix}.csv")
        safe_to_csv(weather, out_dir / f"weather_{year}_{rnd}{suffix}.csv")
        safe_to_csv(results, out_dir / f"results_{year}_{rnd}{suffix}.csv")
        safe_to_csv(meta_df, out_dir / f"meta_{year}_{rnd}{suffix}.csv")

        # Race-only extras
        if ses == "R":
            try:
                rcm = session.race_control_messages.reset_index(drop=True)
                rcm = _add_race_id(rcm, year, rnd)
                safe_to_csv(rcm, out_dir / f"race_ctrl_{year}_{rnd}.csv")
            except Exception:
                pass
            try:
                ts = session.track_status.reset_index(drop=True)
                ts = _add_race_id(ts, year, rnd)
                safe_to_csv(ts, out_dir / f"track_status_{year}_{rnd}.csv")
            except Exception:
                pass

        # Entry list
        try:
            ent = entrylist_from_session(session)
            safe_to_csv(ent, out_dir / f"entrylist_{year}_{rnd}{suffix}.csv")
        except Exception:
            pass

        # Stints (from laps)
        try:
            st = stints_from_laps(laps)
            if not st.empty:
                safe_to_csv(st, out_dir / f"stints_{year}_{rnd}{suffix}.csv")
        except Exception:
            pass

        # Telemetry + Position per driver
        drivers: List[int] = list(session.drivers)
        if driver_limit is not None:
            drivers = drivers[:driver_limit]

        tel_cnt = 0
        pos_cnt = 0
        for drv in drivers:
            info = session.get_driver(drv)
            abbr = info.get("Abbreviation", str(drv))

            # car telemetry
            try:
                car = session.laps.pick_driver(drv).get_car_data()
                if car is not None and len(car):
                    tdf = car.reset_index(drop=True)
                    tdf = _add_race_id(tdf, year, rnd)
                    if telemetry_stride and telemetry_stride > 1:
                        tdf = tdf.iloc[::telemetry_stride].reset_index(drop=True)
                    safe_to_csv(tdf, out_dir / f"telemetry_{year}_{rnd}{suffix}_{abbr}.csv")
                    tel_cnt += 1
            except Exception:
                pass

            # position XY
            try:
                pos = session.laps.pick_driver(drv).get_pos_data()
                if pos is not None and len(pos):
                    pdf = pos.reset_index(drop=True)
                    pdf = _add_race_id(pdf, year, rnd)
                    safe_to_csv(pdf, out_dir / f"position_{year}_{rnd}{suffix}_{abbr}.csv")
                    pos_cnt += 1
            except Exception:
                pass

        # Derived pit stops (R only — most useful for race strategy features)
        pit_cnt = 0
        if ses == "R":
            try:
                pits = derive_pitstops_from_laps(laps)
                if not pits.empty:
                    safe_to_csv(pits, out_dir / f"pit_stops_{year}_{rnd}.csv")
                    pit_cnt = pits.shape[0]
            except Exception:
                pass

        # Return a compact summary for logging
        return {"race": tag, "status": "ok", "drivers_tel": tel_cnt, "drivers_pos": pos_cnt, "pit_rows": pit_cnt, "secs": time.time() - t0}

    except Exception as e:
        return {"race": tag, "status": f"error: {e}", "drivers_tel": 0, "drivers_pos": 0, "pit_rows": 0, "secs": time.time() - t0}


def get_max_round(year: int) -> int:
    try:
        sched = fastf1.get_event_schedule(year)
        return int(sched["RoundNumber"].max())
    except Exception:
        # reasonable upper bound
        return 25


def export_season(year: int, sessions: Iterable[str], out_dir: Path, telemetry_stride: int, max_workers: int, driver_limit: Optional[int], skip_existing: bool) -> List[Dict[str, object]]:
    rounds = get_max_round(year)
    tasks = []
    logs = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for rnd in range(1, rounds + 1):
            for ses in sessions:
                if skip_existing:
                    # quick check: if laps file exists, skip this session
                    suffix = "" if ses == "R" else f"_{ses}"
                    if (out_dir / f"laps_{year}_{rnd}{suffix}.csv").exists():
                        continue
                tasks.append(ex.submit(export_one_session, year, rnd, ses, out_dir, telemetry_stride, driver_limit))
        for fut in as_completed(tasks):
            res = fut.result()
            logs.append(res)
            print(f"• {res['race']}: {res['status']} (tel:{res['drivers_tel']}, pos:{res['drivers_pos']}, pit:{res['pit_rows']}, {res['secs']:.1f}s)")
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
        # keep one row per raceId preferring Race over other sessions
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
        safe_to_csv(df, out_dir / "races.csv")  # lightweight schedule (year, round, raceId, names)


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Export FastF1 datasets for pre-race features.")
    ap.add_argument("--out-dir", type=Path, default=Path("data/raw_csv"), help="Where to save CSV files.")
    ap.add_argument("--cache-dir", type=Path, default=Path("data/fastf1_cache"), help="FastF1 cache directory.")
    ap.add_argument("--years", nargs="+", type=int, default=None, help="Years to export, e.g. --years 2024 2025. Default: last two years.")
    ap.add_argument("--last-n-seasons", type=int, default=2, help="If --years not set, export this many most-recent seasons.")
    ap.add_argument("--sessions", nargs="+", default=["R", "Q"], help="Session codes to export (e.g., R Q FP2 S SS).")
    ap.add_argument("--telemetry-stride", type=int, default=1, help="Downsample telemetry rows (1=no downsample, 5=every 5th row).")
    ap.add_argument("--max-workers", type=int, default=4, help="Parallel workers.")
    ap.add_argument("--driver-limit", type=int, default=None, help="Limit number of drivers per session (debug).")
    ap.add_argument("--skip-existing", action="store_true", help="Skip sessions that already have laps CSV.")
    args = ap.parse_args()

    # Resolve years
    if args.years:
        years = sorted(set(args.years))
    else:
        from datetime import date
        cur = date.today().year
        # last N seasons including current
        years = list(range(cur - args.last_n_seasons + 1, cur + 1))

    setup_dirs(args.out_dir, args.cache_dir)

    print(f"== Exporting years: {', '.join(map(str, years))} ==")
    for y in years:
        print(f"\n=== Season {y} ===")
        export_season(
            year=y,
            sessions=args.sessions,
            out_dir=args.out_dir,
            telemetry_stride=args.telemetry_stride,
            max_workers=args.max_workers,
            driver_limit=args.driver_limit,
            skip_existing=args.skip_existing,
        )

    # Aggregates
    export_schedule(years, args.out_dir)
    aggregate_weather(args.out_dir, years, args.sessions)

    print("\n✅ Done. Files saved in", args.out_dir)
    print("   Next: run scripts.build_features with --raw-dir pointing to this folder.")

if __name__ == "__main__":
    main()
