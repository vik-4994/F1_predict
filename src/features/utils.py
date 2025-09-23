"""
Utility functions shared by feature modules (LEAK‑SAFE rev).

Focus:
- robust CSV/timedelta handling
- stint detection
- clean‑laps & best10 pace
- tyre compound usage
- lap bounds + telemetry↔lap alignment (single or multi‑driver)
- telemetry aggregation per‑lap and per‑race
- race control / track status windows → per‑lap shares
- path helpers & tolerant filename parsing

All functions are pure and side‑effect free.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Literal

import numpy as np
import pandas as pd
import re

# Public API
__all__ = [
    # I/O & conversions
    "to_td",
    "read_csv_if_exists",
    "read_csv_header",
    # path helpers
    "laps_path", "weather_path", "results_path", "telemetry_paths",
    "race_control_path", "track_status_path", "parse_year_round",
    # laps enrichment
    "load_laps_enriched",
    "compute_stints",
    "best10_and_clean",
    "compound_counts",
    # field helpers
    "ensure_driver_index",
    # bounds + telemetry alignment
    "build_lap_bounds",
    "attach_lapnumber_to_telem",
    "attach_for_all_drivers",
    # telemetry aggregation
    "telem_per_lap_agg",
    "telem_per_race_agg",
    # race control / track status helpers
    "load_race_control", "load_track_status",
    "build_status_windows", "lap_shares_from_windows",
    "get_outlaps", "tag_events_in_windows",
    "positions_by_lap",
]

# =========================
# I/O & Conversions
# =========================

def to_td(x) -> pd.Timedelta:
    """Parse object to pandas Timedelta (NaT on errors)."""
    return pd.to_timedelta(x, errors="coerce")


def read_csv_if_exists(p: Path, **kwargs) -> pd.DataFrame:
    """Return DataFrame if CSV exists, else empty DataFrame."""
    return pd.read_csv(p, **kwargs) if p.exists() else pd.DataFrame()


def read_csv_header(p: Path) -> List[str]:
    """Return CSV header (column names) without loading all rows."""
    if not p.exists():
        return []
    return list(pd.read_csv(p, nrows=0).columns)

# =========================
# Path helpers
# =========================

def laps_path(raw_dir: Path, year: int, rnd: int) -> Path:
    return Path(raw_dir) / f"laps_{year}_{rnd}.csv"


def weather_path(raw_dir: Path, year: int, rnd: int) -> Path:
    return Path(raw_dir) / f"weather_{year}_{rnd}.csv"


def results_path(raw_dir: Path, year: int, rnd: int) -> Path:
    return Path(raw_dir) / f"results_{year}_{rnd}.csv"


def race_control_path(raw_dir: Path, year: int, rnd: int) -> Path:
    return Path(raw_dir) / f"race_ctrl_{year}_{rnd}.csv"


def track_status_path(raw_dir: Path, year: int, rnd: int) -> Path:
    return Path(raw_dir) / f"track_status_{year}_{rnd}.csv"


def telemetry_paths(raw_dir: Path, year: int, rnd: int) -> List[Path]:
    """All per‑driver telemetry CSVs (telemetry_YYYY_R_ABBR.csv) if present."""
    return sorted(Path(raw_dir).glob(f"telemetry_{year}_{rnd}_*.csv"))


def parse_year_round(name: str) -> Tuple[int, int]:
    """Parse (year, round) from file name.

    Supports:
      - laps_2024_1.csv
      - weather_2025_7.csv
      - results_2024_22.csv
      - meta_2025_3_Q.csv, telemetry_2024_1_ALB.csv (suffix after round is ok)
    """
    m = re.search(r"_(\d{4})_(\d{1,2})(?:_[A-Za-z0-9]+)?\.csv$", name)
    if not m:
        raise ValueError(f"Cannot parse year/round from: {name}")
    return int(m.group(1)), int(m.group(2))

# =========================
# Laps enrichment
# =========================

def load_laps_enriched(raw_dir: Path, year: int, rnd: int) -> pd.DataFrame:
    """Load laps_YYYY_R.csv and enrich with:
      - *_s seconds columns for lap/sector times
      - pit flag is_pit
      - sorted by (Driver, LapNumber)
      - ffilled Compound per driver and computed stint_id
    Returns empty DataFrame if file missing.
    """
    p = laps_path(raw_dir, year, rnd)
    if not p.exists():
        return pd.DataFrame()

    df = pd.read_csv(p)

    # normalize driver key
    if "Driver" in df.columns:
        df["Driver"] = df["Driver"].astype(str)
    elif "Abbreviation" in df.columns:
        df["Driver"] = df["Abbreviation"].astype(str)
    elif "DriverNumber" in df.columns:
        df["Driver"] = df["DriverNumber"].astype(str)

    # Timedeltas
    td_cols = [
        "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
        "PitInTime", "PitOutTime", "LapStartTime", "SessionTime", "Time",
    ]
    for c in td_cols:
        if c in df.columns:
            df[c] = to_td(df[c])

    # Seconds versions
    for src, dst in [
        ("LapTime", "LapTime_s"),
        ("Sector1Time", "S1_s"),
        ("Sector2Time", "S2_s"),
        ("Sector3Time", "S3_s"),
    ]:
        if src in df.columns:
            df[dst] = df[src].dt.total_seconds()

    # Pit flags
    df["is_pit"] = False
    if "PitInTime" in df.columns:
        df["is_pit"] |= df["PitInTime"].notna()
    if "PitOutTime" in df.columns:
        df["is_pit"] |= df["PitOutTime"].notna()

    # Order for stint detection
    if {"Driver", "LapNumber"}.issubset(df.columns):
        df = df.sort_values(["Driver", "LapNumber"], kind="mergesort").reset_index(drop=True)

    # Stints by compound
    df = compute_stints(df)
    return df


def compute_stints(df: pd.DataFrame) -> pd.DataFrame:
    """Compute stints per driver using compound changes.
    - forward‑fill Compound within driver
    - new stint when Compound changes
    - produce integer 'stint_id' starting from 1 per driver
    """
    out = df.copy()
    if "Driver" not in out.columns:
        out["Driver"] = out.get("Abbreviation", out.get("DriverNumber", pd.Series([], dtype=str))).astype(str)

    if "Compound" in out.columns:
        out["Compound"] = out.groupby("Driver", group_keys=False)["Compound"].ffill()
    else:
        out["Compound"] = np.nan

    out["stint_id"] = (
        out.groupby("Driver", group_keys=False)["Compound"].apply(lambda s: (s != s.shift()).cumsum())
    )
    return out


def best10_and_clean(no_pit: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per‑driver:
      - best10_pace_s: mean of 10 fastest clean laps
      - clean_laps_share: share of laps within (<= Q3 + 2*IQR)
    Return (best10_df, clean_df) with 'Driver' key.
    """
    rows_b10, rows_clean = [], []
    if no_pit.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = no_pit.copy()
    if "LapTime_s" in df.columns:
        df["LapTime_s"] = pd.to_numeric(df["LapTime_s"], errors="coerce")

    for drv, sub in df.groupby("Driver"):
        s = sub["LapTime_s"].dropna()
        if s.empty:
            continue
        if len(s) >= 3:
            q1, q3 = s.quantile([0.25, 0.75])
            thr = float(q3 + 2 * (q3 - q1))
            clean_mask = s <= thr
            rows_clean.append({"Driver": drv, "clean_laps_share": float(clean_mask.mean())})
            clean = s[clean_mask]
        else:
            clean = s
        rows_b10.append({"Driver": drv, "best10_pace_s": float(clean.nsmallest(10).mean())})

    return pd.DataFrame(rows_b10), pd.DataFrame(rows_clean)


def compound_counts(laps: pd.DataFrame) -> pd.DataFrame:
    """Pivot counts of laps per Compound → columns: tyre_<COMPOUND>_laps."""
    if "Compound" not in laps.columns or laps.empty:
        return pd.DataFrame()
    comp = laps.pivot_table(index="Driver", columns="Compound", values="LapNumber", aggfunc="count").fillna(0)
    comp.columns = [f"tyre_{c}_laps" for c in comp.columns]
    return comp.reset_index()

# =========================
# Field helpers
# =========================

def ensure_driver_index(drivers: pd.Series | Iterable[str], values: Dict[str, float]) -> pd.DataFrame:
    """Broadcast feature dict to a DF with one row per Driver."""
    if isinstance(drivers, pd.Series):
        drivers = drivers.dropna().astype(str).unique().tolist()
    rows = [{"Driver": d, **values} for d in drivers]
    return pd.DataFrame(rows)

# =========================
# Bounds & telemetry alignment
# =========================

def build_lap_bounds(laps: pd.DataFrame) -> pd.DataFrame:
    """Create per‑lap time bounds [start, end) for telemetry alignment.
    Requires columns: Driver, LapNumber, LapStartTime, LapTime (Timedelta).
    Returns: [Driver, LapNumber, start, end] sorted by (Driver, start).
    """
    need = {"Driver", "LapNumber", "LapStartTime", "LapTime"}
    if not need.issubset(laps.columns):
        missing = sorted(list(need - set(laps.columns)))
        raise RuntimeError(f"Missing columns in laps for bounds: {missing}")

    b = laps[["Driver", "LapNumber", "LapStartTime", "LapTime"]].dropna(subset=["LapStartTime", "LapTime"]).copy()
    b["start"] = b["LapStartTime"]
    b["end"] = b["LapStartTime"] + b["LapTime"]
    b = b.drop(columns=["LapStartTime", "LapTime"])
    return b.sort_values(["Driver", "start"]).reset_index(drop=True)


def attach_lapnumber_to_telem(
    telem: pd.DataFrame,
    bounds: pd.DataFrame,
    driver: Optional[str] = None,
    time_col: str = "Time",
) -> pd.DataFrame:
    """Assign LapNumber to telemetry rows via merge_asof against lap bounds.
    telem[time_col] must be Timedelta; bounds from build_lap_bounds().
    If 'driver' is provided, bounds are filtered to that driver.
    If 'driver' is None and both frames have 'Driver', merge_asof uses 'by="Driver"'.
    Returns telemetry with LapNumber (rows outside bounds dropped).
    """
    if telem.empty or time_col not in telem.columns:
        return pd.DataFrame()

    out = telem.copy()
    out[time_col] = to_td(out[time_col])
    out = out.sort_values(time_col).reset_index(drop=True)

    b = bounds.copy()
    if driver is not None and "Driver" in b.columns:
        b = b[b["Driver"] == str(driver)]

    if b.empty:
        return pd.DataFrame()

    b = b.sort_values("start").reset_index(drop=True)

    # choose merge key
    right_cols = ["start", "end", "LapNumber"]
    by = None
    if driver is None and "Driver" in out.columns and "Driver" in b.columns:
        by = "Driver"
        right_cols = ["Driver", "start", "end", "LapNumber"]

    aligned = pd.merge_asof(
        out,
        b[right_cols],
        left_on=time_col,
        right_on="start",
        direction="backward",
        allow_exact_matches=True,
        by=by,
    )
    # keep only points within the lap window
    aligned = aligned[aligned[time_col] <= aligned["end"]]
    return aligned.drop(columns=["start", "end"]).reset_index(drop=True)


def attach_for_all_drivers(telem: pd.DataFrame, bounds: pd.DataFrame, time_col="Time") -> pd.DataFrame:
    """Helper: attach LapNumber for multi‑driver telemetry by looping per driver."""
    if telem.empty or "Driver" not in telem.columns:
        return pd.DataFrame()
    parts = []
    for d, sub in telem.groupby("Driver"):
        parts.append(attach_lapnumber_to_telem(sub, bounds, driver=str(d), time_col=time_col))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

# =========================
# Telemetry aggregation
# =========================

def telem_per_lap_agg(
    telem_labeled: pd.DataFrame,
    lap_col: str = "LapNumber",
    time_col: str = "Time",
) -> pd.DataFrame:
    """Aggregate labeled telemetry into per‑lap features.
    Expected columns if available: Speed, Throttle, Brake, nGear, RPM, DRS, Time.
    Returns DF with one row per lap and columns like:
      - Speed_mean/max/std
      - Throttle_mean/std, Throttle_p90_share
      - Brake_mean/std, Brake_active_share
      - nGear_mean, RPM_mean, RPM_p95
      - DRS_share
      - accel_p95, decel_p95 (from diff(Speed)/diff(Time))
      - shift_rate_hz (gear changes per second)
    """
    if telem_labeled.empty or lap_col not in telem_labeled.columns:
        return pd.DataFrame()

    df = telem_labeled.copy()
    # Ensure time is timedelta for rate metrics
    if time_col in df.columns:
        df[time_col] = to_td(df[time_col])

    cols = [c for c in ["Speed", "Throttle", "Brake", "nGear", "RPM", "DRS"] if c in df.columns]
    g = df.groupby(lap_col, dropna=True)

    # Basic stats
    agg = pd.DataFrame(index=g.size().index)
    for c in cols:
        agg[f"{c}_mean"] = g[c].mean()
        agg[f"{c}_max"] = g[c].max()
        agg[f"{c}_std"] = g[c].std()

    # Shares
    if "Throttle" in df.columns:
        agg["Throttle_p90_share"] = g["Throttle"].apply(lambda s: float((s >= 90).mean()))
    if "Brake" in df.columns:
        agg["Brake_active_share"] = g["Brake"].apply(lambda s: float((s > 0).mean()))
    if "DRS" in df.columns:
        agg["DRS_share"] = g["DRS"].apply(lambda s: float((s > 0).mean()))

    # RPM p95
    if "RPM" in df.columns:
        agg["RPM_p95"] = g["RPM"].quantile(0.95)

    # Accel/decel (approx) and shift rate
    if "Speed" in df.columns and time_col in df.columns:
        def _accels(sub: pd.DataFrame) -> Tuple[float, float]:
            s = sub.sort_values(time_col)
            ds = s["Speed"].diff()
            dt = s[time_col].diff().dt.total_seconds()
            with np.errstate(divide="ignore", invalid="ignore"):
                a = ds / dt
            a = a.replace([np.inf, -np.inf], np.nan).dropna()
            if a.empty:
                return np.nan, np.nan
            pos = a[a > 0]
            neg = -a[a < 0]
            return (
                float(np.nanpercentile(pos, 95)) if not pos.empty else np.nan,
                float(np.nanpercentile(neg, 95)) if not neg.empty else np.nan,
            )

        acc = g.apply(_accels)
        agg["accel_p95"] = [t[0] if isinstance(t, tuple) else np.nan for t in acc]
        agg["decel_p95"] = [t[1] if isinstance(t, tuple) else np.nan for t in acc]

    if "nGear" in df.columns and time_col in df.columns:
        def _shift_rate(sub: pd.DataFrame) -> float:
            s = sub.sort_values(time_col)
            gear = s["nGear"].astype("float").dropna()
            if gear.empty:
                return np.nan
            shifts = (gear.diff() != 0).sum()
            dur = (s[time_col].iloc[-1] - s[time_col].iloc[0]).total_seconds() if len(s) > 1 else np.nan
            return float(shifts / dur) if dur and dur > 0 else np.nan

        agg["shift_rate_hz"] = g.apply(_shift_rate)

    agg = agg.reset_index()  # brings back LapNumber
    return agg


def telem_per_race_agg(
    telem_labeled: pd.DataFrame,
    lap_col: str = "LapNumber",
    time_col: str = "Time",
) -> pd.DataFrame:
    """Aggregate labeled telemetry for the whole race (single row, no Driver column)."""
    per_lap = telem_per_lap_agg(telem_labeled, lap_col=lap_col, time_col=time_col)
    if per_lap.empty:
        return pd.DataFrame()
    # take mean of numeric columns across laps
    num = per_lap.select_dtypes(include=[np.number])
    if num.empty:
        return pd.DataFrame()
    row = num.mean(numeric_only=True).to_frame().T
    return row

# ---------- loaders ----------

def load_race_control(raw_dir: Path, year: int, rnd: int) -> pd.DataFrame:
    p = race_control_path(raw_dir, year, rnd)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    # normalize basic fields
    for col in ["Utc"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    for col in ["Lap"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ["Category", "Message", "Status", "Flag", "Scope"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def load_track_status(raw_dir: Path, year: int, rnd: int) -> pd.DataFrame:
    p = track_status_path(raw_dir, year, rnd)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    # expected columns: Time (session Timedelta), Status (str code), Message (str)
    if "Time" in df.columns:
        df["Time"] = to_td(df["Time"])
    if "Status" in df.columns:
        df["Status"] = df["Status"].astype(str)
    if "Message" in df.columns:
        df["Message"] = df["Message"].astype(str)
    if "Time" in df.columns:
        df = df.sort_values("Time").reset_index(drop=True)
    return df

# ---------- status windows ----------
# FastF1 Track Status codes:
# 1: Clear, 2: Yellow, 4: SC, 5: Red, 6: VSC deployed, 7: VSC ending


def _mk_windows_from_status(
    ts: pd.DataFrame,
    code_start: str,
    end_when_codes: List[str],
) -> List[Tuple[pd.Timedelta, pd.Timedelta]]:
    """Stitch intervals: start on Status==code_start, end on first Status in end_when_codes."""
    if ts.empty or "Time" not in ts.columns or "Status" not in ts.columns:
        return []
    wins: List[Tuple[pd.Timedelta, pd.Timedelta]] = []
    open_t0: Optional[pd.Timedelta] = None
    for _, r in ts.iterrows():
        s = str(r["Status"])
        t = r["Time"]
        if s == code_start and open_t0 is None:
            open_t0 = t
        elif open_t0 is not None and s in end_when_codes:
            wins.append((open_t0, t))
            open_t0 = None
    if open_t0 is not None:  # close at last known time
        t_end = ts["Time"].dropna().max()
        if pd.notna(t_end) and t_end > open_t0:
            wins.append((open_t0, t_end))
    return wins


def build_status_windows(ts: pd.DataFrame) -> pd.DataFrame:
    """Return table of status windows: kind in {'SC','VSC','YELLOW'}, with t0, t1 columns."""
    if ts.empty:
        return pd.DataFrame(columns=["kind", "t0", "t1"])

    sc = _mk_windows_from_status(ts, "4", end_when_codes=["1", "2", "5", "6", "7"])
    vsc = _mk_windows_from_status(ts, "6", end_when_codes=["1"])  # '7' is transitional
    yel = _mk_windows_from_status(ts, "2", end_when_codes=["1", "4", "5", "6", "7"])

    rows = []
    rows += [{"kind": "SC", "t0": a, "t1": b} for (a, b) in sc]
    rows += [{"kind": "VSC", "t0": a, "t1": b} for (a, b) in vsc]
    rows += [{"kind": "YELLOW", "t0": a, "t1": b} for (a, b) in yel]
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["t0", "t1"]).reset_index(drop=True)
    return out

# ---------- overlap onto laps ----------

def _overlap_seconds(
    a0: pd.Timedelta, a1: pd.Timedelta,
    b0: pd.Timedelta, b1: pd.Timedelta,
) -> float:
    """Seconds of intersection of [a0,a1] and [b0,b1]."""
    start = max(a0, b0)
    end = min(a1, b1)
    if pd.isna(start) or pd.isna(end) or end <= start:
        return 0.0
    return float((end - start).total_seconds())


def lap_shares_from_windows(lap_bounds: pd.DataFrame, wins: pd.DataFrame) -> pd.DataFrame:
    """Map status windows onto laps to compute per‑lap shares.
    Input:
      lap_bounds with [Driver, LapNumber, start, end]
      wins with [kind, t0, t1]
    Output:
      [Driver, LapNumber, sc_share, vsc_share, yellow_share]
    """
    if lap_bounds.empty:
        return pd.DataFrame(columns=["Driver", "LapNumber", "sc_share", "vsc_share", "yellow_share"])
    if wins.empty:
        base = lap_bounds[["Driver", "LapNumber"]].copy()
        return base.assign(sc_share=np.nan, vsc_share=np.nan, yellow_share=np.nan)

    out_rows: List[Dict] = []
    for (drv, ln), lb in lap_bounds.groupby(["Driver", "LapNumber"], sort=False):
        start = lb["start"].iloc[0]
        end = lb["end"].iloc[0]
        lap_len = float((end - start).total_seconds()) if pd.notna(start) and pd.notna(end) else np.nan
        sc = vsc = yel = 0.0
        if lap_len and lap_len > 0:
            for _, w in wins.iterrows():
                ov = _overlap_seconds(start, end, w["t0"], w["t1"])
                if ov <= 0:
                    continue
                share = ov / lap_len
                k = w["kind"]
                if k == "SC":
                    sc += share
                elif k == "VSC":
                    vsc += share
                elif k == "YELLOW":
                    yel += share
        out_rows.append({"Driver": drv, "LapNumber": int(ln),
                         "sc_share": sc, "vsc_share": vsc, "yellow_share": yel})
    return pd.DataFrame(out_rows)

# ---------- pit stops in windows ----------

def get_outlaps(laps: pd.DataFrame) -> pd.DataFrame:
    """Return only outlaps (Lap with non‑null PitOutTime).
    Columns: Driver, LapNumber, LapTime_s, PitOutTime, LapStartTime
    """
    if laps.empty:
        return pd.DataFrame(columns=["Driver", "LapNumber", "LapTime_s", "PitOutTime", "LapStartTime"])
    out = laps[laps["PitOutTime"].notna()].copy()
    out["LapTime_s"] = pd.to_numeric(out.get("LapTime_s"), errors="coerce")
    return out[["Driver", "LapNumber", "LapTime_s", "PitOutTime", "LapStartTime"]]


def tag_events_in_windows(
    times: pd.Series,
    wins: pd.DataFrame,
    prefer_col: Literal["PitOutTime", "LapStartTime"] = "PitOutTime",
) -> pd.Series:
    """Tag each timestamp with a window kind {'SC','VSC','YELLOW', None}."""
    if wins.empty or times.empty:
        return pd.Series([None] * len(times), index=times.index)
    tags = []
    for t in times:
        if pd.isna(t):
            tags.append(None)
            continue
        tag = None
        for _, w in wins.iterrows():
            if w["t0"] <= t <= w["t1"]:
                tag = w["kind"]
                break
        tags.append(tag)
    return pd.Series(tags, index=times.index)

# ---------- positions per lap ----------

def positions_by_lap(laps: pd.DataFrame) -> pd.DataFrame:
    """Per‑lap positions (end‑of‑lap) from session.laps.
    Returns columns Driver, LapNumber, Position (Int64).
    """
    if laps.empty:
        return pd.DataFrame(columns=["Driver", "LapNumber", "Position"])
    pos = laps[["Driver", "LapNumber", "Position"]].dropna()
    pos["LapNumber"] = pd.to_numeric(pos["LapNumber"], errors="coerce").astype("Int64")
    pos["Position"] = pd.to_numeric(pos["Position"], errors="coerce").astype("Int64")
    return pos
