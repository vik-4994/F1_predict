from __future__ import annotations
"""
Telemetry history priors (LEAK-SAFE, pre-race)

Что делает:
- Формирует прайоры по телеметрии до гонки: по прошлым этапам ((y<year) или (y==year & r<rnd)).
- Поддерживает aggregate/per-driver telemetry и fallback на laps.
- Никаких утечек из таргет-гонки (ростер берём из entrylist/meta/qual).
- ТОЛЬКО featurize(ctx) или старая сигнатура; если races_df нет — строим календарь по файлам.

Вывод (по Driver):
    tele_pre_hist_n
    tele_pre_bestlap_p50_s
    tele_pre_bestlap_iqr_s
    tele_pre_pace_z_mean
    tele_pre_pace_z_trend
    tele_pre_topspeed_p75
    tele_pre_same_track_n
    tele_pre_last_seen_year
    tele_pre_last_seen_round
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import re
import numpy as np
import pandas as pd
import math

# ---------------------------- utils & constants ----------------------------

NUMERIC_TIME_COLS = (
    "LapTime_s", "LapTimeSec", "LapTimeSeconds", "LapTime",
    "BestLapTime_s", "BestLapTime",
    "lap_time", "laptime", "LapTimeMs"
)

TOPSPEED_COLS = (
    "TopSpeed", "TopSpeedKph", "TopSpeedKMH", "TopSpeed_kph", "SpeedTrapKph", "SpeedTrapKMH"
)

DRIVER_COL_CANDS = ("Driver", "Abbreviation", "code", "driverRef", "DriverRef", "DriverCode")

ROSTER_FILES_ORDER = (
    "entrylist_{y}_{r}_Q.csv",
    "entrylist_{y}_{r}.csv",
    "results_{y}_{r}_Q.csv",  # fallback only
)

META_FILES_ORDER = (
    "meta_{y}_{r}.csv",
    "meta_{y}_{r}_Q.csv",
)

TIME_COL_CANDS = ("Time", "SessionTime", "Date", "Timestamp")


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _to_seconds(series: pd.Series) -> pd.Series:
    """Best-effort convert lap time-like values to seconds (float)."""
    s = series.copy()
    # If timedelta-like strings, let pandas parse
    try:
        td = pd.to_timedelta(s, errors="coerce")
        if td.notna().any():
            sec = td.dt.total_seconds()
            if sec.notna().mean() > 0.5:
                return sec
    except Exception:
        pass
    # Heuristic: if looks like ms, divide
    s = pd.to_numeric(s, errors="coerce")
    med = s.replace([np.inf, -np.inf], np.nan).median()
    if pd.notna(med) and med > 1e3:
        return s / 1000.0
    return s


def _event_slug_from_meta(raw_dir: Path, year: int, rnd: int) -> str:
    for pat in META_FILES_ORDER:
        df = _read_csv(raw_dir / pat.format(y=year, r=rnd))
        if df.empty:
            continue
        for c in ("EventName", "Event", "GrandPrix", "Circuit", "CircuitShortName", "Name"):
            if c in df.columns and df[c].notna().any():
                v = str(df[c].iloc[0]).strip()
                if v:
                    return re.sub(r"\s+", "_", v.lower())
    return f"{year}_{rnd}"


def _detect_driver_col(df: pd.DataFrame) -> Optional[str]:
    for c in DRIVER_COL_CANDS:
        if c in df.columns:
            return c
    low = {c.lower(): c for c in df.columns}
    for c in DRIVER_COL_CANDS:
        if c.lower() in low:
            return low[c.lower()]
    return None


def _ensure_driver(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    dcol = _detect_driver_col(df)
    if dcol and dcol != "Driver":
        df = df.rename(columns={dcol: "Driver"})
    if "Driver" not in df.columns:
        df["Driver"] = np.nan
    df["Driver"] = df["Driver"].astype(str)
    return df


def _read_first(paths: Sequence[Path]) -> pd.DataFrame:
    for p in paths:
        df = _read_csv(p)
        if not df.empty:
            return df
    return pd.DataFrame()


# ---------- calendar helpers (works even without races_df) ----------
def _scan_available_rounds(raw_dir: Path) -> List[Tuple[int, int]]:
    rr = set()
    for pat in ("results_*.csv", "laps_*.csv", "entrylist_*.csv", "results_*_Q.csv", "entrylist_*_Q.csv",
                "telemetry_*.csv", "telemetry_agg_*.csv", "meta_*.csv", "meta_*_Q.csv"):
        for p in raw_dir.glob(pat):
            m = re.search(r"(\d{4})[_-](\d{1,2})", p.stem)
            if m:
                rr.add((int(m.group(1)), int(m.group(2))))
    return sorted(rr)


def _calendar_df_from_files(raw_dir: Path) -> pd.DataFrame:
    pairs = _scan_available_rounds(raw_dir)
    if not pairs:
        return pd.DataFrame(columns=["year", "round"])
    df = pd.DataFrame(pairs, columns=["year", "round"])
    return df.sort_values(["year", "round"]).reset_index(drop=True)


def _list_past_races_by_df(races_df: pd.DataFrame, year: int, rnd: int, max_lookback: int) -> List[Tuple[int, int]]:
    if races_df is None or races_df.empty:
        return []
    yr_col = next((c for c in races_df.columns if c.lower() in ("year", "season")), None)
    rd_col = next((c for c in races_df.columns if c.lower() == "round"), None)
    if yr_col is None or rd_col is None:
        return []
    df = races_df[[yr_col, rd_col]].copy()
    df.columns = ["year", "round"]
    past = df[(df["year"] < year) | ((df["year"] == year) & (df["round"] < rnd))]
    past = past.sort_values(["year", "round"], ascending=[False, False]).head(max_lookback)
    return list(map(tuple, past.values.tolist()))


def _list_past_races_by_scan(raw_dir: Path, year: int, rnd: int, max_lookback: int) -> List[Tuple[int, int]]:
    all_rr = _scan_available_rounds(raw_dir)
    past = [(y, r) for (y, r) in all_rr if (y < year) or (y == year and r < rnd)]
    past.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return past[:max_lookback]


# ---------- roster ----------
def _load_roster(raw_dir: Path, year: int, rnd: int, races_df: Optional[pd.DataFrame]) -> List[str]:
    # Prefer target entrylist (safe)
    for pat in ROSTER_FILES_ORDER:
        df = _read_csv(raw_dir / pat.format(y=year, r=rnd))
        if not df.empty:
            dcol = _detect_driver_col(df)
            if dcol:
                vals = (df[dcol].astype(str).dropna().str.strip()
                        .replace({"nan": np.nan}).dropna().drop_duplicates().tolist())
                if vals:
                    return vals
    # Fallback: previous rounds (calendar or scan)
    prev = _list_past_races_by_df(races_df, year, rnd, max_lookback=5) if races_df is not None else \
           _list_past_races_by_scan(raw_dir, year, rnd, max_lookback=5)
    for (y, r) in prev:
        for pat in ROSTER_FILES_ORDER:
            df = _read_csv(raw_dir / pat.format(y=y, r=r))
            if not df.empty:
                dcol = _detect_driver_col(df)
                if dcol:
                    vals = (df[dcol].astype(str).dropna().str.strip()
                            .replace({"nan": np.nan}).dropna().drop_duplicates().tolist())
                    if vals:
                        return vals
    return []


# ---------- telemetry per race ----------
def _load_telemetry_for_race(raw_dir: Path, y: int, r: int) -> pd.DataFrame:
    """Return per-driver telemetry summary for a past race (best lap seconds, top speed)."""
    # 1) aggregate telemetry
    agg = _read_first([
        raw_dir / f"telemetry_agg_{y}_{r}.csv",
        raw_dir / f"telemetry_{y}_{r}.csv",
        raw_dir / f"telemetry_laps_{y}_{r}.csv",
    ])
    agg = _ensure_driver(agg)
    if not agg.empty:
        out = agg[[c for c in agg.columns if c in ("Driver",) + NUMERIC_TIME_COLS + TOPSPEED_COLS]].copy()
        # best lap time column
        tcol = next((c for c in NUMERIC_TIME_COLS if c in out.columns), None)
        out["best_lap_s"] = _to_seconds(out[tcol]) if tcol else np.nan
        # top speed
        tscol = next((c for c in TOPSPEED_COLS if c in out.columns), None)
        out["top_speed_kph"] = pd.to_numeric(out.get(tscol, np.nan), errors="coerce")
        return out[["Driver", "best_lap_s", "top_speed_kph"]]

    # 2) per-driver telemetry files: telemetry_{y}_{r}_*.csv
    rows = []
    for p in raw_dir.glob(f"telemetry_{y}_{r}_*.csv"):
        t = _ensure_driver(_read_csv(p))
        if t.empty:
            continue
        d = str(t["Driver"].iloc[0]) if t["Driver"].notna().any() else p.stem.split("_")[-1]
        tcol = next((c for c in NUMERIC_TIME_COLS if c in t.columns), None)
        blap = float(_to_seconds(t[tcol]).min()) if tcol else np.nan
        tscol = next((c for c in TOPSPEED_COLS if c in t.columns), None)
        ts = float(pd.to_numeric(t[tscol], errors="coerce").max()) if tscol else np.nan
        rows.append({"Driver": d, "best_lap_s": blap, "top_speed_kph": ts})
    if rows:
        return pd.DataFrame(rows)

    # 3) laps fallback
    laps = _read_first([
        raw_dir / f"laps_{y}_{r}.csv",
        raw_dir / f"laps_{y}_{r}_R.csv",
        raw_dir / f"laps_{y}_{r}_race.csv",
    ])
    laps = _ensure_driver(laps)
    if not laps.empty:
        for flag_col in ("IsPitIn", "IsPitOut", "PitIn", "PitOut", "InPit", "OutPit"):
            if flag_col not in laps.columns:
                laps[flag_col] = False
        clean = laps[(~laps["IsPitIn"]) & (~laps["IsPitOut"]) & (~laps["PitIn"]) & (~laps["PitOut"])].copy()
        tcol = next((c for c in NUMERIC_TIME_COLS if c in clean.columns), None)
        if tcol is None:
            tcol = next((c for c in ("BestLapTime", "BestLapTime_s") if c in clean.columns), None)
        if tcol is None:
            return pd.DataFrame()
        clean["_sec"] = _to_seconds(clean[tcol])
        g = clean.groupby("Driver", dropna=False)["_sec"].min().reset_index()
        g = g.rename(columns={"_sec": "best_lap_s"})
        g["top_speed_kph"] = np.nan
        return g

    return pd.DataFrame()


def _zscore_inv(x: pd.Series) -> pd.Series:
    """Z-score where LOWER raw value is BETTER: higher z is better."""
    x = pd.to_numeric(x, errors="coerce")
    m = x.mean()
    s = x.std(ddof=0)
    if not np.isfinite(s) or s == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return -(x - m) / s


def _linreg_slope(y: Sequence[float]) -> float:
    """Slope of simple linear regression of y over 0..n-1."""
    arr = np.asarray(pd.to_numeric(pd.Series(y), errors="coerce").dropna())
    n = arr.size
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x -= x.mean()
    y = arr - arr.mean()
    denom = float((x**2).sum())
    if denom == 0:
        return 0.0
    return float((x * y).sum() / denom)


@dataclass
class FeaturizeOptions:
    max_lookback: int = 10
    min_hist: int = 1
    same_track_bonus: float = 0.50
    require_same_track_if_short: int = 0


def _featurize_impl(
    raw_dir: Path | str,
    races_df: Optional[pd.DataFrame],
    year: int,
    rnd: int,
    roster: Optional[Sequence[str]] = None,
    options: Optional[FeaturizeOptions] = None,
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    opt = options or FeaturizeOptions()

    # 1) Roster (pre-race safe)
    drivers = list(roster) if roster else _load_roster(raw_dir, year, rnd, races_df)

    # 2) Past races
    past = _list_past_races_by_df(races_df, year, rnd, max_lookback=opt.max_lookback) if races_df is not None else \
           _list_past_races_by_scan(raw_dir, year, rnd, max_lookback=opt.max_lookback)

    if not past:
        base = pd.DataFrame({"Driver": drivers}) if drivers else pd.DataFrame()
        if base.empty:
            return base
        return _attach_defaults(base)

    target_slug = _event_slug_from_meta(raw_dir, year, rnd)

    # 3) Collect per-race telemetry summaries
    rows: List[pd.DataFrame] = []
    for (y, r) in past:
        df = _load_telemetry_for_race(raw_dir, y, r)
        if df.empty:
            continue
        df = df.copy()
        df["year"] = y
        df["round"] = r
        df["same_track"] = int(_event_slug_from_meta(raw_dir, y, r) == target_slug)
        rows.append(df)

    if not rows:
        base = pd.DataFrame({"Driver": drivers}) if drivers else pd.DataFrame()
        if base.empty:
            return base
        return _attach_defaults(base)

    hist = pd.concat(rows, ignore_index=True, sort=False)
    hist = _ensure_driver(hist)
    hist["best_lap_s"] = pd.to_numeric(hist["best_lap_s"], errors="coerce")
    hist["top_speed_kph"] = pd.to_numeric(hist["top_speed_kph"], errors="coerce")
    hist = hist.dropna(subset=["Driver"])

    # 4) Field-normalized pace z per race
    hist = hist.sort_values(["year", "round"]).reset_index(drop=True)
    hist["race_id"] = hist["year"].astype(str) + ":" + hist["round"].astype(str)
    z_list = []
    for _, g in hist.groupby("race_id", sort=False):
        z = _zscore_inv(g["best_lap_s"])  # higher is better
        z.index = g.index
        z_list.append(z)
    hist["pace_z"] = pd.concat(z_list).sort_index()

    # 5) Aggregate per driver
    def _agg_driver(g: pd.DataFrame) -> pd.Series:
        g = g.dropna(subset=["best_lap_s"])
        if g.empty:
            return pd.Series(dtype=float)
        w = np.ones(len(g), dtype=float)
        if "same_track" in g.columns:
            w = w + opt.same_track_bonus * g["same_track"].to_numpy(dtype=float)
        w = w / w.sum()
        # robust percentiles via numpy
        arr = g["best_lap_s"].to_numpy(dtype=float)
        best_p50 = float(np.nanpercentile(arr, 50))
        iqr = float(np.nanpercentile(arr, 75) - np.nanpercentile(arr, 25))
        pace_mean = float(np.nansum(g["pace_z"].to_numpy(dtype=float) * w))
        trend = _linreg_slope(g.sort_values(["year","round"])["pace_z"].to_numpy())
        ts = g["top_speed_kph"].to_numpy(dtype=float)
        ts_p75 = float(np.nanpercentile(ts, 75)) if np.isfinite(ts).any() else np.nan
        last = g.sort_values(["year","round"]).tail(1)[["year","round"]].iloc[0]
        same_n = int(g["same_track"].sum()) if "same_track" in g.columns else 0
        return pd.Series({
            "tele_pre_hist_n": int(len(g)),
            "tele_pre_bestlap_p50_s": best_p50,
            "tele_pre_bestlap_iqr_s": iqr,
            "tele_pre_pace_z_mean": pace_mean,
            "tele_pre_pace_z_trend": trend,
            "tele_pre_topspeed_p75": ts_p75,
            "tele_pre_same_track_n": same_n,
            "tele_pre_last_seen_year": int(last["year"]),
            "tele_pre_last_seen_round": int(last["round"]),
        })

    agg = hist.groupby("Driver", dropna=False).apply(_agg_driver).reset_index()

    # 6) Align to roster & fill defaults
    if drivers:
        base = pd.DataFrame({"Driver": drivers})
        out = base.merge(agg, on="Driver", how="left")
        out = _fill_defaults(out)
    else:
        out = _fill_defaults(agg)

    for c in ("tele_pre_hist_n", "tele_pre_same_track_n", "tele_pre_last_seen_year", "tele_pre_last_seen_round"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")

    return out


def _attach_defaults(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["tele_pre_hist_n"] = 0
    out["tele_pre_bestlap_p50_s"] = np.nan
    out["tele_pre_bestlap_iqr_s"] = np.nan
    out["tele_pre_pace_z_mean"] = 0.0
    out["tele_pre_pace_z_trend"] = 0.0
    out["tele_pre_topspeed_p75"] = np.nan
    out["tele_pre_same_track_n"] = 0
    out["tele_pre_last_seen_year"] = pd.NA
    out["tele_pre_last_seen_round"] = pd.NA
    return out


def _fill_defaults(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "tele_pre_hist_n" not in out.columns:
        return _attach_defaults(out)
    out["tele_pre_hist_n"] = out["tele_pre_hist_n"].fillna(0)
    out["tele_pre_bestlap_p50_s"] = out["tele_pre_bestlap_p50_s"].astype(float)
    out["tele_pre_bestlap_iqr_s"] = out["tele_pre_bestlap_iqr_s"].astype(float)
    out["tele_pre_pace_z_mean"] = out["tele_pre_pace_z_mean"].fillna(0.0)
    out["tele_pre_pace_z_trend"] = out["tele_pre_pace_z_trend"].fillna(0.0)
    out["tele_pre_topspeed_p75"] = out["tele_pre_topspeed_p75"].astype(float)
    out["tele_pre_same_track_n"] = out["tele_pre_same_track_n"].fillna(0)
    return out


# ---------------------------- Public API ----------------------------
@dataclass
class CtxOptions:
    max_lookback: int = 10
    min_hist: int = 1
    same_track_bonus: float = 0.50
    require_same_track_if_short: int = 0


def featurize(*args, **kwargs) -> pd.DataFrame:
    """
    Совместимость:
      - featurize(ctx)  — ожидает в ctx: raw_dir, year/season, rnd/round, опц.: roster, races_df/races_csv,
                           max_lookback, same_track_bonus, require_same_track_if_short.
      - featurize(raw_dir, races_df, year, rnd, roster=None, options=None)
    """
    # Variant 1: featurize(ctx)
    if len(args) == 1 and not kwargs and not isinstance(args[0], (str, Path)):
        ctx = args[0]
        get = (ctx.get if isinstance(ctx, dict) else lambda k, d=None: getattr(ctx, k, d))
        raw_dir = Path(get("raw_dir", get("raw", ".")))
        # calendar
        races_df = get("races_df", None)
        races_csv = get("races_csv", None)
        if races_df is None and races_csv is not None:
            try:
                races_df = pd.read_csv(races_csv)
            except Exception:
                races_df = None
        if races_df is None:
            races_df = _calendar_df_from_files(raw_dir)

        # ids
        year = get("year", get("season", None))
        rnd = get("rnd", get("round", None))
        # fallback: extract from build label like "2024_2"
        if year is None or rnd is None:
            label = get("build_id", get("build", get("label", get("name", None))))
            if label:
                m = re.search(r"(\d{4})[^\d]+(\d{1,2})", str(label))
                if m:
                    year = year or int(m.group(1))
                    rnd = rnd or int(m.group(2))
        if year is None or rnd is None:
            raise TypeError("telemetry_history_pre.featurize(ctx): укажите 'year'/'season' и 'rnd'/'round' (или build_id вида 'YYYY_R').")

        # roster (list/df/series)
        roster = None
        roster_src = get("roster", get("drivers", None))
        if roster_src is not None:
            if isinstance(roster_src, pd.DataFrame):
                dcol = _detect_driver_col(roster_src)
                if dcol:
                    roster = roster_src[dcol].astype(str).dropna().unique().tolist()
            elif isinstance(roster_src, (list, tuple, set, pd.Series)):
                roster = [str(x) for x in roster_src]

        opt = FeaturizeOptions(
            max_lookback=int(get("max_lookback", 10)),
            min_hist=int(get("min_hist", 1)),
            same_track_bonus=float(get("same_track_bonus", 0.50)),
            require_same_track_if_short=int(get("require_same_track_if_short", 0)),
        )

        return _featurize_impl(raw_dir, races_df, int(year), int(rnd), roster=roster, options=opt)

    # Variant 2: legacy signature
    if len(args) >= 4 and isinstance(args[0], (str, Path)):
        raw_dir, races_df, year, rnd = args[:4]
        roster = args[4] if len(args) >= 5 else kwargs.get("roster")
        options = args[5] if len(args) >= 6 else kwargs.get("options")
        return _featurize_impl(Path(raw_dir), races_df, int(year), int(rnd), roster=roster,
                               options=options or FeaturizeOptions())

    raise TypeError("telemetry_history_pre.featurize: ожидается featurize(ctx) или featurize(raw_dir, races_df, year, rnd, ...).")


# ---------------------------- CLI (optional) ----------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Build leak-safe telemetry priors")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--races-csv", default=None)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--roster", default=None)
    ap.add_argument("--max-lookback", type=int, default=10)
    ap.add_argument("--same-track-bonus", type=float, default=0.50)
    ap.add_argument("--require-same-track-if-short", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    races_df = pd.read_csv(args.races_csv) if args.races_csv else _calendar_df_from_files(raw_dir)
    roster = None
    if args.roster:
        r = pd.read_csv(args.roster)
        col = next((c for c in r.columns if c.lower() in ("driver", "abbreviation")), None)
        if col:
            roster = r[col].astype(str).dropna().unique().tolist()

    opts = FeaturizeOptions(
        max_lookback=args.max_lookback,
        same_track_bonus=args.same_track_bonus,
        require_same_track_if_short=args.require_same_track_if_short,
    )
    out = _featurize_impl(raw_dir, races_df, args.year, args.round, roster=roster, options=opts)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Saved priors to {args.out} — rows={len(out)}")
