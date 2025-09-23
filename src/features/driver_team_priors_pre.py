from __future__ import annotations
"""
Driver & Team priors (LEAK-SAFE, pre-race)

- Только прошлые этапы: (y < year) или (y == year и r < rnd)
- РОСТЕР И КОМАНДЫ текущего этапа — только из entrylist_{y}_{r}[_Q].csv или results_{y}_{r}_Q.csv
- Если календарь не передали, строим его по файлам в raw_dir

Выход — по одному ряду на Driver:
  driver_team_pre_driver_hist_n
  driver_team_pre_driver_finish_p50
  driver_team_pre_driver_finish_iqr
  driver_team_pre_driver_top10_rate
  driver_team_pre_driver_dnf_rate
  driver_team_pre_driver_grid_p50
  driver_team_pre_driver_points_mean
  driver_team_pre_team_hist_n
  driver_team_pre_team_finish_p50
  driver_team_pre_team_top10_rate
  driver_team_pre_team_dnf_rate
  driver_team_pre_team_points_mean
  driver_team_pre_last_seen_year
  driver_team_pre_last_seen_round
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union
import re

import numpy as np
import pandas as pd

# ----------------------------- I/O utils -----------------------------

DRIVER_COLS = ("Driver", "Abbreviation", "driverRef", "DriverRef", "DriverCode", "BroadcastName")
TEAM_COLS   = ("Team", "TeamName", "Constructor", "ConstructorName", "ConstructorTeam")
POS_COLS    = ("Position", "position", "ResultPosition", "FinishPosition")
GRID_COLS   = ("GridPosition", "Grid", "GridPos")
PTS_COLS    = ("Points", "points")
STATUS_COLS = ("Status", "status", "ResultStatus", "Classified")


def _read_csv(p: Path) -> pd.DataFrame:
    try:
        if not p.exists():
            return pd.DataFrame()
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _detect_col(df: pd.DataFrame, cands: Sequence[str]) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None

# ----------------------------- Calendar -----------------------------

def _list_past(races_df: pd.DataFrame, year: int, rnd: int, max_lookback: int) -> List[Tuple[int, int]]:
    if races_df is None or races_df.empty:
        return []
    ycol = next((c for c in races_df.columns if c.lower() in ("year", "season")), None)
    rcol = next((c for c in races_df.columns if c.lower() == "round"), None)
    if not ycol or not rcol:
        return []
    df = races_df[[ycol, rcol]].copy()
    df.columns = ["year", "round"]

    # приведение к числам на всякий случай
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["round"] = pd.to_numeric(df["round"], errors="coerce")

    past = df[(df["year"] < year) | ((df["year"] == year) & (df["round"] < rnd))]
    past = past.sort_values(["year", "round"], ascending=[False, False]).head(max_lookback)
    return list(map(tuple, past[["year", "round"]].values.tolist()))


def _scan_available_rounds(raw_dir: Path) -> List[Tuple[int, int]]:
    rr = set()
    for pat in ("results_*.csv", "laps_*.csv", "entrylist_*.csv", "results_*_Q.csv", "entrylist_*_Q.csv"):
        for p in raw_dir.glob(pat):
            m = re.search(r"(\d{4})[_-](\d{1,2})", p.stem)
            if m:
                rr.add((int(m.group(1)), int(m.group(2))))
    return sorted(rr)

def _calendar_df_from_files(raw_dir: Path) -> pd.DataFrame:
    pairs = _scan_available_rounds(raw_dir)
    if not pairs:
        return pd.DataFrame(columns=["year","round"])
    return pd.DataFrame(pairs, columns=["year","round"]).sort_values(["year","round"]).reset_index(drop=True)

def _list_past_by_scan(raw_dir: Path, year: int, rnd: int, max_lookback: int) -> List[Tuple[int, int]]:
    all_rr = _scan_available_rounds(raw_dir)
    past = [(y, r) for (y, r) in all_rr if (y < year) or (y == year and r < rnd)]
    past.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return past[:max_lookback]

# -------------------- Current roster & team (SAFE) --------------------

_ROSTER_FILES = (
    "entrylist_{y}_{r}_Q.csv",
    "entrylist_{y}_{r}.csv",
    "results_{y}_{r}_Q.csv",   # safe (qual only)
)

def _current_roster_team(raw_dir: Path, year: int, rnd: int) -> pd.DataFrame:
    """DF[Driver, Team] на ТЕКУЩИЙ раунд из безопасных источников."""
    for pat in _ROSTER_FILES:
        df = _read_csv(raw_dir / pat.format(y=year, r=rnd))
        if df.empty:
            continue
        dcol = _detect_col(df, DRIVER_COLS)
        if not dcol:
            continue
        out = pd.DataFrame({"Driver": df[dcol].astype(str)})
        tcol = _detect_col(df, TEAM_COLS)
        out["Team"] = (df[tcol].astype(str) if tcol else np.nan)
        out = out.dropna(subset=["Driver"]).drop_duplicates("Driver")
        if not out.empty:
            return out[["Driver","Team"]]
    return pd.DataFrame(columns=["Driver","Team"])  # may be empty

def _last_known_team(raw_dir: Path, past: List[Tuple[int, int]]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for (y, r) in past:
        res = _read_csv(raw_dir / f"results_{y}_{r}.csv")
        if res.empty:
            continue
        dcol = _detect_col(res, DRIVER_COLS)
        tcol = _detect_col(res, TEAM_COLS)
        if not dcol:
            continue
        sub = pd.DataFrame({"Driver": res[dcol].astype(str)})
        sub["Team"] = (res[tcol].astype(str) if tcol else np.nan)
        sub["year"], sub["round"] = y, r
        rows.append(sub)
    if not rows:
        return pd.DataFrame(columns=["Driver","Team"])
    hist = pd.concat(rows, ignore_index=True)
    hist = hist.sort_values(["Driver","year","round"]).reset_index(drop=True)
    last = hist.groupby("Driver").tail(1)[["Driver","Team"]]
    return last.reset_index(drop=True)

# --------------------------- Past results ---------------------------

_DNF_TOKENS = (
    "DNF", "Accident", "Collision", "Engine", "Hydraulics", "Electrical",
    "Gearbox", "Transmission", "Suspension", "Brakes", "Puncture",
    "Overheating", "Mechanical", "Exhaust", "Clutch", "Wheel", "Driveshaft",
    "Steering", "Fuel", "Oil leak", "Water pressure", "Power Unit", "PU",
)

def _results_table(raw_dir: Path, y: int, r: int) -> pd.DataFrame:
    df = _read_csv(raw_dir / f"results_{y}_{r}.csv")
    if df.empty:
        return df
    dcol = _detect_col(df, DRIVER_COLS)
    tcol = _detect_col(df, TEAM_COLS)
    pcol = _detect_col(df, POS_COLS)
    gcol = _detect_col(df, GRID_COLS)
    scol = _detect_col(df, STATUS_COLS)
    if dcol and dcol != "Driver":
        df = df.rename(columns={dcol: "Driver"})
    if tcol and tcol != "Team":
        df = df.rename(columns={tcol: "Team"})
    if pcol: df["Position"] = pd.to_numeric(df[pcol], errors="coerce")
    if gcol: df["GridPosition"] = pd.to_numeric(df[gcol], errors="coerce")
    pts = _detect_col(df, PTS_COLS)
    if pts: df["Points"] = pd.to_numeric(df[pts], errors="coerce")
    if scol:
        s = df[scol].astype(str)
        if scol.lower() == "classified":
            dnf = ~df[scol].astype(bool)
        else:
            sup = s.str.upper()
            dnf = sup.str.contains("DNF") | sup.eq("NC")
            for tok in _DNF_TOKENS:
                dnf = dnf | s.str.contains(tok, case=False, na=False)
        df["DNF"] = dnf.astype(float)
    else:
        df["DNF"] = np.nan
    return df[[c for c in ["Driver","Team","Position","GridPosition","Points","DNF"] if c in df.columns]]

# ---------------------------- Aggregations ----------------------------

def _agg_driver(g: pd.DataFrame) -> pd.Series:
    pos = pd.to_numeric(g.get("Position"), errors="coerce")
    grid = pd.to_numeric(g.get("GridPosition"), errors="coerce")
    pts = pd.to_numeric(g.get("Points"), errors="coerce")
    dnf = pd.to_numeric(g.get("DNF"), errors="coerce")

    hist_n = int(len(g))
    p50 = float(np.nanmedian(pos)) if pos.notna().any() else np.nan
    iqr = float(np.nanpercentile(pos.dropna(), 75) - np.nanpercentile(pos.dropna(), 25)) if pos.notna().sum() >= 2 else np.nan
    top10 = float((pos <= 10).mean()) if pos.notna().any() else np.nan
    dnf_rate = float(dnf.mean()) if dnf.notna().any() else np.nan
    grid_p50 = float(np.nanmedian(grid)) if grid.notna().any() else np.nan
    pts_mean = float(pts.mean()) if pts.notna().any() else np.nan

    return pd.Series({
        "driver_team_pre_driver_hist_n": hist_n,
        "driver_team_pre_driver_finish_p50": p50,
        "driver_team_pre_driver_finish_iqr": iqr,
        "driver_team_pre_driver_top10_rate": top10,
        "driver_team_pre_driver_dnf_rate": dnf_rate,
        "driver_team_pre_driver_grid_p50": grid_p50,
        "driver_team_pre_driver_points_mean": pts_mean,
    })


def _agg_team(g: pd.DataFrame) -> pd.Series:
    pos = pd.to_numeric(g.get("Position"), errors="coerce")
    pts = pd.to_numeric(g.get("Points"), errors="coerce")
    dnf = pd.to_numeric(g.get("DNF"), errors="coerce")

    hist_n = int(len(g))
    p50 = float(np.nanmedian(pos)) if pos.notna().any() else np.nan
    top10 = float((pos <= 10).mean()) if pos.notna().any() else np.nan
    dnf_rate = float(dnf.mean()) if dnf.notna().any() else np.nan
    pts_mean = float(pts.mean()) if pts.notna().any() else np.nan

    return pd.Series({
        "driver_team_pre_team_hist_n": hist_n,
        "driver_team_pre_team_finish_p50": p50,
        "driver_team_pre_team_top10_rate": top10,
        "driver_team_pre_team_dnf_rate": dnf_rate,
        "driver_team_pre_team_points_mean": pts_mean,
    })

# ------------------------------- API -------------------------------

@dataclass
class Options:
    max_lookback: int = 10


def _featurize_impl(
    raw_dir: Path | str,
    races_df: pd.DataFrame,
    year: int,
    rnd: int,
    roster: Optional[Sequence[str]] = None,
    options: Optional[Options] = None,
) -> pd.DataFrame:
    """Leak-safe driver/team priors for the target round."""
    raw_dir = Path(raw_dir)
    opt = options or Options()

    # прошлые гонки
    past = _list_past(races_df, year, rnd, max_lookback=opt.max_lookback) if races_df is not None else \
           _list_past_by_scan(raw_dir, year, rnd, max_lookback=opt.max_lookback)

    # собираем прошлые результаты
    rows = []
    for (y, r) in past:
        tbl = _results_table(raw_dir, y, r)
        if not tbl.empty:
            tbl["year"], tbl["round"] = y, r
            rows.append(tbl)
    if rows:
        hist = pd.concat(rows, ignore_index=True)
    else:
        hist = pd.DataFrame(columns=["Driver","Team","Position","GridPosition","Points","DNF","year","round"])

    # агрегаты по пилотам/командам
    drv_agg = (hist.groupby("Driver", dropna=False).apply(_agg_driver).reset_index()) if not hist.empty else pd.DataFrame(columns=["Driver"])
    team_agg_raw = (hist.groupby("Team", dropna=False).apply(_agg_team).reset_index()) if not hist.empty else pd.DataFrame(columns=["Team"])

    # текущая команда (без утечек)
    cur = _current_roster_team(raw_dir, year, rnd)
    if cur.empty:
        cur = _last_known_team(raw_dir, past)

    if roster:
        cur = pd.DataFrame({"Driver": list(roster)}).merge(cur, on="Driver", how="left")

    # базовый список пилотов
    if cur.empty:
        drivers = sorted(set(hist["Driver"].dropna().astype(str)))
        base = pd.DataFrame({"Driver": drivers})
    else:
        base = cur[["Driver","Team"]].drop_duplicates("Driver")

    # join
    out = base.merge(drv_agg, on="Driver", how="left")
    if "Team" in base.columns and not team_agg_raw.empty:
        out = out.merge(team_agg_raw.rename(columns={"Team":"_join_team"}), left_on="Team", right_on="_join_team", how="left")
        out = out.drop(columns=[c for c in out.columns if c == "_join_team"])

    # last seen
    if not hist.empty:
        last_seen = hist.groupby("Driver").apply(lambda g: g.sort_values(["year","round"]).iloc[-1][["year","round"]])
        last_seen = last_seen.reset_index()[["Driver","year","round"]]
        last_seen = last_seen.rename(columns={"year":"driver_team_pre_last_seen_year", "round":"driver_team_pre_last_seen_round"})
        out = out.merge(last_seen, on="Driver", how="left")
    else:
        out["driver_team_pre_last_seen_year"] = pd.NA
        out["driver_team_pre_last_seen_round"] = pd.NA

    # типы
    for c in ("driver_team_pre_driver_hist_n", "driver_team_pre_team_hist_n",
              "driver_team_pre_last_seen_year", "driver_team_pre_last_seen_round"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")

    return out

# ------------------------------- CLI -------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser("Leak-safe driver & team priors (pre-race)")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--races-csv", default=None)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--roster-csv", default=None, help="Optional CSV with a 'Driver' column")
    ap.add_argument("--max-lookback", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    races_df = pd.read_csv(args.races_csv) if args.races_csv else _calendar_df_from_files(raw_dir)

    roster = None
    if args.roster_csv:
        r = _read_csv(Path(args.roster_csv))
        c = _detect_col(r, DRIVER_COLS)
        if c:
            roster = r[c].astype(str).dropna().unique().tolist()

    out = _featurize_impl(raw_dir, races_df, args.year, args.round, roster=roster, options=Options(max_lookback=args.max_lookback))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Saved driver/team priors to {args.out} (rows={len(out)})")

# ------------------------------- Wrapper -------------------------------

def featurize(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """
    Совместимость:
      - featurize(ctx) — ctx.raw_dir обязателен; races_df/races_csv опциональны (есть fallback-скан).
        Поддерживаются ключи: year/season, rnd/round, build_id='YYYY_R'; roster/drivers; max_lookback.
      - featurize(raw_dir, races_df, year, rnd, roster=None, options=None)
    """
    # ctx-style
    if len(args) == 1 and not kwargs and not isinstance(args[0], (str, Path)):
        ctx = args[0]
        get = (ctx.get if isinstance(ctx, dict) else lambda k, d=None: getattr(ctx, k, d))
        raw_dir = Path(get("raw_dir", get("raw", ".")))

        # races_df: direct, via CSV, или по файлам
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
        rnd  = get("rnd", get("round", None))
        if year is None or rnd is None:
            label = get("build_id", get("build", get("label", get("name", None))))
            if label:
                m = re.search(r"(\d{4})[^\d]+(\d{1,2})", str(label))
                if m:
                    year = year or int(m.group(1))
                    rnd  = rnd  or int(m.group(2))
        if year is None or rnd is None:
            raise TypeError("driver_team_priors_pre.featurize(ctx): укажите 'year'/'season' и 'rnd'/'round' (или build_id 'YYYY_R').")

        # roster
        roster = get("roster", get("drivers", None))
        if isinstance(roster, pd.DataFrame):
            c = _detect_col(roster, DRIVER_COLS)
            if c:
                roster = roster[c].astype(str).dropna().unique().tolist()

        max_lb = int(get("max_lookback", 10))
        opts = Options(max_lookback=max_lb)
        return _featurize_impl(raw_dir, races_df, int(year), int(rnd), roster=roster, options=opts)

    # positional legacy
    return _featurize_impl(*args, **kwargs)
