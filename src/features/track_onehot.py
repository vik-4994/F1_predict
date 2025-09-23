from __future__ import annotations
"""
track_onehot — признаки трека для текущего билда (pre-race, leak-safe)

Что делает:
  1) Создаёт one-hot колонку трека: track_is_<slug> (=1 для всех пилотов текущего раунда).
  2) Считает per-driver "same track" историю ТОЛЬКО по прошлым гонкам на том же треке:
     - track_same_hist_n
     - track_same_finish_p50
     - track_same_bestpos_min
     - track_same_top10_rate
     - track_same_dnf_rate
     - track_same_points_mean
     - track_same_grid_p50
     - track_same_last_seen_year
     - track_same_last_seen_round

Совместимость:
  - featurize(ctx): ctx.raw_dir обязателен; year/season и rnd/round; опц.: track, races_df/races_csv, roster/drivers
  - featurize(raw_dir, races_df, year, rnd, roster=None, options=None)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union, List
import re
import unicodedata
import numpy as np
import pandas as pd

# -------------------- config --------------------

DRIVER_COLS   = ("Driver", "Abbreviation", "driverRef", "DriverRef", "DriverCode", "BroadcastName")
TEAM_COLS     = ("Team", "TeamName", "Constructor", "ConstructorName", "ConstructorTeam")
EVENT_COLS    = ("EventName", "Event", "GrandPrix", "RaceName", "raceName", "Name")
CIRCUIT_COLS  = ("Circuit", "CircuitName", "CircuitShortName", "Track", "Venue")
POS_COLS      = ("Position", "position", "ResultPosition", "FinishPosition")
GRID_COLS     = ("GridPosition", "Grid", "GridPos")
PTS_COLS      = ("Points", "points")
STATUS_COLS   = ("Status", "status", "ResultStatus", "Classified")

DNF_TOKENS = (
    "DNF", "NC", "Accident", "Collision", "Engine", "Hydraulics", "Electrical",
    "Gearbox", "Transmission", "Suspension", "Brakes", "Puncture", "Overheating",
    "Mechanical", "Exhaust", "Clutch", "Wheel", "Driveshaft", "Steering",
    "Fuel", "Oil leak", "Water pressure", "Power Unit", "PU",
)

# -------------------- utils --------------------

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

def _slugify(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return re.sub(r"_+", "_", s)

def _event_name_from_ctx(ctx) -> Optional[str]:
    for k in ("track", "event", "circuit", "grand_prix", "race"):
        v = ctx.get(k) if isinstance(ctx, dict) else getattr(ctx, k, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def _event_name_from_meta(raw_dir: Path, year: int, rnd: int) -> Optional[str]:
    for pat in (f"meta_{year}_{rnd}.csv", f"meta_{year}_{rnd}_Q.csv"):
        df = _read_csv(raw_dir / pat)
        if df.empty:
            continue
        for c in EVENT_COLS:
            if c in df.columns and df[c].notna().any():
                v = str(df[c].iloc[0]).strip()
                if v: return v
        for c in CIRCUIT_COLS:
            if c in df.columns and df[c].notna().any():
                v = str(df[c].iloc[0]).strip()
                if v: return v
    return None

def _event_slug_from_meta(raw_dir: Path, year: int, rnd: int) -> Optional[str]:
    name = _event_name_from_meta(raw_dir, year, rnd)
    return _slugify(name) if name else None

def _event_name_from_races_df(races_df: pd.DataFrame, year: int, rnd: int) -> Optional[str]:
    if races_df is None or races_df.empty:
        return None
    ycol = next((c for c in races_df.columns if c.lower() in ("year", "season")), None)
    rcol = next((c for c in races_df.columns if c.lower() == "round"), None)
    if ycol is None or rcol is None:
        return None
    df = races_df.copy()
    df = df[(pd.to_numeric(df[ycol], errors="coerce") == year) & (pd.to_numeric(df[rcol], errors="coerce") == rnd)]
    if df.empty:
        return None
    for c in (*EVENT_COLS, *CIRCUIT_COLS):
        if c in df.columns and df[c].notna().any():
            return str(df[c].iloc[0]).strip()
    return None

def _ensure_driver_list(ctx, raw_dir: Path, year: int, rnd: int) -> list[str]:
    roster = getattr(ctx, "roster", None) if not isinstance(ctx, dict) else ctx.get("roster")
    drivers = getattr(ctx, "drivers", None) if not isinstance(ctx, dict) else ctx.get("drivers")
    src = roster if roster is not None else drivers
    if isinstance(src, pd.DataFrame):
        c = _detect_col(src, DRIVER_COLS)
        if c:
            vals = src[c].astype(str).dropna().unique().tolist()
            if vals: return vals
    elif isinstance(src, (list, tuple, set, pd.Series)):
        vals = [str(x) for x in src if str(x).strip()]
        if vals: return list(dict.fromkeys(vals))
    for fname in (f"entrylist_{year}_{rnd}_Q.csv", f"entrylist_{year}_{rnd}.csv", f"results_{year}_{rnd}_Q.csv"):
        df = _read_csv(raw_dir / fname)
        if not df.empty:
            c = _detect_col(df, DRIVER_COLS)
            if c:
                vals = df[c].astype(str).dropna().unique().tolist()
                if vals: return vals
    cand = sorted(raw_dir.glob("results_*.csv"))
    if cand:
        df = _read_csv(cand[-1])
        c = _detect_col(df, DRIVER_COLS)
        if c:
            vals = df[c].astype(str).dropna().unique().tolist()
            if vals: return vals
    return []

# -------------------- calendar helpers --------------------

def _scan_available_rounds(raw_dir: Path) -> List[Tuple[int, int]]:
    rr = set()
    for pat in ("results_*.csv", "entrylist_*.csv", "results_*_Q.csv", "entrylist_*_Q.csv", "meta_*.csv", "meta_*_Q.csv"):
        for p in raw_dir.glob(pat):
            m = re.search(r"(\d{4})[_-](\d{1,2})", p.stem)
            if m:
                rr.add((int(m.group(1)), int(m.group(2))))
    return sorted(rr)

def _calendar_df_from_files(raw_dir: Path) -> pd.DataFrame:
    pairs = _scan_available_rounds(raw_dir)
    if not pairs:
        return pd.DataFrame(columns=["year", "round"])
    return pd.DataFrame(pairs, columns=["year", "round"]).sort_values(["year", "round"]).reset_index(drop=True)

def _list_past(races_df: Optional[pd.DataFrame], raw_dir: Path, year: int, rnd: int, max_lookback: int) -> List[Tuple[int, int]]:
    if races_df is not None and not races_df.empty:
        ycol = next((c for c in races_df.columns if c.lower() in ("year", "season")), None)
        rcol = next((c for c in races_df.columns if c.lower() == "round"), None)
        if ycol and rcol:
            df = races_df[[ycol, rcol]].copy(); df.columns = ["year", "round"]
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df["round"] = pd.to_numeric(df["round"], errors="coerce")
            past = df[(df["year"] < year) | ((df["year"] == year) & (df["round"] < rnd))]
            past = past.sort_values(["year", "round"], ascending=[False, False]).head(max_lookback)
            return list(map(tuple, past[["year", "round"]].values.tolist()))
    # fallback — по файлам
    all_rr = _scan_available_rounds(raw_dir)
    past = [(y, r) for (y, r) in all_rr if (y < year) or (y == year and r < rnd)]
    past.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return past[:max_lookback]

# -------------------- results per round --------------------

def _results_table(raw_dir: Path, y: int, r: int) -> pd.DataFrame:
    df = _read_csv(raw_dir / f"results_{y}_{r}.csv")
    if df.empty:
        return df
    dcol = _detect_col(df, DRIVER_COLS)
    tcol = _detect_col(df, TEAM_COLS)
    pcol = _detect_col(df, POS_COLS)
    gcol = _detect_col(df, GRID_COLS)
    scol = _detect_col(df, STATUS_COLS)
    pts  = _detect_col(df, PTS_COLS)

    if dcol and dcol != "Driver": df = df.rename(columns={dcol: "Driver"})
    if tcol and tcol != "Team":   df = df.rename(columns={tcol: "Team"})
    if pcol: df["Position"]     = pd.to_numeric(df[pcol], errors="coerce")
    if gcol: df["GridPosition"] = pd.to_numeric(df[gcol], errors="coerce")
    if pts:  df["Points"]       = pd.to_numeric(df[pts], errors="coerce")

    if scol:
        s = df[scol].astype(str)
        if scol.lower() == "classified":
            dnf = ~df[scol].astype(bool)
        else:
            su = s.str.upper()
            dnf = su.str.contains("DNF") | su.eq("NC")
            for tok in DNF_TOKENS:
                dnf = dnf | s.str.contains(tok, case=False, na=False)
        df["DNF"] = dnf.astype(float)
    else:
        df["DNF"] = np.nan

    df["year"] = y; df["round"] = r
    keep = [c for c in ["Driver","Team","Position","GridPosition","Points","DNF","year","round"] if c in df.columns]
    return df[keep]

# -------------------- core --------------------

@dataclass
class Options:
    max_lookback: int = 20        # сколько прошлых раундов сканировать
    include_onehot: bool = True   # делать ли track_is_<slug>
    require_meta_match: bool = True  # матчить трек по meta_*; если False — пытаться сопоставлять по races_df именам
    min_hist: int = 0             # минимальная история; если 0 — просто будут NaN/0

def _build_same_track_agg(raw_dir: Path, target_slug: str, past: List[Tuple[int,int]]) -> pd.DataFrame:
    """Собрать историю только по тем прошлым этапам, у которых slug(meta_{y}_{r}) совпадает с target_slug."""
    rows = []
    for (y, r) in past:
        slug = _event_slug_from_meta(raw_dir, y, r)
        if not slug or slug != target_slug:
            continue
        tbl = _results_table(raw_dir, y, r)
        if not tbl.empty:
            rows.append(tbl)
    if not rows:
        return pd.DataFrame(columns=["Driver"])
    hist = pd.concat(rows, ignore_index=True)

    # --- агрегаты по пилоту ---
    g = hist.groupby("Driver", dropna=False)

    def _p50(s: pd.Series) -> float:
        x = pd.to_numeric(s, errors="coerce")
        return float(np.nanmedian(x)) if x.notna().any() else np.nan

    def _iqr(s: pd.Series) -> float:
        x = pd.to_numeric(s, errors="coerce").dropna().to_numpy()
        return float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25)) if x.size >= 2 else np.nan

    agg = pd.DataFrame({
        "track_same_hist_n": g.size().astype(int),
        "track_same_finish_p50": g["Position"].apply(_p50),
        "track_same_bestpos_min": g["Position"].min(),
        "track_same_top10_rate": g["Position"].apply(lambda s: float((pd.to_numeric(s, errors="coerce") <= 10).mean()) if s.notna().any() else np.nan),
        "track_same_dnf_rate": g["DNF"].mean(),
        "track_same_points_mean": g["Points"].mean(),
        "track_same_grid_p50": g["GridPosition"].apply(_p50),
    }).reset_index()

    # last seen
    last = g[["year","round"]].max().reset_index().rename(columns={"year":"track_same_last_seen_year", "round":"track_same_last_seen_round"})
    out = agg.merge(last, on="Driver", how="left")

    # типы
    for c in ("track_same_hist_n","track_same_last_seen_year","track_same_last_seen_round","track_same_bestpos_min"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    return out

def _featurize_impl(
    raw_dir: Union[str, Path],
    year: int,
    rnd: int,
    races_df: Optional[pd.DataFrame] = None,
    roster: Optional[Sequence[str]] = None,
    options: Optional[Options] = None,
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    opt = options or Options()

    # 1) Имя и slug трека
    name = _event_name_from_ctx(locals().get("ctx", {}))
    if not name:
        name = _event_name_from_meta(raw_dir, year, rnd) or _event_name_from_races_df(races_df, year, rnd)
    slug = _slugify(name) if name else None

    # 2) База пилотов
    drivers = [str(x) for x in (roster or [])]
    if not drivers:
        dummy_ctx = {"roster": None, "drivers": None}
        drivers = _ensure_driver_list(dummy_ctx, raw_dir, year, rnd)
    if not drivers:
        return pd.DataFrame()  # empty

    base = pd.DataFrame({"Driver": drivers})

    # 3) One-hot (по желанию)
    if opt.include_onehot:
        col = f"track_is_{slug}" if slug else "track_is_unknown"
        base[col] = 1

    # 4) Same-track агрегации (только прошлое)
    past = _list_past(races_df, raw_dir, year, rnd, max_lookback=opt.max_lookback)
    if slug:
        same = _build_same_track_agg(raw_dir, slug, past)
        if not same.empty:
            base = base.merge(same, on="Driver", how="left")
        else:
            # если истории нет — аккуратно подложим дефолты
            for c in ["track_same_hist_n","track_same_finish_p50","track_same_bestpos_min",
                      "track_same_top10_rate","track_same_dnf_rate","track_same_points_mean",
                      "track_same_grid_p50","track_same_last_seen_year","track_same_last_seen_round"]:
                base[c] = base.get(c, np.nan)
            base["track_same_hist_n"] = base["track_same_hist_n"].fillna(0).astype("Int64")
    else:
        # неизвестный трек — только one-hot 'unknown' (если включено)
        pass

    return base

# -------------------- public API --------------------

def featurize(*args, **kwargs) -> pd.DataFrame:
    # ctx-режим
    if len(args) == 1 and not kwargs and not isinstance(args[0], (str, Path)):
        ctx = args[0]
        get = (ctx.get if isinstance(ctx, dict) else lambda k, d=None: getattr(ctx, k, d))
        raw_dir = Path(get("raw_dir", "."))
        races_df = get("races_df", None)
        races_csv = get("races_csv", None)
        if races_df is None and races_csv:
            try:
                races_df = pd.read_csv(races_csv)
            except Exception:
                races_df = None
        year = get("year", get("season", None))
        rnd  = get("rnd", get("round", None))
        if year is None or rnd is None:
            label = get("build_id", get("build", get("label", None)))
            if label:
                m = re.search(r"(\d{4})[^\d]+(\d{1,2})", str(label))
                if m:
                    year = int(m.group(1)); rnd = int(m.group(2))
        if year is None or rnd is None:
            raise TypeError("track_onehot.featurize(ctx): укажите 'year'/'season' и 'rnd'/'round' (или build_id 'YYYY_R').")

        roster = get("roster", get("drivers", None))
        if isinstance(roster, pd.DataFrame):
            c = _detect_col(roster, DRIVER_COLS)
            roster = roster[c].astype(str).dropna().unique().tolist() if c else None
        elif isinstance(roster, (list, tuple, set, pd.Series)):
            roster = [str(x) for x in roster]
        else:
            roster = None

        opts = Options(
            max_lookback=int(get("max_lookback", 20)),
            include_onehot=bool(get("include_onehot", True)),
        )
        return _featurize_impl(raw_dir, int(year), int(rnd), races_df=races_df, roster=roster, options=opts)

    # старая сигнатура
    if len(args) >= 4 and isinstance(args[0], (str, Path)):
        raw_dir, races_df, year, rnd = args[:4]
        roster = args[4] if len(args) >= 5 else kwargs.get("roster")
        options = args[5] if len(args) >= 6 else kwargs.get("options")
        return _featurize_impl(Path(raw_dir), int(year), int(rnd), races_df=races_df, roster=roster, options=options or Options())

    raise TypeError("track_onehot.featurize: ожидается featurize(ctx) или featurize(raw_dir, races_df, year, rnd, ...).")
