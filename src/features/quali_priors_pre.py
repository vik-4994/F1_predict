from __future__ import annotations
"""
quali_priors_pre — исторические приоры по квалификации (pre-race, leak-safe)

Что считается по прошлым КВАЛИФИКАЦИЯМ до целевого этапа:
  - quali_pre_hist_n                (Int64)   — кол-во участий в квале на прошлых этапах
  - quali_pre_pos_p50               (float)   — медиана позиции старта
  - quali_pre_pos_iqr               (float)   — межквартильный размах по позиции
  - quali_pre_bestpos_min           (Int64)   — лучший старт (минимальная позиция)
  - quali_pre_top10_rate            (float)   — доля стартов с позицией <= 10
  - quali_pre_notime_or_pen_rate    (float)   — доля случаев без времени/штрафов (по статусу)
  - quali_pre_points_mean           (float)   — средние очки исходно из quali-файла (если есть)
  - quali_pre_last_seen_year        (Int64)
  - quali_pre_last_seen_round       (Int64)

Совместимость:
  - featurize(ctx): ctx.raw_dir обязателен; year/season и rnd/round; опц.: races_df/races_csv, roster/drivers, max_lookback
  - featurize(raw_dir, races_df, year, rnd, roster=None, options=None)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union, List
import re
import numpy as np
import pandas as pd

# -------------------- колонки/утилиты --------------------

DRIVER_COLS = ("Driver", "Abbreviation", "driverRef", "DriverRef", "DriverCode", "BroadcastName")
TEAM_COLS   = ("Team", "TeamName", "Constructor", "ConstructorName", "ConstructorTeam")

# для квал-позиций берём специализированные названия в приоритете
QUALI_POS_CANDS = ("QualiPosition", "QPosition", "Q_Pos", "QualifyingPosition")
POS_COLS        = ("Position", "position", "ResultPosition", "FinishPosition", *QUALI_POS_CANDS)

GRID_COLS = ("GridPosition", "Grid", "GridPos")
PTS_COLS  = ("Points", "points")
STAT_COLS = ("Status", "status", "ResultStatus", "Classified", "Note", "Penalty")

NO_TIME_TOKENS = (
    "NO TIME", "NO-TIME", "NOTIME", "DNS", "DNF", "DSQ", "DSQ.", "DISQUAL",
    "EXCLUDED", "PENALTY", "PEN", "DELETED", "TRACK LIMITS",
)

QUALI_PATTERNS = (
    "results_{y}_{r}_Q.csv",   # основная квалификация
    "qualifying_{y}_{r}.csv",  # альтернативное именование
    "quali_{y}_{r}.csv",       # ещё одно
    "results_{y}_{r}_SQ.csv",  # спринт-квалификация, если есть
)

ENTRY_PATTERNS = (
    "entrylist_{y}_{r}_Q.csv",
    "entrylist_{y}_{r}.csv",
    "results_{y}_{r}_Q.csv",
)

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

# -------------------- календарь --------------------

def _scan_available_rounds(raw_dir: Path) -> List[Tuple[int, int]]:
    rr = set()
    for pat in ("results_*_Q.csv", "qualifying_*.csv", "quali_*.csv", "results_*_SQ.csv", "entrylist_*.csv"):
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
    # сначала races_df, если дан
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
    # иначе — по файлам
    all_rr = _scan_available_rounds(raw_dir)
    past = [(y, r) for (y, r) in all_rr if (y < year) or (y == year and r < rnd)]
    past.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return past[:max_lookback]

# -------------------- данные по текущему этапу --------------------

def _current_roster(raw_dir: Path, year: int, rnd: int) -> pd.DataFrame:
    for pat in ENTRY_PATTERNS:
        df = _read_csv(raw_dir / pat.format(y=year, r=rnd))
        if df.empty:
            continue
        dcol = _detect_col(df, DRIVER_COLS)
        if not dcol:
            continue
        out = pd.DataFrame({"Driver": df[dcol].astype(str)})
        out = out.dropna(subset=["Driver"]).drop_duplicates("Driver")
        if not out.empty:
            return out
    return pd.DataFrame(columns=["Driver"])

def _last_known_roster_from_quali(raw_dir: Path, past: List[Tuple[int, int]]) -> pd.DataFrame:
    for (y, r) in past:
        for pat in QUALI_PATTERNS:
            df = _read_csv(raw_dir / pat.format(y=y, r=r))
            if df.empty:
                continue
            dcol = _detect_col(df, DRIVER_COLS)
            if dcol:
                out = pd.DataFrame({"Driver": df[dcol].astype(str)}).dropna().drop_duplicates("Driver")
                if not out.empty:
                    return out
    return pd.DataFrame(columns=["Driver"])

# -------------------- чтение квал-таблицы --------------------

def _qualifying_table(raw_dir: Path, y: int, r: int) -> pd.DataFrame:
    df = pd.DataFrame()
    for pat in QUALI_PATTERNS:
        f = raw_dir / pat.format(y=y, r=r)
        t = _read_csv(f)
        if not t.empty:
            df = t
            break
    if df.empty:
        return df

    dcol = _detect_col(df, DRIVER_COLS)
    tcol = _detect_col(df, TEAM_COLS)
    qcol = _detect_col(df, QUALI_POS_CANDS) or _detect_col(df, POS_COLS)
    gcol = _detect_col(df, GRID_COLS)
    pcol = _detect_col(df, PTS_COLS)
    scol = _detect_col(df, STAT_COLS)

    if dcol and dcol != "Driver": df = df.rename(columns={dcol: "Driver"})
    if tcol and tcol != "Team":   df = df.rename(columns={tcol: "Team"})

    if qcol:
        df["QualiPos"] = pd.to_numeric(df[qcol], errors="coerce")
    elif "Position" in df.columns:
        df["QualiPos"] = pd.to_numeric(df["Position"], errors="coerce")
    else:
        df["QualiPos"] = np.nan

    if gcol:
        df["GridPosition"] = pd.to_numeric(df[gcol], errors="coerce")
    if pcol:
        df["Points"] = pd.to_numeric(df[pcol], errors="coerce")

    # статус → индикатор no-time/penalty (включая DNS/DSQ)
    if scol:
        s = df[scol].astype(str).str.upper()
        bad = s.eq("NC") | s.eq("DNS") | s.eq("DNF") | s.eq("DSQ")
        for tok in NO_TIME_TOKENS:
            bad = bad | s.str.contains(tok, case=False, na=False)
        df["NoTimeOrPen"] = bad.astype(float)
    else:
        # в некоторых наборах нет статуса — считаем "нет времени", если позиция NaN
        df["NoTimeOrPen"] = df["QualiPos"].isna().astype(float)

    df["year"] = y; df["round"] = r
    keep = [c for c in ["Driver","Team","QualiPos","GridPosition","Points","NoTimeOrPen","year","round"] if c in df.columns]
    return df[keep]

# -------------------- агрегации --------------------

def _agg_quali(hist: pd.DataFrame) -> pd.DataFrame:
    if hist.empty:
        return pd.DataFrame(columns=["Driver"])

    # групповая агрегация без apply (чтобы не ловить FutureWarning)
    g = hist.groupby("Driver", dropna=False)

    def p50(s: pd.Series) -> float:
        x = pd.to_numeric(s, errors="coerce")
        return float(np.nanmedian(x)) if x.notna().any() else np.nan

    def iqr(s: pd.Series) -> float:
        x = pd.to_numeric(s, errors="coerce").dropna().to_numpy()
        return float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25)) if x.size >= 2 else np.nan

    out = pd.DataFrame({
        "quali_pre_hist_n": g.size().astype(int),
        "quali_pre_pos_p50": g["QualiPos"].apply(p50),
        "quali_pre_pos_iqr": g["QualiPos"].apply(iqr),
        "quali_pre_bestpos_min": g["QualiPos"].min().astype("float"),
        "quali_pre_top10_rate": g["QualiPos"].apply(lambda s: float((pd.to_numeric(s, errors="coerce") <= 10).mean()) if s.notna().any() else np.nan),
        "quali_pre_notime_or_pen_rate": g["NoTimeOrPen"].mean(),
        "quali_pre_points_mean": (g["Points"].mean() if "Points" in hist.columns else pd.Series(dtype=float)),
    }).reset_index()

    # last seen (макс. год/раунд)
    last = g[["year","round"]].max().reset_index().rename(columns={
        "year": "quali_pre_last_seen_year",
        "round": "quali_pre_last_seen_round"
    })

    out = out.merge(last, on="Driver", how="left")

    # типы
    for c in ("quali_pre_hist_n","quali_pre_last_seen_year","quali_pre_last_seen_round","quali_pre_bestpos_min"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    return out

# -------------------- ядро --------------------

@dataclass
class Options:
    max_lookback: int = 30  # сколько прошлых этапов искать (с запасом)

def _featurize_impl(
    raw_dir: Union[str, Path],
    races_df: Optional[pd.DataFrame],
    year: int,
    rnd: int,
    roster: Optional[Sequence[str]] = None,
    options: Optional[Options] = None,
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    opt = options or Options()

    # 1) базовый список пилотов
    if roster is not None:
        drivers = [str(x) for x in roster]
    else:
        cur = _current_roster(raw_dir, year, rnd)
        if cur.empty:
            past = _list_past(races_df, raw_dir, year, rnd, max_lookback=opt.max_lookback)
            cur = _last_known_roster_from_quali(raw_dir, past)
        drivers = cur["Driver"].astype(str).tolist() if not cur.empty else []

    if not drivers:
        # Нечего возвращать
        return pd.DataFrame()

    base = pd.DataFrame({"Driver": drivers})

    # 2) прошлые этапы и сбор истории
    past = _list_past(races_df, raw_dir, year, rnd, max_lookback=opt.max_lookback)
    rows = []
    for (y, r) in past:
        t = _qualifying_table(raw_dir, y, r)
        if not t.empty:
            rows.append(t)
    hist = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["Driver"])

    # 3) агрегации и join
    if not hist.empty:
        agg = _agg_quali(hist)
        out = base.merge(agg, on="Driver", how="left")
    else:
        out = base
        # аккуратные дефолты при отсутствии истории
        out["quali_pre_hist_n"] = 0
        for c in ("quali_pre_pos_p50","quali_pre_pos_iqr","quali_pre_top10_rate",
                  "quali_pre_notime_or_pen_rate","quali_pre_points_mean"):
            out[c] = np.nan
        out["quali_pre_bestpos_min"] = pd.Series([pd.NA] * len(out), dtype="Int64")
        out["quali_pre_last_seen_year"] = pd.Series([pd.NA] * len(out), dtype="Int64")
        out["quali_pre_last_seen_round"] = pd.Series([pd.NA] * len(out), dtype="Int64")

    return out

# -------------------- публичный API --------------------

def featurize(*args, **kwargs) -> pd.DataFrame:
    """
    Режимы:
      - featurize(ctx): ctx должен содержать raw_dir, year/season и rnd/round; опц.: races_df/races_csv, roster/drivers, max_lookback
      - featurize(raw_dir, races_df, year, rnd, roster=None, options=None)
    """
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
            raise TypeError("quali_priors_pre.featurize(ctx): укажите 'year'/'season' и 'rnd'/'round' (или build_id вида 'YYYY_R').")

        roster = get("roster", get("drivers", None))
        if isinstance(roster, pd.DataFrame):
            c = _detect_col(roster, DRIVER_COLS)
            roster = roster[c].astype(str).dropna().unique().tolist() if c else None
        elif isinstance(roster, (list, tuple, set, pd.Series)):
            roster = [str(x) for x in roster]
        else:
            roster = None

        max_lb = int(get("max_lookback", 30))
        return _featurize_impl(raw_dir, races_df, int(year), int(rnd), roster=roster, options=Options(max_lookback=max_lb))

    # старая сигнатура
    if len(args) >= 4 and isinstance(args[0], (str, Path)):
        raw_dir, races_df, year, rnd = args[:4]
        roster = args[4] if len(args) >= 5 else kwargs.get("roster")
        options = args[5] if len(args) >= 6 else kwargs.get("options")
        return _featurize_impl(Path(raw_dir), races_df, int(year), int(rnd), roster=roster, options=options or Options())

    raise TypeError("quali_priors_pre.featurize: ожидается featurize(ctx) или featurize(raw_dir, races_df, year, rnd, ...).")

# -------------------- CLI для отладки --------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Quali priors (pre-race)")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--races-csv", default=None)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--roster-csv", default=None, help="CSV с колонкой 'Driver' (опционально)")
    ap.add_argument("--max-lookback", type=int, default=30)
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
    print(f"Saved quali priors to {args.out} (rows={len(out)})")
