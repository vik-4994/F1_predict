from __future__ import annotations
"""
history_form.py — leak-safe "driver form" перед гонкой

Ключевые свойства:
- НЕ требует races_df / races_csv. Список прошлых этапов определяется по файлам
  в raw_dir (results_YYYY_R.csv / laps_YYYY_R.csv).
- Работает через featurize(ctx). Дополнительно поддержана старая сигнатура
  featurize(raw_dir, races_df, year, rnd, ...), но races_df можно передать None.
- Не трогает целевой раунд: данные берём только из прошлых лет/раундов.
- Устойчив к разным именам колонок: Driver/driverRef/...; Team/Constructor/...

Выход (по одному ряду на пилота):
  Driver
  hist_pre_hist_n
  hist_pre_best10_pace_p50_s
  hist_pre_best10_pace_iqr_s
  hist_pre_clean_share_mean
  hist_pre_dnf_rate
  hist_pre_last_seen_year
  hist_pre_last_seen_round
  hist_pre_team_stability
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union
import re

import numpy as np
import pandas as pd

__all__ = ["featurize"]

# --------- настройки/константы ---------
DRIVER_COLS = ("Driver", "Abbreviation", "driverRef", "DriverRef", "DriverCode", "BroadcastName")
TEAM_COLS   = ("Team", "TeamName", "Constructor", "ConstructorName", "ConstructorTeam")
TIME_COLS   = ("LapTime_s", "LapTime", "LapTimeMs", "LapTimeSec", "LapTimeSeconds")
DNF_TOKENS  = (
    "DNF", "Accident", "Collision", "Engine", "Hydraulics", "Electrical",
    "Gearbox", "Transmission", "Suspension", "Brakes", "Puncture",
    "Overheating", "Mechanical", "Exhaust", "Clutch", "Wheel", "Driveshaft",
    "Steering", "Fuel", "Oil leak", "Water pressure", "Power Unit", "PU",
)

ROSTER_FILES = (
    "entrylist_{y}_{r}_Q.csv",
    "entrylist_{y}_{r}.csv",
    "results_{y}_{r}_Q.csv",  # для состава команд безопасно
)

@dataclass
class Options:
    max_lookback: int = 10  # сколько прошлых гонок использовать (в сумме, назад по времени)


# --------- утилиты ввода/поиска ---------
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

def _to_seconds(series: pd.Series) -> pd.Series:
    s = series.copy()
    # попытка распарсить как timedelta
    try:
        td = pd.to_timedelta(s, errors="coerce")
        if td.notna().any():
            sec = td.dt.total_seconds()
            if sec.notna().mean() > 0.5:
                return sec
    except Exception:
        pass
    # числовой фолбэк
    s = pd.to_numeric(s, errors="coerce")
    med = s.replace([np.inf, -np.inf], np.nan).median()
    if pd.notna(med) and med > 1e3:
        return s / 1000.0  # похоже на миллисекунды
    return s

def _get_any(ctx_obj: Union[dict, object], keys: Sequence[str], default=None):
    if isinstance(ctx_obj, dict):
        for k in keys:
            if k in ctx_obj:
                return ctx_obj[k]
        return default
    else:
        for k in keys:
            if hasattr(ctx_obj, k):
                return getattr(ctx_obj, k)
        return default

def _parse_build_from_text(txt: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    if not txt:
        return None, None
    m = re.search(r"(?P<y>\d{4})[^\d]+(?P<r>\d{1,2})", str(txt))
    if m:
        return int(m.group("y")), int(m.group("r"))
    return None, None

# --------- поиск прошлых гонок по файлам в raw_dir ---------
def _scan_available_rounds(raw_dir: Path) -> List[Tuple[int, int]]:
    rounds = set()
    for pat in ("results_*.csv", "laps_*.csv", "entrylist_*.csv", "results_*_Q.csv", "entrylist_*_Q.csv"):
        for p in raw_dir.glob(pat):
            m = re.search(r"(\d{4})[_-](\d{1,2})", p.stem)
            if m:
                y, r = int(m.group(1)), int(m.group(2))
                rounds.add((y, r))
    out = sorted(rounds)  # по возрастанию (year, round)
    return out

def _list_past_by_files(raw_dir: Path, year: int, rnd: int, max_lookback: int) -> List[Tuple[int, int]]:
    all_rr = _scan_available_rounds(raw_dir)
    past = [(y, r) for (y, r) in all_rr if (y < year) or (y == year and r < rnd)]
    past.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return past[:max_lookback]

# --------- состав/команды для целевого раунда (без утечек) ---------
def _load_roster_team_current(raw_dir: Path, year: int, rnd: int) -> pd.DataFrame:
    for pat in ROSTER_FILES:
        df = _read_csv(raw_dir / pat.format(y=year, r=rnd))
        if df.empty:
            continue
        dcol = _detect_col(df, DRIVER_COLS)
        if not dcol:
            continue
        drivers = df[dcol].astype(str)
        tcol = _detect_col(df, TEAM_COLS)
        teams = df[tcol].astype(str) if tcol else pd.Series([np.nan] * len(df))
        out = pd.DataFrame({"Driver": drivers, "Team": teams})
        out = out.dropna(subset=["Driver"]).drop_duplicates(subset=["Driver"])
        if not out.empty:
            return out[["Driver", "Team"]]
    return pd.DataFrame(columns=["Driver", "Team"])

def _last_known_team_map(raw_dir: Path, past: List[Tuple[int, int]]) -> pd.DataFrame:
    frames = []
    for (y, r) in past:
        df = _read_csv(raw_dir / f"results_{y}_{r}.csv")
        if df.empty:
            continue
        dcol = _detect_col(df, DRIVER_COLS)
        if not dcol:
            continue
        tcol = _detect_col(df, TEAM_COLS)
        sub = pd.DataFrame({"Driver": df[dcol].astype(str)})
        sub["Team"] = df[tcol].astype(str) if tcol else np.nan
        sub["year"], sub["round"] = y, r
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=["Driver", "Team"])
    hist = pd.concat(frames, ignore_index=True)
    hist = hist.sort_values(["Driver", "year", "round"])
    last = hist.groupby("Driver", as_index=False).tail(1)[["Driver", "Team"]]
    return last.reset_index(drop=True)

# --------- метрики круга (PACE) из laps ---------
def _best10_and_clean_for_race(raw_dir: Path, y: int, r: int) -> pd.DataFrame:
    p = raw_dir / f"laps_{y}_{r}.csv"
    df = _read_csv(p)
    if df.empty:
        return pd.DataFrame()

    dcol = _detect_col(df, DRIVER_COLS)
    if not dcol:
        return pd.DataFrame()
    if dcol != "Driver":
        df = df.rename(columns={dcol: "Driver"})

    tcol = next((c for c in TIME_COLS if c in df.columns), None)
    if not tcol:
        return pd.DataFrame()
    df["LapTime_s"] = _to_seconds(df[tcol])

    # исключаем пит-ин/аут, если есть флаги
    for flag_col in ("IsPitIn", "IsPitOut", "PitIn", "PitOut"):
        if flag_col not in df.columns:
            df[flag_col] = False
    df = df[(~df["IsPitIn"]) & (~df["IsPitOut"]) & (~df["PitIn"]) & (~df["PitOut"])].copy()

    rows = []
    for drv, sub in df.groupby("Driver"):
        s = pd.to_numeric(sub["LapTime_s"], errors="coerce").dropna()
        if s.empty:
            continue
        if len(s) >= 3:
            q1, q3 = s.quantile([0.25, 0.75])
            thr = float(q3 + 2 * (q3 - q1))
            clean_mask = s <= thr
            clean = s[clean_mask]
            clean_share = float(clean_mask.mean())
        else:
            clean = s
            clean_share = np.nan
        val10 = clean.nsmallest(10)
        best10_mean = float(val10.mean()) if len(val10) else np.nan
        rows.append({"Driver": drv, "best10_mean_s": best10_mean, "clean_share": clean_share})
    return pd.DataFrame(rows)

# --------- DNF по results ---------
def _dnf_table_for_race(raw_dir: Path, y: int, r: int) -> pd.DataFrame:
    p = raw_dir / f"results_{y}_{r}.csv"
    df = _read_csv(p)
    if df.empty:
        return pd.DataFrame()
    dcol = _detect_col(df, DRIVER_COLS)
    if not dcol:
        return pd.DataFrame()
    if dcol != "Driver":
        df = df.rename(columns={dcol: "Driver"})
    s_col = _detect_col(df, ("Status", "status", "ResultStatus", "Classified"))
    if not s_col:
        return pd.DataFrame({"Driver": df["Driver"], "dnf": np.nan})
    s = df[s_col].astype(str)
    if s_col.lower() == "classified":
        dnf = ~df[s_col].astype(bool)
    else:
        s_up = s.str.upper()
        dnf = s_up.str.contains("DNF", na=False)
        for tok in DNF_TOKENS:
            dnf = dnf | s.str.contains(tok, case=False, na=False)
        dnf = dnf | s_up.eq("NC")
    return pd.DataFrame({"Driver": df["Driver"], "dnf": dnf.astype(float)})

# --------- агрегации ---------
def _aggregate(pace: pd.DataFrame, dnfs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if pace is None or pace.empty:
        pace_agg = pd.DataFrame(columns=[
            "Driver",
            "hist_pre_hist_n",
            "hist_pre_best10_pace_p50_s",
            "hist_pre_best10_pace_iqr_s",
            "hist_pre_clean_share_mean",
            "hist_pre_last_seen_year",
            "hist_pre_last_seen_round",
        ])
    else:
        def _agg(g: pd.DataFrame) -> pd.Series:
            vals = pd.to_numeric(g["best10_mean_s"], errors="coerce").dropna()
            share = pd.to_numeric(g["clean_share"], errors="coerce")
            p50 = float(np.nanmedian(vals)) if len(vals) else np.nan
            q25 = np.nanpercentile(vals, 25) if len(vals) else np.nan
            q75 = np.nanpercentile(vals, 75) if len(vals) else np.nan
            iqr = float(q75 - q25) if np.isfinite(q75) and np.isfinite(q25) else np.nan
            cmean = float(share.mean()) if share.notna().any() else np.nan
            last = g.sort_values(["year", "round"]).tail(1)[["year", "round"]].iloc[0] if not g.empty else {"year": pd.NA, "round": pd.NA}
            return pd.Series({
                "hist_pre_hist_n": int(g.shape[0]),
                "hist_pre_best10_pace_p50_s": p50,
                "hist_pre_best10_pace_iqr_s": iqr,
                "hist_pre_clean_share_mean": cmean,
                "hist_pre_last_seen_year": int(last["year"]) if pd.notna(last["year"]) else pd.NA,
                "hist_pre_last_seen_round": int(last["round"]) if pd.notna(last["round"]) else pd.NA,
            })
        pace_agg = pace.groupby("Driver", dropna=False).apply(_agg).reset_index()

    if dnfs is None or dnfs.empty:
        dnf_agg = pd.DataFrame(columns=["Driver", "hist_pre_dnf_rate"])
    else:
        dnf_agg = (dnfs.groupby("Driver", dropna=False)["dnf"]
                        .apply(lambda s: float(pd.to_numeric(s, errors="coerce").mean()) if s.notna().any() else np.nan)
                        .rename("hist_pre_dnf_rate")
                        .reset_index())
    return pace_agg, dnf_agg

def _team_stability(raw_dir: Path, past: List[Tuple[int, int]], current: pd.DataFrame) -> pd.DataFrame:
    if current is None or current.empty:
        return pd.DataFrame(columns=["Driver", "hist_pre_team_stability"])
    merged = []
    for (y, r) in past:
        df = _read_csv(raw_dir / f"results_{y}_{r}.csv")
        if df.empty:
            continue
        dcol = _detect_col(df, DRIVER_COLS)
        if not dcol:
            continue
        tcol = _detect_col(df, TEAM_COLS)
        sub = pd.DataFrame({"Driver": df[dcol].astype(str)})
        sub["Team"] = df[tcol].astype(str) if tcol else np.nan
        merged.append(sub)
    if not merged:
        return pd.DataFrame(columns=["Driver", "hist_pre_team_stability"])
    past_team = pd.concat(merged, ignore_index=True)
    cur = current[["Driver", "Team"]].drop_duplicates("Driver")
    comp = past_team.merge(cur, on="Driver", how="left", suffixes=("_past", "_cur"))
    comp["eq"] = (comp["Team_past"].astype(str) == comp["Team_cur"].astype(str)).astype(float)
    stab = comp.groupby("Driver")["eq"].mean().rename("hist_pre_team_stability").reset_index()
    return stab

# --------- основная логика ---------
def _featurize_impl(
    raw_dir: Union[str, Path],
    year: int,
    rnd: int,
    roster: Optional[Sequence[str]] = None,
    options: Optional[Options] = None,
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    opt = options or Options()

    # прошлые гонки (по файлам)
    past = _list_past_by_files(raw_dir, year, rnd, opt.max_lookback)
    # «текущий» состав и команды (для сравнения, без утечек)
    current = _load_roster_team_current(raw_dir, year, rnd)

    # собираем по прошлым гонкам pace и DNF
    pace_rows, dnf_rows = [], []
    for (y, r) in past:
        pr = _best10_and_clean_for_race(raw_dir, y, r)
        if not pr.empty:
            pr["year"], pr["round"] = y, r
            pace_rows.append(pr)
        dr = _dnf_table_for_race(raw_dir, y, r)
        if not dr.empty:
            dr["year"], dr["round"] = y, r
            dnf_rows.append(dr)

    pace = pd.concat(pace_rows, ignore_index=True) if pace_rows else pd.DataFrame(columns=["Driver","best10_mean_s","clean_share","year","round"])
    dnfs = pd.concat(dnf_rows, ignore_index=True) if dnf_rows else pd.DataFrame(columns=["Driver","dnf","year","round"])

    pace_agg, dnf_agg = _aggregate(pace, dnfs)
    team_stab = _team_stability(raw_dir, past, current)

    # база по пилотам: либо current roster, либо объединение из pace/dnfs
    if roster:
        base = pd.DataFrame({"Driver": [str(x) for x in roster]}).drop_duplicates("Driver")
    elif current is not None and not current.empty:
        base = current[["Driver"]].drop_duplicates("Driver").copy()
    else:
        drivers = sorted(set(pace["Driver"].dropna().astype(str)).union(set(dnfs["Driver"].dropna().astype(str))))
        base = pd.DataFrame({"Driver": drivers})

    out = (base.merge(pace_agg, on="Driver", how="left")
                .merge(dnf_agg, on="Driver", how="left")
                .merge(team_stab, on="Driver", how="left"))

    # типы
    for c in ("hist_pre_hist_n", "hist_pre_last_seen_year", "hist_pre_last_seen_round"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")

    return out

# --------- публичный API ---------
def featurize(*args, **kwargs) -> pd.DataFrame:
    """
    Основной режим: featurize(ctx)
      ctx.raw_dir: обязательный путь к сырым csv
      ctx.year/season/yr или build_id 'YYYY_X'
      ctx.rnd/round/rd/race
      ctx.roster (опц.): список пилотов или DF с колонкой Driver/driverRef
      ctx.max_lookback (опц.): int

    Совместимость: featurize(raw_dir, races_df, year, rnd, roster=None, options=None)
      races_df допускается None — модуль и так сканирует файлы.
    """
    # Вариант 1: featurize(ctx)
    if len(args) == 1 and not kwargs and not isinstance(args[0], (str, Path)):
        ctx = args[0]
        raw_dir = _get_any(ctx, ["raw_dir", "raw", "rawPath", "raw_dir_path"], ".")
        # year / round из множества ключей или из build_id
        year = _get_any(ctx, ["year", "season", "yr", "y", "build_year"], None)
        rnd  = _get_any(ctx, ["rnd", "round", "rd", "race", "build_round"], None)
        if year is None or rnd is None:
            by, br = _parse_build_from_text(_get_any(ctx, ["build_id", "build", "label", "name"], None))
            year = year if year is not None else by
            rnd  = rnd  if rnd  is not None else br

        # roster (список/series/df)
        roster = None
        roster_src = _get_any(ctx, ["roster", "drivers", "entrylist"], None)
        if roster_src is not None:
            if isinstance(roster_src, pd.DataFrame):
                c = _detect_col(roster_src, DRIVER_COLS)
                if c:
                    roster = roster_src[c].astype(str).dropna().unique().tolist()
            elif isinstance(roster_src, (list, tuple, set, pd.Series)):
                roster = [str(x) for x in roster_src]

        max_lookback = _get_any(ctx, ["max_lookback", "history_lookback", "lookback"], 10)
        opt = Options(max_lookback=int(max_lookback))

        if year is None or rnd is None:
            # без year/rnd невозможно исключить целевой этап
            raise TypeError("history_form.featurize(ctx): укажите 'year'/'season' и 'rnd'/'round' (или build_id вида 'YYYY_R').")

        return _featurize_impl(Path(raw_dir), int(year), int(rnd), roster=roster, options=opt)

    # Вариант 2: старая сигнатура — races_df можно передать None
    # featurize(raw_dir, races_df, year, rnd, roster=None, options=None)
    if len(args) >= 4 and isinstance(args[0], (str, Path)):
        raw_dir, _races_df, year, rnd = args[:4]
        roster = args[4] if len(args) >= 5 else kwargs.get("roster")
        options = args[5] if len(args) >= 6 else kwargs.get("options")
        return _featurize_impl(Path(raw_dir), int(year), int(rnd), roster=roster, options=options or Options())

    # если что-то иное — лучше явно сообщить
    raise TypeError("history_form.featurize: ожидается featurize(ctx) или featurize(raw_dir, races_df, year, rnd, ...).")

# --------- CLI (на всякий случай) ---------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser("Leak-safe driver history form (pre-race)")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--roster-csv", default=None)
    ap.add_argument("--max-lookback", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    roster = None
    if args.roster_csv:
        r = _read_csv(Path(args.roster_csv))
        c = _detect_col(r, DRIVER_COLS)
        if c:
            roster = r[c].astype(str).dropna().unique().tolist()

    df = _featurize_impl(raw_dir, args.year, args.round, roster=roster, options=Options(max_lookback=args.max_lookback))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[history_form] saved: {args.out} (rows={len(df)})")
