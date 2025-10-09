from __future__ import annotations
"""
Driver × Track-Cluster Priors (leak-safe)

Генерирует приоры по ПИЛОТУ для выбранного типа трассы (кластера):
    • driver_trackc_pre_finish_p50
    • driver_trackc_pre_points_mean
    • driver_trackc_pre_top10_rate
    • driver_trackc_pre_grid_p50      (если доступна колонка грида)
    • driver_trackc_pre_retire_rate
    • driver_trackc_pre_hist_n        (кол-во учтённых гонок)

Особенности:
  • Полностью leak-safe: берём ТОЛЬКО гонки строго раньше текущей (по timestamp; при его отсутствии — по (year, round)).
  • Источник истории — сырые results_* в raw_dir (гибкий парсинг колонок). Можно заменить на свой сводный parquet
    без изменений интерфейса (см. _load_results).
  • История фильтруется по КЛАСТЕРУ трассы текущего уик-энда (low_df / balanced / high_df),
    кластер берём из track_profile.TRACK_TO_CLUSTER + slug детекция как в track_onehot.
  • Поддержка окна истории: ctx['history_window'] (int). По умолчанию 8 последних гонок по кластеру.
  • Мягкое сглаживание (empirical Bayes) на малых n: blend с кластерным бенчмарком.

Совместимость:
  • Возвращаем DF с одной строкой на пилота (ключ "Driver"), чтобы main-пайплайн смог join'ить по Driver.
  • Идёт ПОСЛЕ track_profile в списке фич (но может работать и без него — кластер вычисляется локально).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# локальные утилиты
try:
    from .utils import ensure_driver_index
    from .track_onehot import _slugify, _event_slug_from_meta, _ensure_driver_list
    from .track_profile import TRACK_TO_CLUSTER, CLUSTERS, CLUSTER_ID
except Exception:
    from utils import ensure_driver_index  # type: ignore
    def _slugify(s: str) -> str:  # type: ignore
        import re, unicodedata
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
        s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
        return re.sub(r"_+", "_", s)
    def _event_slug_from_meta(raw_dir: Path, year: int, rnd: int):  # type: ignore
        return None
    def _ensure_driver_list(ctx, raw_dir: Path, year: int, rnd: int):  # type: ignore
        return []
    TRACK_TO_CLUSTER = {}
    CLUSTERS = ("low_df", "balanced", "high_df")
    CLUSTER_ID = {c: i for i, c in enumerate(CLUSTERS)}

# --------------- helpers: IO & parsing ---------------

_CAND_DATE_COLS = [
    "EventStartTime", "SessionStart", "DateTime", "Date", "EventDate", "RaceStart",
]
_CAND_POS_COLS = ["FinishPosition", "Position", "Pos", "ResultPosition", "ResultOrder", "Finish"]
_CAND_POINTS_COLS = ["Points", "PTS", "points"]
_CAND_GRID_COLS = ["Grid", "GridPosition", "StartPos", "QualiGrid"]
_CAND_STATUS_COLS = ["Status", "ResultStatus", "FinishStatus"]
_CAND_EVENT_NAME_COLS = [
    "EventName", "Event", "GrandPrix", "RaceName", "raceName", "Name",
    "Circuit", "CircuitName", "CircuitShortName", "Track", "Venue",
]


def _parse_datetime_cols(df: pd.DataFrame) -> pd.Series:
    out = None
    for c in _CAND_DATE_COLS:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce")
            out = s if out is None else out.fillna(s)
    if out is None:
        out = pd.to_datetime(pd.Series([None] * len(df)))
    return out


def _to_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _coerce_pos(col: pd.Series) -> pd.Series:
    # позиция может содержать "DNF" и т.п., приводим к числу
    s = pd.to_numeric(col, errors="coerce")
    return s


def _load_results(raw_dir: Path) -> pd.DataFrame:
    """Подгружаем исторические результаты из raw_dir.
    Ищем CSV, в названии которых есть 'results'. Конкатенируем.
    Минимально требуемые поля после нормализации: Year, Round, Driver, Points, Pos, Grid?, Status?, EventName/slug.
    """
    raw_dir = Path(raw_dir)
    files = sorted([p for p in raw_dir.glob("**/*") if p.is_file() and "results" in p.name.lower() and p.suffix.lower() in (".csv", ".parquet")])
    dfs: List[pd.DataFrame] = []
    for p in files:
        try:
            if p.suffix.lower() == ".parquet":
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty:
            continue
        # нормализуем имена
        cols = {c: c for c in df.columns}
        # Year / Round
        if "Year" not in df.columns and "year" in df.columns:
            cols["year"] = "Year"
        if "Round" not in df.columns and "round" in df.columns:
            cols["round"] = "Round"
        # Driver
        if "Driver" not in df.columns:
            for c in ["DriverCode", "Code", "Abbreviation", "DriverId", "driver"]:
                if c in df.columns:
                    cols[c] = "Driver"
                    break
        df = df.rename(columns=cols)
        # позиции/очки/статус/грid
        pos_col = next((c for c in _CAND_POS_COLS if c in df.columns), None)
        pts_col = next((c for c in _CAND_POINTS_COLS if c in df.columns), None)
        grd_col = next((c for c in _CAND_GRID_COLS if c in df.columns), None)
        st_col  = next((c for c in _CAND_STATUS_COLS if c in df.columns), None)

        if "Year" not in df.columns or "Round" not in df.columns or "Driver" not in df.columns:
            continue

        out = pd.DataFrame({
            "Year": _to_int(df["Year"]),
            "Round": _to_int(df["Round"]),
            "Driver": df["Driver"].astype(str),
            "Points": pd.to_numeric(df[pts_col], errors="coerce") if pts_col else pd.Series([np.nan]*len(df)),
            "Pos": _coerce_pos(df[pos_col]) if pos_col else pd.Series([np.nan]*len(df)),
            "Grid": pd.to_numeric(df[grd_col], errors="coerce") if grd_col else pd.Series([np.nan]*len(df)),
            "Status": df[st_col].astype(str) if st_col else pd.Series([None]*len(df)),
        })

        # событие/slug
        ev_name = None
        for c in _CAND_EVENT_NAME_COLS:
            if c in df.columns and df[c].notna().any():
                ev_name = df[c].astype(str)
                break
        if ev_name is None:
            ev_name = pd.Series(["" for _ in range(len(df))])
        out["EventName"] = ev_name
        out["EventSlug"] = out["EventName"].map(lambda s: _slugify(str(s)))

        # время старта
        out["EventTime"] = _parse_datetime_cols(df)
        dfs.append(out)

    if not dfs:
        return pd.DataFrame()

    res = pd.concat(dfs, axis=0, ignore_index=True)
    # Уберём явный мусор
    res = res.dropna(subset=["Year", "Round", "Driver"], how="any")
    res["Year"] = res["Year"].astype(int)
    res["Round"] = res["Round"].astype(int)
    return res


def _current_event_slug_and_cluster(ctx: dict, raw_dir: Path, year: int, rnd: int) -> Tuple[str, str]:
    # try explicit track from ctx
    if isinstance(ctx, dict):
        for k in ("track", "event", "circuit", "grand_prix", "race"):
            v = ctx.get(k)
            if isinstance(v, str) and v.strip():
                slug = _slugify(v)
                cl = TRACK_TO_CLUSTER.get(slug, "balanced")
                return slug, cl
    # try meta
    slug = _event_slug_from_meta(raw_dir, year, rnd) or ""
    cl = TRACK_TO_CLUSTER.get(slug, "balanced")
    return slug, cl


# --------------- core ---------------

def _order_key(df: pd.DataFrame) -> pd.Series:
    # сначала пытаемся по времени, иначе Year*100 + Round
    if "EventTime" in df.columns and df["EventTime"].notna().any():
        t = pd.to_datetime(df["EventTime"], errors="coerce")
        if t.notna().any():
            return t
    return df["Year"] * 100 + df["Round"]


def _is_retired(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series([False] * 0)
    x = s.fillna("").str.lower()
    flags = [
        "ret", "dnf", "accident", "collision", "engine", "gearbox", "hydraul",
        "electrical", "suspension", "crash", "power unit", "overheating",
    ]
    mask = pd.Series(False, index=s.index)
    for f in flags:
        mask = mask | x.str.contains(f)
    return mask


def _eb_blend(emp: pd.Series, base: float, k: float, n: pd.Series) -> pd.Series:
    # shrinkage: (n*emp + k*base) / (n + k)
    return ((n.astype(float) * emp.fillna(base)) + (k * base)) / (n.astype(float) + k)


def featurize(ctx: dict) -> pd.DataFrame:
    raw_dir = Path(ctx.get("raw_dir"))
    year = int(ctx.get("year"))
    rnd = int(ctx.get("round"))
    hist_n = int(ctx.get("history_window", 8))  # по умолчанию 8 гонок

    drivers = _ensure_driver_list(ctx, raw_dir, year, rnd)
    if not drivers:
        return pd.DataFrame()

    # Текущий кластер
    cur_slug, cur_cluster = _current_event_slug_and_cluster(ctx, raw_dir, year, rnd)

    # Загружаем историю
    res = _load_results(raw_dir)
    if res.empty:
        # отдаём пустые, чтобы пайплайн не упал
        base = {c: np.float32(np.nan) for c in [
            "driver_trackc_pre_finish_p50",
            "driver_trackc_pre_points_mean",
            "driver_trackc_pre_top10_rate",
            "driver_trackc_pre_grid_p50",
            "driver_trackc_pre_retire_rate",
            "driver_trackc_pre_hist_n",
        ]}
        return ensure_driver_index(pd.Series(drivers), base)

    # пометим кластер каждой исторической гонки
    res["EventSlug"] = res["EventSlug"].fillna("")
    res["Cluster"] = res["EventSlug"].map(lambda s: TRACK_TO_CLUSTER.get(str(s), "balanced"))

    # ключ порядка и фильтр ТОЛЬКО прошлых событий
    res["OrderKey"] = _order_key(res)
    # current key
    cur_key = None
    # попробуем найти время текущего события из res (тот же год/раунд/slug)
    cur_rows = res[(res["Year"] == year) & (res["Round"] == rnd)]
    if not cur_rows.empty and cur_rows["EventTime"].notna().any():
        cur_key = pd.to_datetime(cur_rows["EventTime"], errors="coerce").max()
    if cur_key is None or pd.isna(cur_key):
        cur_key = year * 100 + rnd

    res = res[res["OrderKey"] < cur_key]

    # фильтр по кластеру текущей трассы
    res = res[res["Cluster"] == cur_cluster]

    if res.empty:
        # если нет истории по кластеру — вернём NaN, пусть модель/блендер работает дальше
        base = {c: np.float32(np.nan) for c in [
            "driver_trackc_pre_finish_p50",
            "driver_trackc_pre_points_mean",
            "driver_trackc_pre_top10_rate",
            "driver_trackc_pre_grid_p50",
            "driver_trackc_pre_retire_rate",
            "driver_trackc_pre_hist_n",
        ]}
        return ensure_driver_index(pd.Series(drivers), base)

    # оставим только нужные колонки и почистим типы
    res = res[["Driver", "Year", "Round", "Points", "Pos", "Grid", "Status", "OrderKey"]].copy()
    res["Points"] = pd.to_numeric(res["Points"], errors="coerce").fillna(0.0)
    res["Pos_num"] = pd.to_numeric(res["Pos"], errors="coerce")
    res["is_classified"] = res["Pos_num"].notna()
    res["top10"] = res["Pos_num"].between(1, 10, inclusive="both")
    res["retired"] = _is_retired(res["Status"]).astype(bool)

    # выберем последние hist_n гонок ДЛЯ КАЖДОГО PILOT в этом кластере
    res = res.sort_values(["Driver", "OrderKey"])  # стабильная сортировка
    res["rn"] = res.groupby("Driver").cumcount(ascending=True)
    # оставим только последние hist_n по каждому пилоту
    max_rn = res.groupby("Driver")["rn"].transform("max")
    res = res[(max_rn - res["rn"]) < hist_n]

    # агрегаты по пилоту
    agg = res.groupby("Driver").agg(
        hist_n=("Pos_num", lambda s: int(s.notna().sum())),
        finish_p50=("Pos_num", lambda s: float(np.nanpercentile(s, 50)) if s.notna().any() else np.nan),
        points_mean=("Points", "mean"),
        top10_rate=("top10", "mean"),
        grid_p50=("Grid", lambda s: float(np.nanpercentile(pd.to_numeric(s, errors="coerce"), 50)) if pd.to_numeric(s, errors="coerce").notna().any() else np.nan),
        retire_rate=("retired", "mean"),
    ).reset_index()

    # кластерные бенчмарки (по всем пилотам) для сглаживания
    # NB: берём бенч по тому же набору строк res
    base_finish = float(np.nanmedian(agg["finish_p50"])) if agg["finish_p50"].notna().any() else np.nan
    base_points = float(agg["points_mean"].mean()) if agg["points_mean"].notna().any() else 0.0
    base_top10  = float(agg["top10_rate"].mean()) if agg["top10_rate"].notna().any() else 0.0
    base_grid   = float(np.nanmedian(agg["grid_p50"])) if agg["grid_p50"].notna().any() else np.nan
    base_ret    = float(agg["retire_rate"].mean()) if agg["retire_rate"].notna().any() else 0.0

    n = agg["hist_n"].astype(float)
    # Коэффициенты сглаживания: чем меньше n, тем сильнее тянем к среднему кластера
    k_fin, k_pts, k_top, k_grid, k_ret = 3.0, 6.0, 6.0, 3.0, 6.0

    agg["finish_p50_blend"] = _eb_blend(agg["finish_p50"], base_finish, k_fin, n)
    agg["points_mean_blend"] = _eb_blend(agg["points_mean"], base_points, k_pts, n)
    agg["top10_rate_blend"] = _eb_blend(agg["top10_rate"], base_top10, k_top, n)
    agg["grid_p50_blend"] = _eb_blend(agg["grid_p50"], base_grid, k_grid, n)
    agg["retire_rate_blend"] = _eb_blend(agg["retire_rate"], base_ret, k_ret, n)

    # формируем выходные колонки
    out = agg[["Driver"]].copy()
    out["driver_trackc_pre_finish_p50"] = agg["finish_p50_blend"].astype("float32")
    out["driver_trackc_pre_points_mean"] = agg["points_mean_blend"].astype("float32")
    out["driver_trackc_pre_top10_rate"] = agg["top10_rate_blend"].astype("float32")
    out["driver_trackc_pre_grid_p50"] = agg["grid_p50_blend"].astype("float32")
    out["driver_trackc_pre_retire_rate"] = agg["retire_rate_blend"].astype("float32")
    out["driver_trackc_pre_hist_n"] = agg["hist_n"].astype("float32")

    # Размножаем по текущему списку водителей; те, у кого нет истории, получат NaN (или можно заполнить бенчмарком)
    out = ensure_driver_index(pd.Series(drivers), out.set_index("Driver").to_dict(orient="index"))

    return out


__all__ = ["featurize"]
