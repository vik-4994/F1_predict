from __future__ import annotations
"""
track_profile — статический и производный профиль трассы + простой кластер
(улучшенная версия)

Что нового относительно первой версии:
  • Больше базовых признаков: pit_loss, fuel_effect, power_vs_df_index, overtake_index.
  • Аккуратные нормировки и составные индексы (0..1), удобные для моделей.
  • Числовой идентификатор кластера (track_cluster_id) + one‑hot.
  • Возможность переопределять профиль из CSV (если есть файл data/static/track_profiles.csv).
  • Полностью leak‑safe: значения фиксированы для трассы и не зависят от текущего уик‑энда.

Выход: DF с одной строкой на пилота (ключ 'Driver'), т.е. признаки дублируются по всем пилотам
текущего заезда, чтобы модель могла умножать их на пилото‑/командные особенности через свои веса
или (в будущем) через интеракции.

Совместимость:
  • Сосуществует с track_onehot, его НЕ заменяет.
  • Желательно располагать модуль сразу после track_onehot в пайплайне фич.

CSV переопределение (необязательное):
  • Пути поиска: data/static/track_profiles.csv, <raw_dir>/track_profiles.csv
  • Ожидаемые колонки CSV (по названию slug или name):
      slug | name | straight_pct | fast_corner_ratio | lap_km | braking_ev_per_lap | drs_zones
      | tdeg_index | aero_df_index | pit_loss_s | fuel_effect_s_per10kg | cluster
    Любые отсутствующие колонки берутся из встроенного словаря/дефолтов.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

                                 
try:
    from ..utils import ensure_driver_index
    from ..track_onehot import (
        _event_slug_from_meta, _event_name_from_meta, _event_name_from_races_df,
        _ensure_driver_list, _slugify,
    )
    from .track_catalog import (
        TrackProfile, DEFAULT_PROFILE, TRACK_TO_PROFILE,
        normalize_track_slug
    )
except Exception:
                                        
    from utils import ensure_driver_index  # type: ignore
    @dataclass(frozen=True)
    class TrackProfile:  # type: ignore[no-redef]
        straight_pct: float
        fast_corner_ratio: float
        lap_km: float
        braking_ev_per_lap: int
        drs_zones: int
        tdeg_index: float
        aero_df_index: float
        pit_loss_s: float
        fuel_effect_s_per10kg: float

    def _slugify(s: str) -> str:  # type: ignore
        import re, unicodedata
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
        s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
        return re.sub(r"_+", "_", s)
    def _event_slug_from_meta(raw_dir: Path, year: int, rnd: int) -> Optional[str]:  # type: ignore
        return None
    def _event_name_from_meta(raw_dir: Path, year: int, rnd: int) -> Optional[str]:  # type: ignore
        return None
    def _event_name_from_races_df(races_df: pd.DataFrame, year: int, rnd: int) -> Optional[str]:  # type: ignore
        return None
    def _ensure_driver_list(ctx, raw_dir: Path, year: int, rnd: int):  # type: ignore
        return []
    TRACK_TO_PROFILE = {}
    def normalize_track_slug(slug: str) -> str:  # type: ignore
        return _slugify(slug)

                                                               

                            
TRACK_TO_CLUSTER: Dict[str, str] = {
    "belgian_grand_prix": "balanced",
    "italian_grand_prix": "low_df",
    "japanese_grand_prix": "high_df",
    "singapore_grand_prix": "high_df",
}

CLUSTERS = ("low_df", "balanced", "high_df")
CLUSTER_ID = {c: i for i, c in enumerate(CLUSTERS)}

DEFAULT_PROFILE = TrackProfile(
    straight_pct=0.55, fast_corner_ratio=0.50, lap_km=5.5, braking_ev_per_lap=7,
    drs_zones=2, tdeg_index=0.55, aero_df_index=0.55,
    pit_loss_s=19.0, fuel_effect_s_per10kg=0.21,
)

                                                 

def _event_slug_from_ctx(ctx) -> Optional[str]:
    if isinstance(ctx, dict):
        for k in ("track", "event", "circuit", "grand_prix", "race"):
            v = ctx.get(k)
            if isinstance(v, str) and v.strip():
                return _slugify(v)
    return None


def _detect_slug(raw_dir: Path, year: int, rnd: int, ctx) -> Optional[str]:
                             
    s = _event_slug_from_ctx(ctx)
    if s:
        return s
                                    
    s = _event_slug_from_meta(raw_dir, year, rnd)
    if s:
        return s
    for name in (f"results_{year}_{rnd}.csv", f"laps_{year}_{rnd}.csv"):
        p = raw_dir / name
        if p.exists():
            try:
                df = pd.read_csv(p, nrows=1)
            except Exception:
                continue
            for c in ("EventName", "Event", "GrandPrix", "RaceName", "raceName", "Name",
                      "Circuit", "CircuitName", "CircuitShortName", "Track", "Venue"):
                if c in df.columns and df[c].notna().any():
                    v = str(df[c].iloc[0]).strip()
                    if v:
                        return _slugify(v)
    return None


def _norm(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def _compose_derived(p: TrackProfile) -> Dict[str, float]:
    straight_n = _norm(p.straight_pct, 0.35, 0.75)
    fast_n    = _norm(p.fast_corner_ratio, 0.25, 0.75)
    brake_n   = _norm(p.braking_ev_per_lap, 4.0, 12.0)
    drs_n     = _norm(p.drs_zones, 0.0, 4.0)
    lap_n     = _norm(p.lap_km, 3.0, 7.5)
    tdeg_n    = _norm(p.tdeg_index, 0.2, 0.9)
    aero_n    = _norm(p.aero_df_index, 0.2, 0.9)

    power_vs_df = float(np.clip(straight_n - aero_n * 0.7, -1.0, 1.0))

    overtake_index = (
        0.35 * straight_n + 0.40 * brake_n + 0.25 * drs_n - 0.20 * fast_n
    )
    overtake_index = float(np.clip(overtake_index, 0.0, 1.0))

    pit_loss = p.pit_loss_s if p.pit_loss_s is not None else float(17.0 + 0.6 * p.lap_km + 0.8 * p.drs_zones)
    fuel_eff = p.fuel_effect_s_per10kg if p.fuel_effect_s_per10kg is not None else float(0.036 * p.lap_km + 0.005)

    return {
        "track_power_vs_df_index": power_vs_df,
        "track_overtake_index": overtake_index,
        "track_pit_loss_s": float(pit_loss),
        "track_fuel_effect_s_per10kg": float(fuel_eff),
        "track_straight_n": straight_n,
        "track_fast_corner_n": fast_n,
        "track_braking_n": brake_n,
        "track_drs_n": drs_n,
        "track_lap_n": lap_n,
        "track_tdeg_n": tdeg_n,
        "track_aero_df_n": aero_n,
    }


def _maybe_load_csv_profiles(raw_dir: Path) -> Tuple[Dict[str, TrackProfile], Dict[str, str]]:
    """Если есть CSV с профилями — дополняем/переопределяем встроенные данные."""
    def _coerce_row(row: dict) -> Tuple[str, TrackProfile, Optional[str]]:
        name = str(row.get("slug") or row.get("name") or "").strip()
        slug = _slugify(name) if name else None
        tp = TrackProfile(
            straight_pct=float(row.get("straight_pct", np.nan)) if pd.notna(row.get("straight_pct")) else DEFAULT_PROFILE.straight_pct,
            fast_corner_ratio=float(row.get("fast_corner_ratio", np.nan)) if pd.notna(row.get("fast_corner_ratio")) else DEFAULT_PROFILE.fast_corner_ratio,
            lap_km=float(row.get("lap_km", np.nan)) if pd.notna(row.get("lap_km")) else DEFAULT_PROFILE.lap_km,
            braking_ev_per_lap=float(row.get("braking_ev_per_lap", np.nan)) if pd.notna(row.get("braking_ev_per_lap")) else DEFAULT_PROFILE.braking_ev_per_lap,
            drs_zones=float(row.get("drs_zones", np.nan)) if pd.notna(row.get("drs_zones")) else DEFAULT_PROFILE.drs_zones,
            tdeg_index=float(row.get("tdeg_index", np.nan)) if pd.notna(row.get("tdeg_index")) else DEFAULT_PROFILE.tdeg_index,
            aero_df_index=float(row.get("aero_df_index", np.nan)) if pd.notna(row.get("aero_df_index")) else DEFAULT_PROFILE.aero_df_index,
            pit_loss_s=float(row.get("pit_loss_s", np.nan)) if pd.notna(row.get("pit_loss_s")) else None,
            fuel_effect_s_per10kg=float(row.get("fuel_effect_s_per10kg", np.nan)) if pd.notna(row.get("fuel_effect_s_per10kg")) else None,
        )
        cluster = row.get("cluster")
        cluster = str(cluster).strip().lower() if isinstance(cluster, str) else None
        if cluster not in CLUSTER_ID:
            cluster = None
        return slug, tp, cluster

    profiles = {}
    clusters = {}
    for cand in [Path("data/static/track_profiles.csv"), raw_dir / "track_profiles.csv"]:
        if not cand.exists():
            continue
        try:
            df = pd.read_csv(cand)
        except Exception:
            continue
        for _, r in df.iterrows():
            slug, tp, cl = _coerce_row(r.to_dict())
            if slug:
                profiles[slug] = tp
                if cl:
                    clusters[slug] = cl
    return profiles, clusters

                                               

def featurize(ctx: dict) -> pd.DataFrame:
    raw_dir = Path(ctx.get("raw_dir"))
    year = int(ctx.get("year"))
    rnd = int(ctx.get("round"))
    emit = str(ctx.get("track_emit", "full")).lower()
    emit_cluster = bool(ctx.get("track_emit_cluster", True))

    drivers = _ensure_driver_list(ctx, raw_dir, year, rnd)
    if not drivers:
        return pd.DataFrame()

    slug_raw = _detect_slug(raw_dir, year, rnd, ctx) or ctx.get("track") or ""
    slug = normalize_track_slug(slug_raw)

    csv_prof, csv_cl = _maybe_load_csv_profiles(raw_dir)

    p = csv_prof.get(slug) or TRACK_TO_PROFILE.get(slug) or DEFAULT_PROFILE
    cluster = csv_cl.get(slug) or TRACK_TO_CLUSTER.get(slug) or "balanced"

    base_full = {
        "track_slug": slug or "unknown",
        "track_straight_pct": float(p.straight_pct),
        "track_fast_corner_ratio": float(p.fast_corner_ratio),
        "track_lap_km": float(p.lap_km),
        "track_braking_ev_per_lap": float(p.braking_ev_per_lap),
        "track_drs_zones": float(p.drs_zones),
        "track_tdeg_index": float(p.tdeg_index),
        "track_aero_df_index": float(p.aero_df_index),
        "track_cluster": str(cluster),
        "track_cluster_id": float(CLUSTER_ID.get(cluster, 1)),
    }
    for k in CLUSTERS:
        base_full[f"track_cluster_{k}"] = float(cluster == k)
    base_full.update(_compose_derived(p))                                                                               

                      
    if emit == "full":
        base = dict(base_full)
    elif emit == "std":
        drop_n = {
            "track_straight_n","track_fast_corner_n","track_braking_n",
            "track_drs_n","track_lap_n","track_tdeg_n","track_aero_df_n",
        }
        base = {k: v for k, v in base_full.items() if k not in drop_n}
        base.pop("track_cluster_id", None)                                                                           
        base.pop("track_slug", None)                              
    elif emit == "min":
        keep = {
                                          
            "track_power_vs_df_index",                        
            "track_tdeg_index",                         
            "track_pit_loss_s",                             
            "track_overtake_index",                       
            "track_straight_pct",                        
            "track_lap_km",                              
        }
        base = {k: v for k, v in base_full.items() if k in keep}
                                 
        base.pop("track_slug", None)
        base.pop("track_cluster_id", None)
        base.pop("track_cluster", None)
    else:
        base = dict(base_full)

                               
    if emit_cluster:
        for k in CLUSTERS:
            base[f"track_cluster_{k}"] = float(cluster == k)
    else:
        for k in list(base.keys()):
            if k.startswith("track_cluster_"):
                base.pop(k, None)
        base.pop("track_cluster", None)
        base.pop("track_cluster_id", None)

    for k, v in list(base.items()):
        if isinstance(v, (int, float, np.floating)):
            base[k] = np.float32(v)

    out = ensure_driver_index(pd.Series(drivers), base)
    return out


__all__ = [
    "featurize",
    "TRACK_TO_PROFILE",
    "TRACK_TO_CLUSTER",
    "CLUSTERS",
    "CLUSTER_ID",
]
