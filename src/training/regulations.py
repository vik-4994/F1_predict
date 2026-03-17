from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegulationEra:
    name: str
    year_from: int
    year_to: int
    default_weight: float


REGULATION_ERAS: tuple[RegulationEra, ...] = (
    RegulationEra("pre_2000", 0, 1999, 0.03),
    RegulationEra("v10_groove", 2000, 2004, 0.08),
    RegulationEra("v8_refuel", 2005, 2008, 0.12),
    RegulationEra("aero_kers", 2009, 2013, 0.18),
    RegulationEra("hybrid_v6_initial", 2014, 2016, 0.30),
    RegulationEra("wide_aero_hybrid", 2017, 2021, 0.50),
    RegulationEra("ground_effect", 2022, 2025, 0.75),
    RegulationEra("next_gen_2026", 2026, 9999, 1.00),
)


def default_era_weights() -> Dict[str, float]:
    return {era.name: float(era.default_weight) for era in REGULATION_ERAS}


def regulation_era_for_year(year: int) -> str:
    y = int(year)
    for era in REGULATION_ERAS:
        if era.year_from <= y <= era.year_to:
            return era.name
    return REGULATION_ERAS[-1].name


def regulation_era_series(years: Sequence[object] | pd.Series) -> pd.Series:
    year_series = pd.to_numeric(pd.Series(years), errors="coerce")
    labels = [regulation_era_for_year(int(y)) if pd.notna(y) else np.nan for y in year_series.tolist()]
    return pd.Series(labels, index=year_series.index, dtype="object")


def _normalize_weights(weights: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    if not weights:
        return {}
    vals = np.asarray(list(weights.values()), dtype=np.float64)
    mean_val = float(vals.mean()) if vals.size else 1.0
    if not np.isfinite(mean_val) or mean_val <= 0:
        mean_val = 1.0
    return {race: float(weight / mean_val) for race, weight in weights.items()}


def regulation_era_race_weights(
    df: pd.DataFrame,
    *,
    era_weights: Dict[str, float] | None = None,
    normalize: bool = True,
) -> Dict[Tuple[int, int], float]:
    if df is None or df.empty or "year" not in df.columns or "round" not in df.columns:
        return {}
    mapping = default_era_weights()
    mapping.update({str(k): float(v) for k, v in (era_weights or {}).items()})
    races = (
        df[["year", "round"]]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values(["year", "round"])
    )
    out: Dict[Tuple[int, int], float] = {}
    for row in races.itertuples(index=False):
        era = regulation_era_for_year(int(row.year))
        out[(int(row.year), int(row.round))] = float(mapping.get(era, 1.0))
    return _normalize_weights(out) if normalize else out


def combine_race_weight_maps(
    *maps: Dict[Tuple[int, int], float],
    normalize: bool = True,
) -> Dict[Tuple[int, int], float]:
    merged: Dict[Tuple[int, int], float] = {}
    keys: set[Tuple[int, int]] = set()
    for weight_map in maps:
        keys.update(weight_map.keys())
    for race in keys:
        value = 1.0
        for weight_map in maps:
            value *= float(weight_map.get(race, 1.0))
        merged[race] = float(value)
    return _normalize_weights(merged) if normalize else merged


def summarize_era_weights(
    df: pd.DataFrame,
    race_weights: Dict[Tuple[int, int], float],
) -> List[Dict[str, float | int | str]]:
    if df is None or df.empty or "year" not in df.columns or "round" not in df.columns:
        return []
    races = (
        df[["year", "round"]]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values(["year", "round"])
    )
    if races.empty:
        return []
    races["era"] = regulation_era_series(races["year"]).astype(str)
    races["race_weight"] = [
        float(race_weights.get((int(row.year), int(row.round)), 1.0))
        for row in races.itertuples(index=False)
    ]
    out: List[Dict[str, float | int | str]] = []
    for era, group in races.groupby("era", sort=False):
        vals = group["race_weight"].to_numpy(dtype=float)
        out.append(
            {
                "era": str(era),
                "races": int(len(group)),
                "weight_mean": float(np.mean(vals)) if vals.size else 1.0,
                "weight_min": float(np.min(vals)) if vals.size else 1.0,
                "weight_max": float(np.max(vals)) if vals.size else 1.0,
            }
        )
    return out


__all__ = [
    "RegulationEra",
    "REGULATION_ERAS",
    "default_era_weights",
    "regulation_era_for_year",
    "regulation_era_series",
    "regulation_era_race_weights",
    "combine_race_weight_maps",
    "summarize_era_weights",
]
