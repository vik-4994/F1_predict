                               
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from .featureset import sanitize_frame_columns
from .outcomes import DSQ_ID, FINISH_ID, outcome_id_series, outcome_label_series


KEY = ["Driver", "year", "round"]


                                    

def _read_table(path: Path) -> pd.DataFrame:
    """Read parquet or csv; return empty DF if file missing."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def load_all(features_path: Path, targets_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load features and targets tables."""
    F = sanitize_frame_columns(_read_table(features_path))
    T = sanitize_frame_columns(_read_table(targets_path))
                    
    for df in (F, T):
        if "Driver" in df.columns:
            df["Driver"] = df["Driver"].astype(str)
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        if "round" in df.columns:
            df["round"] = pd.to_numeric(df["round"], errors="coerce").astype("Int64")
    return F, T


                                      

def _finish_pos_eff(
    tgt: pd.DataFrame,
    dnf_position: int = 21,
    dsq_position: int = 25,
) -> pd.Series:
    """
    Compute effective finishing position:
      - If 'Status' exists: any status not containing 'Finished' or 'Plus' -> DNF -> dnf_position.
      - Else: just numeric 'finish_position'.
    """
    pos = pd.to_numeric(tgt.get("finish_position"), errors="coerce")
    status = tgt.get("Status", pd.Series(index=tgt.index, dtype="object"))
    outcome_ids = outcome_id_series(status, pos)
    eff = np.where(outcome_ids.eq(DSQ_ID), float(dsq_position), float(dnf_position))
    eff = np.where(outcome_ids.eq(FINISH_ID), pos, eff)
    if status is not None:
        return pd.Series(eff, index=tgt.index)
    return pos


def build_train_table(
    F: pd.DataFrame,
    T: pd.DataFrame,
    dnf_position: int = 21,
    dsq_position: int = 25,
) -> pd.DataFrame:
    """
    Inner-join features with targets on (Driver, year, round) and add:
      - finish_pos_eff
      - result_outcome
      - outcome_id
    Rows without targets are dropped (модель учится только на имеющихся финишах).
    """
    need_cols = set(KEY + ["finish_position", "Status"])
    Tsel = T[[c for c in T.columns if c in need_cols]].copy()
    df = F.merge(Tsel, on=KEY, how="inner")
    if df.empty:
        return df
    df["result_outcome"] = outcome_label_series(df.get("Status"), df.get("finish_position"))
    df["outcome_id"] = outcome_id_series(df.get("Status"), df.get("finish_position"))
    df["finish_pos_eff"] = _finish_pos_eff(df, dnf_position=dnf_position, dsq_position=dsq_position)
    return df


                                        

def races_list(df: pd.DataFrame) -> List[Tuple[int, int]]:
    """Sorted unique (year, round) from a DataFrame containing those columns."""
    if df.empty or "year" not in df.columns or "round" not in df.columns:
        return []
    rs = (df[["year", "round"]]
          .dropna()
          .astype(int)
          .drop_duplicates()
          .sort_values(["year", "round"]))
    return list(map(tuple, rs.to_numpy()))


def time_split(df: pd.DataFrame, val_last: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by races: last `val_last` races → validation, the rest → train.
    Accepts either features, targets, or the joined train table — split by (year, round).
    """
    rc = races_list(df)
    if not rc:
        raise ValueError("time_split: no (year, round) pairs found")
    if len(rc) <= val_last:
        raise ValueError(f"time_split: too few races ({len(rc)}) for val_last={val_last}")
    train_keys = set(rc[:-val_last])
    val_keys = set(rc[-val_last:])
    key_pairs = df[["year", "round"]].apply(lambda r: (int(r["year"]), int(r["round"])), axis=1)
    tr = df[key_pairs.isin(train_keys)].copy()
    va = df[key_pairs.isin(val_keys)].copy()
    return tr, va


                                           

def race_recency_weights(df: pd.DataFrame, half_life: float | None = None) -> Dict[Tuple[int, int], float]:
    """Return chronological race weights normalized to mean=1."""
    rc = races_list(df)
    if not rc:
        return {}
    hl = float(half_life) if half_life is not None else float("nan")
    if not np.isfinite(hl) or hl <= 0:
        return {race: 1.0 for race in rc}
    ages = np.arange(len(rc) - 1, -1, -1, dtype=float)
    decay = np.log(2.0) / hl
    weights = np.exp(-decay * ages)
    mean_weight = float(weights.mean()) if weights.size else 1.0
    if not np.isfinite(mean_weight) or mean_weight <= 0:
        mean_weight = 1.0
    weights = weights / mean_weight
    return {race: float(weight) for race, weight in zip(rc, weights)}


def group_by_race(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Return list of per-race DataFrames (sorted by (year, round))."""
    out = []
    for (y, r), g in df.groupby(["year", "round"], sort=True):
        out.append(g.reset_index(drop=True))
    return out


__all__ = [
    "load_all",
    "build_train_table",
    "time_split",
    "race_recency_weights",
    "races_list",
    "group_by_race",
]
