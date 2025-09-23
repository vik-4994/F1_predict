# FILE: src/training/data_io.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np


KEY = ["Driver", "year", "round"]


# ---------- low-level IO ----------

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
    F = _read_table(features_path)
    T = _read_table(targets_path)
    # normalize keys
    for df in (F, T):
        if "Driver" in df.columns:
            df["Driver"] = df["Driver"].astype(str)
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        if "round" in df.columns:
            df["round"] = pd.to_numeric(df["round"], errors="coerce").astype("Int64")
    return F, T


# ---------- targets / join ----------

def _finish_pos_eff(tgt: pd.DataFrame, dnf_position: int = 21) -> pd.Series:
    """
    Compute effective finishing position:
      - If 'Status' exists: any status not containing 'Finished' or 'Plus' -> DNF -> dnf_position.
      - Else: just numeric 'finish_position'.
    """
    pos = pd.to_numeric(tgt.get("finish_position"), errors="coerce")
    status = tgt.get("Status")
    if status is not None:
        status = status.astype(str)
        finished = status.str.contains("Finished", case=False, na=False) | status.str.contains("Plus", case=False, na=False)
        eff = np.where(finished, pos, float(dnf_position))
        return pd.Series(eff, index=tgt.index)
    return pos


def build_train_table(
    F: pd.DataFrame,
    T: pd.DataFrame,
    dnf_position: int = 21,
) -> pd.DataFrame:
    """
    Inner-join features with targets on (Driver, year, round) and add:
      - finish_pos_eff
    Rows without targets are dropped (модель учится только на имеющихся финишах).
    """
    need_cols = set(KEY + ["finish_position", "Status"])
    Tsel = T[[c for c in T.columns if c in need_cols]].copy()
    df = F.merge(Tsel, on=KEY, how="inner")
    if df.empty:
        return df
    df["finish_pos_eff"] = _finish_pos_eff(df, dnf_position=dnf_position)
    return df


# ---------- timeline & split ----------

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


# ---------- convenience helpers ----------

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
    "races_list",
    "group_by_race",
]
