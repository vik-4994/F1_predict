                         
from __future__ import annotations

from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .featureset import (
    select_feature_cols,
    build_matrix,
    transform_with_scaler_df,
    FeatureScaler,
)
from .outcomes import DSQ_ID, FINISH_ID, outcome_id_series, outcome_label_series

KEY = ["Driver", "year", "round"]


def _finish_pos_eff(df: pd.DataFrame, dnf_position: int = 21, dsq_position: int = 25) -> pd.Series:
    """DNF -> dnf_position, иначе числовая finish_position."""
    pos = pd.to_numeric(df.get("finish_position"), errors="coerce")
    outcome_ids = outcome_id_series(df.get("Status"), pos)
    eff = np.where(outcome_ids.eq(FINISH_ID), pos, float(dnf_position))
    eff = np.where(outcome_ids.eq(DSQ_ID), float(dsq_position), eff)
    if "Status" in df.columns:
        return pd.Series(eff, index=df.index)
    return pos


class RaceListDataset(Dataset):
    """
    Один элемент датасета = одна гонка.
    На вход: объединённая таблица с фичами и таргетами (inner-join по Driver/year/round).
    Если 'finish_pos_eff' нет — посчитаем локально из finish_position/Status.
    При наличии scaler: применяем тот же препроцесс, что и на инференсе.
    """

    def __init__(
        self,
        df_joined: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        dnf_position: int = 21,
        dsq_position: int = 25,
        scaler: Optional[FeatureScaler] = None,
        min_drivers_per_race: int = 2,
        race_weights: Optional[Dict[Tuple[int, int], float]] = None,
    ):
        if df_joined is None or df_joined.empty:
            raise ValueError("RaceListDataset: empty input DataFrame")

        df = df_joined.copy()

                                 
        if "Driver" not in df.columns:
            raise ValueError("RaceListDataset: 'Driver' column is required")
        df["Driver"] = df["Driver"].astype(str)
        for k in ("year", "round"):
            if k not in df.columns:
                raise ValueError(f"RaceListDataset: '{k}' column is required")
            df[k] = pd.to_numeric(df[k], errors="coerce").astype(int)

                                                        
        df = df.drop_duplicates(subset=KEY, keep="last").reset_index(drop=True)

                                     
        if "finish_pos_eff" not in df.columns:
            df["finish_pos_eff"] = _finish_pos_eff(df, dnf_position=dnf_position, dsq_position=dsq_position)
        if "result_outcome" not in df.columns:
            df["result_outcome"] = outcome_label_series(df.get("Status"), df.get("finish_position"))
        if "outcome_id" not in df.columns:
            df["outcome_id"] = outcome_id_series(df.get("Status"), df.get("finish_position"))

                    
        if feature_cols is None:
            feature_cols = select_feature_cols(df)
        else:
                                                                                   
            feature_cols = [c for c in feature_cols if c not in {"finish_pos_eff", "outcome_id"}]

                                                                   
        for c in feature_cols:
            if c not in df.columns:
                df[c] = np.nan

        self.feature_cols: List[str] = list(feature_cols)
        self.scaler: Optional[FeatureScaler] = scaler
        self.race_weights: Dict[Tuple[int, int], float] = {
            (int(y), int(r)): float(w)
            for (y, r), w in (race_weights or {}).items()
        }

                                                  
        groups: List[pd.DataFrame] = []
        for (y, r), g in df.groupby(["year", "round"], sort=True):
            g = g.sort_values("Driver").reset_index(drop=True)
            if g.shape[0] >= int(min_drivers_per_race):
                groups.append(g)
        if not groups:
            raise ValueError(f"RaceListDataset: no races with >={min_drivers_per_race} drivers")

        self.groups = groups

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> Dict:
        g = self.groups[idx]

                           
        if self.scaler is None:
            X, _ = build_matrix(g, self.feature_cols)                       
        else:
            X = transform_with_scaler_df(g, self.feature_cols, self.scaler, as_array=True)                

                                                 
        order = np.argsort(g["finish_pos_eff"].to_numpy(dtype=np.float32))       
        finish_pos = pd.to_numeric(g.get("finish_position"), errors="coerce")
        outcome_ids = pd.to_numeric(g.get("outcome_id"), errors="coerce").fillna(1).astype(int)
        finish_mask = outcome_ids.eq(FINISH_ID) & finish_pos.notna()
        finish_idx = np.flatnonzero(finish_mask.to_numpy(dtype=bool))
        finish_order = np.argsort(finish_pos.loc[finish_mask].to_numpy(dtype=np.float32))

        return {
            "X": torch.from_numpy(X.astype(np.float32)),         
            "order": torch.from_numpy(order.astype(np.int64)),       
            "finish_idx": torch.from_numpy(finish_idx.astype(np.int64)),
            "finish_order": torch.from_numpy(finish_order.astype(np.int64)),
            "status_target": torch.from_numpy(outcome_ids.to_numpy(dtype=np.int64)),
            "drivers": g["Driver"].tolist(),
            "year": int(g["year"].iloc[0]),
            "round": int(g["round"].iloc[0]),
            "race_weight": float(
                self.race_weights.get((int(g["year"].iloc[0]), int(g["round"].iloc[0])), 1.0)
            ),
        }

                       
    @property
    def n_features(self) -> int:
        return len(self.feature_cols)

    def races(self) -> List[Tuple[int, int]]:
        return [(int(g["year"].iloc[0]), int(g["round"].iloc[0])) for g in self.groups]


def collate_races(batch: List[Dict]) -> List[Dict]:
    """
    Коллейтор “как есть”: оставляем список элементов (каждый — отдельная гонка).
    Удобно для list-wise лосса.
    """
    return batch


__all__ = ["RaceListDataset", "collate_races"]
