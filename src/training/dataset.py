# src/training/dataset.py
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

KEY = ["Driver", "year", "round"]


def _finish_pos_eff(df: pd.DataFrame, dnf_position: int = 21) -> pd.Series:
    """DNF -> dnf_position, иначе числовая finish_position."""
    pos = pd.to_numeric(df.get("finish_position"), errors="coerce")
    status = df.get("Status")
    if status is not None:
        status = status.astype(str)
        finished = status.str.contains("Finished", case=False, na=False) | status.str.contains(
            "Plus", case=False, na=False
        )
        return pd.Series(np.where(finished, pos, float(dnf_position)), index=df.index)
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
        scaler: Optional[FeatureScaler] = None,
        min_drivers_per_race: int = 2,
    ):
        if df_joined is None or df_joined.empty:
            raise ValueError("RaceListDataset: empty input DataFrame")

        df = df_joined.copy()

        # нормализуем ключи типов
        if "Driver" not in df.columns:
            raise ValueError("RaceListDataset: 'Driver' column is required")
        df["Driver"] = df["Driver"].astype(str)
        for k in ("year", "round"):
            if k not in df.columns:
                raise ValueError(f"RaceListDataset: '{k}' column is required")
            df[k] = pd.to_numeric(df[k], errors="coerce").astype(int)

        # удалим дубли по ключу гонки/пилота (если есть)
        df = df.drop_duplicates(subset=KEY, keep="last").reset_index(drop=True)

        # таргет: эффективная позиция
        if "finish_pos_eff" not in df.columns:
            df["finish_pos_eff"] = _finish_pos_eff(df, dnf_position=dnf_position)

        # список фич
        if feature_cols is None:
            feature_cols = select_feature_cols(df)
        else:
            # защита от утечки: исключим finish_pos_eff, даже если передали вручную
            feature_cols = [c for c in feature_cols if c != "finish_pos_eff"]

        # добавим отсутствующие колонки (заполнит скейлер средними)
        for c in feature_cols:
            if c not in df.columns:
                df[c] = np.nan

        self.feature_cols: List[str] = list(feature_cols)
        self.scaler: Optional[FeatureScaler] = scaler

        # группировка по гонкам (минимум 2 пилота)
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

        # матрица признаков
        if self.scaler is None:
            X, _ = build_matrix(g, self.feature_cols)  # [N,F] без скейлинга
        else:
            X = transform_with_scaler_df(g, self.feature_cols, self.scaler, as_array=True)  # [N,F] scaled

        # наблюдаемый порядок (победитель первым)
        order = np.argsort(g["finish_pos_eff"].to_numpy(dtype=np.float32))  # [N]

        return {
            "X": torch.from_numpy(X.astype(np.float32)),  # [N,F]
            "order": torch.from_numpy(order.astype(np.int64)),  # [N]
            "drivers": g["Driver"].tolist(),
            "year": int(g["year"].iloc[0]),
            "round": int(g["round"].iloc[0]),
        }

    # удобные аксессоры
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
