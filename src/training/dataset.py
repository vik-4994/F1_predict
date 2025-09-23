# FILE: src/training/dataset.py
from __future__ import annotations

from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .featureset import select_feature_cols, build_matrix

KEY = ["Driver", "year", "round"]


def _finish_pos_eff(df: pd.DataFrame, dnf_position: int = 21) -> pd.Series:
    """DNF -> dnf_position, иначе числовая finish_position."""
    pos = pd.to_numeric(df.get("finish_position"), errors="coerce")
    status = df.get("Status")
    if status is not None:
        status = status.astype(str)
        finished = status.str.contains("Finished", case=False, na=False) | status.str.contains("Plus", case=False, na=False)
        return pd.Series(np.where(finished, pos, float(dnf_position)), index=df.index)
    return pos


class RaceListDataset(Dataset):
    """
    Один элемент = одна гонка.
    На вход: таблица с фичами и таргетами (inner-join по Driver/year/round), желательно с finish_pos_eff.
    Если finish_pos_eff нет — посчитаем локально.
    """
    def __init__(
        self,
        df_joined: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        dnf_position: int = 21,
    ):
        if df_joined is None or df_joined.empty:
            raise ValueError("RaceListDataset: empty input DataFrame")

        df = df_joined.copy()

        # ключи к строковым/интовым типам
        df["Driver"] = df["Driver"].astype(str)
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
        df["round"] = pd.to_numeric(df["round"], errors="coerce").astype(int)

        # таргет: эффективная позиция
        if "finish_pos_eff" not in df.columns:
            df["finish_pos_eff"] = _finish_pos_eff(df, dnf_position=dnf_position)

        # выберем фичи
        if feature_cols is None:
            feature_cols = select_feature_cols(df)
        self.feature_cols: List[str] = list(feature_cols)

        # валидные строки (все фичи существуют и хотя бы одна не NaN)
        keep_mask = np.ones(len(df), dtype=bool)
        for c in self.feature_cols:
            if c not in df.columns:
                df[c] = np.nan  # добавим отсутствующие (заполнит скейлер потом)
        # выкидывать ничего не будем здесь; импьютация будет в скейлере

        df = df.loc[keep_mask]

        # группировка по гонкам
        groups: List[pd.DataFrame] = []
        for (y, r), g in df.groupby(["year", "round"], sort=True):
            g = g.sort_values("Driver").reset_index(drop=True)
            # если в гонке <2 пилотов, в обучении смысла нет — пропустим
            if g.shape[0] < 2:
                continue
            groups.append(g)
        if not groups:
            raise ValueError("RaceListDataset: no races with >=2 drivers")

        self.groups = groups

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> Dict:
        g = self.groups[idx]
        # матрица признаков
        X, _ = build_matrix(g, self.feature_cols)  # [N,F]
        # наблюдаемый порядок (победитель первым)
        order = np.argsort(g["finish_pos_eff"].to_numpy(dtype=np.float32))  # [N]
        item = {
            "X": torch.from_numpy(X.astype(np.float32)),                   # [N,F]
            "order": torch.from_numpy(order.astype(np.int64)),             # [N]
            "drivers": g["Driver"].tolist(),
            "year": int(g["year"].iloc[0]),
            "round": int(g["round"].iloc[0]),
        }
        return item

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
