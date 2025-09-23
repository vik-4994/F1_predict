# src/training/inference.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False
    nn = object  # type: ignore

from .featureset import (
    FeatureScaler,
    FeatureSpec,
    build_matrix as _build_matrix_core,
    load_feature_cols,
    select_feature_cols,
    transform_with_scaler_df,
)

# =====================================================================================
#                                        API
# =====================================================================================

@dataclass
class Artifacts:
    """Собранные артефакты для инференса."""
    model: Optional[nn.Module]
    scaler: FeatureScaler
    feature_cols: List[str]
    meta: Dict[str, Any]
    device: str = "cpu"


def _auto_device(device: Optional[str] = None) -> str:
    if device:
        return device
    if _HAS_TORCH and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_artifacts(
    artifacts_dir: Path | str,
    make_model: Optional[Callable[[int], nn.Module]] = None,
    device: Optional[str] = None,
) -> Artifacts:
    """Загружаем артефакты из директории.

    Ожидаем файлы (см. engine.save_checkpoint):
      - ranker.pt        (state_dict модели)
      - scaler.json      (параметры скейлера)
      - feature_cols.txt (порядок фич)
      - meta.json        (конфиг/служебная инфа)

    Если `make_model` не указан, пытаемся загрузить state_dict в простой MLP (если мета содержит
    in_dim/hidden/dropout). В противном случае можно передать свой билдер.
    """
    adir = Path(artifacts_dir)
    dev = _auto_device(device)

    # --- feature_cols ---
    fcols_path = adir / "feature_cols.txt"
    feature_cols = load_feature_cols(fcols_path)

    # --- scaler ---
    scal_path = adir / "scaler.json"
    scaler = FeatureScaler.load(scal_path)

    # --- meta ---
    meta_path = adir / "meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    # --- model ---
    mdl: Optional[nn.Module] = None
    pt_path = adir / "ranker.pt"

    if _HAS_TORCH and pt_path.exists():
        if make_model is None:
            # Пытаемся собрать минимальный MLP из меты
            in_dim = int(meta.get("in_dim", len(feature_cols)))
            hidden = list(map(int, meta.get("hidden", [256, 128])))
            dropout = float(meta.get("dropout", 0.1))
            make_model = lambda _: _DefaultMLP(in_dim, hidden, dropout)  # noqa: E731
        try:
            obj = torch.load(pt_path, map_location=dev)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Не удалось torch.load('{pt_path}') — {e}")

        if isinstance(obj, dict) and "state_dict" in obj:
            state = obj["state_dict"]
            mdl = make_model(len(feature_cols))
            mdl.to(dev)
            mdl.load_state_dict(state, strict=False)
        elif isinstance(obj, dict):
            # возможно, это просто state_dict
            mdl = make_model(len(feature_cols))
            mdl.to(dev)
            mdl.load_state_dict(obj, strict=False)
        else:
            # torch.save(model) — редкий случай; пробуем напрямую
            mdl = obj
            mdl.to(dev)
        if hasattr(mdl, "eval"):
            mdl.eval()
    else:
        mdl = None  # позволим работать без торча/модели (например, только нормализация и отладка)

    return Artifacts(model=mdl, scaler=schk(scaler), feature_cols=feature_cols, meta=meta, device=dev)


# =====================================================================================
#                              Модель по умолчанию (MLP)
# =====================================================================================

class _DefaultMLP(nn.Module):
    """Простой MLP-скорер на случай, если не передали make_model.
    Архитектура: ReLU -> Dropout между слоями, выход 1.
    """
    def __init__(self, in_dim: int, hidden: Sequence[int], dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, D) -> (N,)
        out = self.net(x)
        return out.squeeze(-1)


# =====================================================================================
#                             Построение матрицы/симуляций
# =====================================================================================

def build_matrix(df: pd.DataFrame, artifacts: Artifacts, as_array: bool = True):
    """Удобная обёртка над featureset.build_matrix + transform.
    Возвращает (Xs, meta_dict) — где Xs уже отскейлен.
    """
    X, meta = _build_matrix_core(df, artifacts.feature_cols)
    Xs = transform_with_scaler_df(df, artifacts.feature_cols, artifacts.scaler, as_array=True)
    return (Xs if as_array else pd.DataFrame(Xs, columns=artifacts.feature_cols, index=df.index)), meta


def build_sim_frame(
    df_base: pd.DataFrame,
    global_overrides: Optional[Dict[str, Any]] = None,
    per_driver_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    driver_col: str = "Driver",
) -> pd.DataFrame:
    """Собираем датафрейм для симуляции на основе `df_base` и оверрайдов.

    Примеры:
      - global_overrides={"track_is_Baku":1, "temp_air":26}
      - per_driver_overrides={"VER": {"grid": 1}, "NOR": {"grid": 2}}
    """
    df = df_base.copy()
    if global_overrides:
        for k, v in global_overrides.items():
            if k in df.columns:
                df[k] = v
    if per_driver_overrides and driver_col in df.columns:
        idx_by_driver = df.groupby(driver_col).indices
        for drv, repl in per_driver_overrides.items():
            idx = idx_by_driver.get(drv)
            if idx is None:
                continue
            for k, v in repl.items():
                if k in df.columns:
                    df.loc[df.index[idx], k] = v
    return df


# =====================================================================================
#                              Инференс: скоринг/ранжирование
# =====================================================================================

def apply_temperature(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    t = float(temperature)
    if t <= 0:
        t = 1.0
    return scores / t


def _softmax_stable(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    m = np.max(x)
    z = np.exp(x - m)
    s = z.sum()
    return (z / s).astype(np.float32)


def predict_scores(
    artifacts: Artifacts,
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Возвращает сырые `scores` формы [N]. Если модели нет — нули.
    Использует порядок колонок из artifacts.feature_cols по умолчанию.
    """
    fcols = list(feature_cols or artifacts.feature_cols)
    Xs = transform_with_scaler_df(df, fcols, artifacts.scaler, as_array=True)

    if not (_HAS_TORCH and artifacts.model is not None):
        return np.zeros(Xs.shape[0], dtype=np.float32)

    x = torch.from_numpy(Xs).to(artifacts.device)
    with torch.no_grad():
        out = artifacts.model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
    s = out.detach().float().cpu().numpy().reshape(-1)
    return s


def attach_win_probs(
    df: pd.DataFrame,
    scores: np.ndarray,
    by: Sequence[str] = ("year", "round"),
    temperature: float = 1.0,
    prob_col: str = "p_win",
) -> pd.DataFrame:
    """Добавляет столбец вероятностей победы по группам (гонкам) через softmax(scores/T)."""
    out = df.copy()
    out["score"] = scores

    if by is None:
        probs = _softmax_stable(apply_temperature(scores, temperature))
        out[prob_col] = probs
        return out

    # по гонкам
    out[prob_col] = 0.0
    for _, idx in out.groupby(list(by)).indices.items():
        sc = out.loc[idx, "score"].to_numpy(dtype=np.float32)
        probs = _softmax_stable(apply_temperature(sc, temperature))
        out.loc[idx, prob_col] = probs
    return out


def make_ranking_df(
    df: pd.DataFrame,
    by: Sequence[str] = ("year", "round"),
    score_col: str = "score",
    prob_col: str = "p_win",
    ascending: bool = False,
) -> pd.DataFrame:
    """Возвращает ранжированный датафрейм внутри каждой гонки.
    По умолчанию больший score → лучше (descending). Если нужно наоборот, установите ascending=True.
    """
    cols = list(df.columns)
    key_cols = [c for c in ["Driver", "Team", "year", "round"] if c in cols]
    order_cols = list(by) if by else []
    rank_col = "rank"

    def _rank_one(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(score_col, ascending=ascending, kind="mergesort").copy()
        g[rank_col] = np.arange(1, len(g) + 1, dtype=np.int32)
        return g

    if by is None:
        out = _rank_one(df)
    else:
        out = df.groupby(list(by), group_keys=False).apply(_rank_one)

    # лёгкая перестановка колонок
    lead = [*order_cols, *key_cols]
    trail = [c for c in df.columns if c not in lead and c not in {rank_col}]
    out = out[[*lead, rank_col, *trail]].reset_index(drop=True)
    return out


def predict_custom(
    artifacts: Artifacts,
    features_df: pd.DataFrame,
    temperature: Optional[float] = None,
    by: Sequence[str] = ("year", "round"),
    include_probs: bool = True,
    ascending: bool = False,
) -> pd.DataFrame:
    """Высокоуровневый пайплайн: скоринг → вероятности → ранжирование.

    - temperature: если None, берём из meta["temperature"] или 1.0.
    - by: группировка для softmax/ранжирования (по умолчанию по гонкам).
    - ascending: если True — меньший score лучше.
    """
    temp = float(temperature if temperature is not None else artifacts.meta.get("temperature", 1.0))

    s = predict_scores(artifacts, features_df)
    df_sc = attach_win_probs(features_df, s, by=by, temperature=temp) if include_probs else (
        features_df.assign(score=s)
    )
    rank_df = make_ranking_df(df_sc, by=by, ascending=ascending)
    return rank_df


# =====================================================================================
#                                    Раннер
# =====================================================================================

class InferenceRunner:
    """Упрощённый раннер для инференса: хранит артефакты и предоставляет удобные методы."""
    def __init__(self, artifacts: Artifacts):
        self.artifacts = artifacts

    @classmethod
    def from_dir(cls, artifacts_dir: Path | str, make_model: Optional[Callable[[int], nn.Module]] = None, device: Optional[str] = None):
        arts = load_artifacts(artifacts_dir, make_model=make_model, device=device)
        return cls(arts)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return predict_scores(self.artifacts, df)

    def rank(self, df: pd.DataFrame, temperature: Optional[float] = None, by: Sequence[str] = ("year", "round"), include_probs: bool = True, ascending: bool = False) -> pd.DataFrame:
        return predict_custom(self.artifacts, df, temperature=temperature, by=by, include_probs=include_probs, ascending=ascending)


# =====================================================================================
#                                     Sanity
# =====================================================================================

def schk(scaler: FeatureScaler) -> FeatureScaler:
    """Простая проверка размерностей скейлера."""
    if not isinstance(scaler, FeatureScaler):
        raise TypeError("Expected FeatureScaler")
    if scaler.mean.shape != scaler.std.shape:
        raise ValueError("FeatureScaler: mean/std shape mismatch")
    return scaler


__all__ = [
    "Artifacts",
    "load_artifacts",
    "build_sim_frame",
    "build_matrix",
    "predict_scores",
    "attach_win_probs",
    "make_ranking_df",
    "predict_custom",
    "InferenceRunner",
    "apply_temperature",
]
