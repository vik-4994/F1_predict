                           
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
    load_feature_cols,
    select_feature_cols,
    transform_with_scaler_df,
)
from .models.race_outcome_ranker import make_model as make_race_outcome_ranker
from .outcomes import OUTCOME_LABELS, outcome_priority

                                                                                       
                                            
                                                                                       

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

    Если `make_model` не указан, собираем минимальный MLP по мета.
    """
    adir = Path(artifacts_dir)
    dev = _auto_device(device)

                          
    fcols_path = adir / "feature_cols.txt"
    feature_cols = load_feature_cols(fcols_path)

                    
    scal_path = adir / "scaler.json"
    scaler = FeatureScaler.load(scal_path)

                  
    meta_path = adir / "meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

                   
    mdl: Optional[nn.Module] = None
    pt_path = adir / "ranker.pt"

    if _HAS_TORCH and pt_path.exists():
        if make_model is None:
            in_dim = int(meta.get("in_dim", len(feature_cols)))
            hidden = list(map(int, meta.get("hidden", [256, 128])))
            dropout = float(meta.get("dropout", 0.1))
            num_status_classes = int(meta.get("num_status_classes", 0))
            if num_status_classes > 0 or str(meta.get("target_mode", "")).strip().lower() == "multitask_finish_dnf_dsq":
                make_model = lambda _: make_race_outcome_ranker(  # noqa: E731
                    in_dim=in_dim,
                    hidden=hidden,
                    dropout=dropout,
                    num_status_classes=num_status_classes,
                )
            else:
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
                                             
            mdl = make_model(len(feature_cols))
            mdl.to(dev)
            mdl.load_state_dict(obj, strict=False)
        else:
                                                                 
            mdl = obj
            mdl.to(dev)
        if hasattr(mdl, "eval"):
            mdl.eval()
    else:
        mdl = None                                                                                

    return Artifacts(model=mdl, scaler=schk(scaler), feature_cols=feature_cols, meta=meta, device=dev)


                                                                                       
                                                        
                                                                                       

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:                  
        out = self.net(x)
        return out.squeeze(-1)


def _unpack_model_output(output: Any) -> tuple[Any, Any | None]:
    if isinstance(output, dict):
        rank_scores = output.get("rank_scores")
        status_logits = output.get("status_logits")
        if rank_scores is None:
            raise ValueError("Model output dict must contain 'rank_scores'")
        return rank_scores, status_logits
    if isinstance(output, (list, tuple)):
        if not output:
            raise ValueError("Model returned an empty tuple/list")
        return output[0], (output[1] if len(output) > 1 else None)
    return output, None


                                                                                       
                                                          
                                                                                       

def _make_meta(df: pd.DataFrame) -> Dict[str, Any]:
    """Мини-мета как в featureset.build_matrix, но без лишних вычислений."""
    return {
        "drivers": df.get("Driver").tolist() if "Driver" in df.columns else None,
        "year": int(df["year"].iloc[0]) if "year" in df.columns and len(df) else None,
        "round": int(df["round"].iloc[0]) if "round" in df.columns and len(df) else None,
    }


def build_matrix(df: pd.DataFrame, artifacts: Artifacts, as_array: bool = True):
    """featureset.transform_with_scaler_df с возвратом meta (без двойного построения X)."""
    Xs = transform_with_scaler_df(df, artifacts.feature_cols, artifacts.scaler, as_array=True)
    meta = _make_meta(df)
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


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D array for row-wise softmax")
    shifted = arr - np.max(arr, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    denom = exp_vals.sum(axis=1, keepdims=True)
    denom = np.where(denom > 0, denom, 1.0)
    return (exp_vals / denom).astype(np.float32)


def _group_keys(df: pd.DataFrame, by: Sequence[str] | None) -> list[np.ndarray]:
    if by is None:
        return [df.index.to_numpy(dtype=int)]
    cols = [col for col in list(by) if col in df.columns]
    if not cols:
        return [df.index.to_numpy(dtype=int)]
    return [np.asarray(idx, dtype=int) for _, idx in df.groupby(cols, sort=False).indices.items()]


def _groupwise_rank_component(scores: np.ndarray, df: pd.DataFrame, by: Sequence[str] | None) -> np.ndarray:
    out = np.zeros(len(scores), dtype=np.float32)
    for idx in _group_keys(df, by):
        vals = scores[idx].astype(np.float32, copy=False)
        if vals.size <= 1:
            out[idx] = 0.0
            continue
        mu = float(vals.mean())
        sigma = float(vals.std())
        if not np.isfinite(sigma) or sigma < 1e-6:
            out[idx] = vals - mu
        else:
            out[idx] = np.clip((vals - mu) / sigma, -3.0, 3.0)
    return out


def _predict_model_outputs(
    artifacts: Artifacts,
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    fcols = list(feature_cols or artifacts.feature_cols)
    Xs = transform_with_scaler_df(df, fcols, artifacts.scaler, as_array=True)

    if not (_HAS_TORCH and artifacts.model is not None):
        return np.zeros(Xs.shape[0], dtype=np.float32), None

    x = torch.from_numpy(Xs).to(artifacts.device)
    with torch.no_grad():
        output = artifacts.model(x)
    rank_scores, status_logits = _unpack_model_output(output)
    rank_np = rank_scores.detach().float().cpu().numpy().reshape(-1)
    status_np = None
    if status_logits is not None:
        status_np = status_logits.detach().float().cpu().numpy()
    return rank_np, status_np


def predict_scores(
    artifacts: Artifacts,
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Возвращает сырые `scores` формы [N]. Если модели нет — нули.
    Использует порядок колонок из artifacts.feature_cols по умолчанию.
    """
    rank_scores, _ = _predict_model_outputs(artifacts, df, feature_cols=feature_cols)
    return rank_scores


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
        chunks: List[pd.DataFrame] = []
        for _, idx in df.groupby(list(by), sort=False).indices.items():
            chunks.append(_rank_one(df.loc[idx]))
        out = pd.concat(chunks, axis=0, ignore_index=False) if chunks else df.iloc[0:0].copy()

                                              
    lead: List[str] = []
    for c in [*(order_cols or []), *key_cols]:
        if c in out.columns and c not in lead:
            lead.append(c)
    trail = [c for c in out.columns if c not in lead and c not in {rank_col}]
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
    features_df = features_df.reset_index(drop=True).copy()
    temp = float(temperature if temperature is not None else artifacts.meta.get("temperature", 1.0))
    rank_scores, status_logits = _predict_model_outputs(artifacts, features_df)

    if status_logits is None:
        df_sc = attach_win_probs(features_df, rank_scores, by=by, temperature=temp) if include_probs else (
            features_df.assign(score=rank_scores)
        )
        rank_df = make_ranking_df(df_sc, by=by, ascending=ascending)
        return rank_df

    df_sc = features_df.copy()
    labels = [str(label) for label in artifacts.meta.get("outcome_labels", list(OUTCOME_LABELS))]
    if len(labels) != int(status_logits.shape[1]):
        labels = list(OUTCOME_LABELS[: status_logits.shape[1]])
        if len(labels) < int(status_logits.shape[1]):
            labels.extend(f"class_{idx}" for idx in range(len(labels), int(status_logits.shape[1])))
    probs = _softmax_rows(status_logits)
    for idx, label in enumerate(labels):
        df_sc[f"p_{label}"] = probs[:, idx]
    pred_idx = probs.argmax(axis=1)
    df_sc["predicted_outcome"] = [labels[int(idx)] for idx in pred_idx]
    df_sc["score_rank"] = rank_scores.astype(np.float32)
    df_sc["score_status"] = probs @ outcome_priority(labels).to_numpy(dtype=np.float32)
    df_sc["score"] = (
        10.0 * df_sc["score_status"].to_numpy(dtype=np.float32)
        + _groupwise_rank_component(rank_scores, features_df, by)
    ).astype(np.float32)

    if include_probs:
        p_finish = df_sc.get("p_finish")
        finish_prob = p_finish.to_numpy(dtype=np.float32) if p_finish is not None else np.ones(len(df_sc), dtype=np.float32)
        win_logits = rank_scores + np.log(np.clip(finish_prob, 1e-6, 1.0))
        pwin_df = attach_win_probs(features_df.copy(), win_logits, by=by, temperature=temp)
        df_sc["p_win"] = pwin_df["p_win"].to_numpy(dtype=np.float32)

    rank_df = make_ranking_df(df_sc, by=by, ascending=ascending)
    return rank_df


                                                                                       
                                           
                                                                                       

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
    
                                   
                                 
                                   
    @property
    def feature_columns(self) -> List[str]:
        """Совместимость со старыми скриптами: alias на artifacts.feature_cols."""
        return list(self.artifacts.feature_cols)

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Совместимость со старыми скриптами: alias на predict_scores(...)."""
        return predict_scores(self.artifacts, df)


                                                                                       
                                            
                                                                                       

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
