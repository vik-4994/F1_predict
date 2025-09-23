# src/training/__init__.py
# -*- coding: utf-8 -*-
"""
Единая точка входа пакета `src.training`.

Реэкспорт только стабильного публичного API, без висячих имён.
Совместимо с переписанными `featureset.py` и `inference.py`.
"""
from __future__ import annotations

# -----------------------------------------------------------------------------
# config
# -----------------------------------------------------------------------------
from .config import (
    TrainConfig,
    build_argparser,
    from_args,
)

# -----------------------------------------------------------------------------
# featureset: отбор фич, построение матрицы, скейлинг, сериализация
# -----------------------------------------------------------------------------
from .featureset import (
    select_feature_cols,
    build_matrix as build_matrix,  # базовое: сырые признаки -> X, meta
    FeatureScaler,
    fit_scaler_on_df,
    transform_with_scaler_df,
    # back-compat алиасы
    fit_scaler,
    transform_with_scaler,
    get_feature_groups,
    # persist
    save_feature_cols,
    load_feature_cols,
    # типы
    FeatureMatrix,
    FeatureSpec,
)

# -----------------------------------------------------------------------------
# data_io / dataset
# -----------------------------------------------------------------------------
from .data_io import (
    load_all,
    build_train_table,
    time_split,
    races_list,
    group_by_race,
)

from .dataset import (
    RaceListDataset,
    collate_races,
)

# -----------------------------------------------------------------------------
# training engine / losses / metrics
# -----------------------------------------------------------------------------
from .engine import (
    train_one_epoch,
    evaluate,
    save_checkpoint,
    load_checkpoint,
    format_metrics,
)

from .losses import (
    PlackettLuceLoss,
    plackett_luce_nll,
    plackett_luce_topk_nll,
    listmle_nll,
)

from .metrics import (
    metrics_for_race,
    aggregate_metrics,
    order_to_ranks,
    scores_to_order,
    spearmanr_from_ranks,
    mae_positions,
    top1_accuracy,
    ndcg_at_k,
)

# -----------------------------------------------------------------------------
# inference
# -----------------------------------------------------------------------------
from .inference import (
    Artifacts,
    load_artifacts,
    build_sim_frame,
    predict_scores,
    attach_win_probs,
    make_ranking_df,
    predict_custom,
    InferenceRunner,
    apply_temperature,
)

# Внимание: в inference есть своя build_matrix (возвращает уже отскейленные фичи).
# Чтобы не перекрыть базовую featureset.build_matrix, экспортируем под иным именем.
from .inference import build_matrix as build_matrix_scaled  # noqa: E402

# -----------------------------------------------------------------------------
# utils
# -----------------------------------------------------------------------------
from .utils import (
    log,
    Timer,
    set_seed,
    get_device,
    make_worker_init_fn,
    count_parameters,
    save_json,
    load_json,
    pushdir,
)

__all__ = [
    # config
    "TrainConfig", "build_argparser", "from_args",
    # featureset
    "select_feature_cols", "build_matrix", "FeatureScaler", "fit_scaler_on_df",
    "transform_with_scaler_df", "fit_scaler", "transform_with_scaler",
    "get_feature_groups", "save_feature_cols", "load_feature_cols",
    "save_scaler_json", "load_scaler_json", "FeatureMatrix", "FeatureSpec",
    # data_io / dataset
    "load_all", "build_train_table", "time_split", "races_list", "group_by_race",
    "RaceListDataset", "collate_races",
    # training engine / losses / metrics
    "train_one_epoch", "evaluate", "save_checkpoint", "load_checkpoint", "format_metrics",
    "PlackettLuceLoss", "plackett_luce_nll", "plackett_luce_topk_nll", "listmle_nll",
    "metrics_for_race", "aggregate_metrics", "order_to_ranks", "scores_to_order",
    "spearmanr_from_ranks", "mae_positions", "top1_accuracy", "ndcg_at_k",
    # inference
    "Artifacts", "load_artifacts", "build_sim_frame", "predict_scores",
    "attach_win_probs", "make_ranking_df", "predict_custom", "InferenceRunner",
    "apply_temperature", "build_matrix_scaled",
    # utils
    "log", "Timer", "set_seed", "get_device", "make_worker_init_fn",
    "count_parameters", "save_json", "load_json", "pushdir",
]

__version__ = "0.2.0"
