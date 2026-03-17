                          
# -*- coding: utf-8 -*-
"""
Единая точка входа пакета `src.training`.

Реэкспорт только стабильного публичного API, без висячих имён.
Совместимо с переписанными `featureset.py` и `inference.py`.
"""
from __future__ import annotations

                                                                               
        
                                                                               
from .config import (
    TrainConfig,
    build_argparser,
    from_args,
)

                                                                               
                                                                   
                                                                               
from .featureset import (
    select_feature_cols,
    build_matrix as build_matrix,                                      
    FeatureScaler,
    fit_scaler_on_df,
    transform_with_scaler_df,
                        
    fit_scaler,
    transform_with_scaler,
    sanitize_frame_columns,
    get_feature_groups,
             
    save_feature_cols,
    load_feature_cols,
          
    FeatureMatrix,
    FeatureSpec,
)

                                                                               
                   
                                                                               
from .data_io import (
    load_all,
    build_train_table,
    time_split,
    race_recency_weights,
    races_list,
    group_by_race,
)

from .dataset import (
    RaceListDataset,
    collate_races,
)

                                                                               
                                    
                                                                               
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
    outcome_metrics,
)

from .outcomes import (
    OUTCOME_LABELS,
    OUTCOME_TO_ID,
    FINISH_ID,
    DNF_ID,
    DSQ_ID,
    normalize_outcome_status,
    outcome_label_series,
    outcome_id_series,
    outcome_priority,
)

from .regulations import (
    RegulationEra,
    REGULATION_ERAS,
    default_era_weights,
    regulation_era_for_year,
    regulation_era_series,
    regulation_era_race_weights,
    combine_race_weight_maps,
    summarize_era_weights,
)

                                                                               
           
                                                                               
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

                                                                                  
                                                                                   
from .inference import build_matrix as build_matrix_scaled  # noqa: E402

                                                                               
       
                                                                               
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
            
    "TrainConfig", "build_argparser", "from_args",
                
    "select_feature_cols", "build_matrix", "FeatureScaler", "fit_scaler_on_df",
    "transform_with_scaler_df", "fit_scaler", "transform_with_scaler",
    "sanitize_frame_columns",
    "get_feature_groups", "save_feature_cols", "load_feature_cols",
    "FeatureMatrix", "FeatureSpec",
                       
    "load_all", "build_train_table", "time_split", "race_recency_weights", "races_list", "group_by_race",
    "RaceListDataset", "collate_races",
                                        
    "train_one_epoch", "evaluate", "save_checkpoint", "load_checkpoint", "format_metrics",
    "PlackettLuceLoss", "plackett_luce_nll", "plackett_luce_topk_nll", "listmle_nll",
    "metrics_for_race", "aggregate_metrics", "order_to_ranks", "scores_to_order",
    "spearmanr_from_ranks", "mae_positions", "top1_accuracy", "ndcg_at_k", "outcome_metrics",
    "OUTCOME_LABELS", "OUTCOME_TO_ID", "FINISH_ID", "DNF_ID", "DSQ_ID",
    "normalize_outcome_status", "outcome_label_series", "outcome_id_series", "outcome_priority",
    "RegulationEra", "REGULATION_ERAS", "default_era_weights", "regulation_era_for_year",
    "regulation_era_series", "regulation_era_race_weights", "combine_race_weight_maps", "summarize_era_weights",
               
    "Artifacts", "load_artifacts", "build_sim_frame", "predict_scores",
    "attach_win_probs", "make_ranking_df", "predict_custom", "InferenceRunner",
    "apply_temperature", "build_matrix_scaled",
           
    "log", "Timer", "set_seed", "get_device", "make_worker_init_fn",
    "count_parameters", "save_json", "load_json", "pushdir",
]

__version__ = "0.2.1"
