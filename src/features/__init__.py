"""
Feature modules registry (PRE-RACE ONLY).

Каждый модуль экспортирует функцию:
    featurize(ctx: dict) -> pandas.DataFrame
где ctx = {'raw_dir': Path|str, 'year': int, 'round': int, 'verbose'?: bool}
и возвращает DF с ключом 'Driver' (одна строка на пилота).

В реестр включены только pre-race фичи (без лейкеджа) и
сборщик featurize_pre(...), который их мёрджит. Таргеты держим отдельно.
"""
from __future__ import annotations
import pandas as pd

# -------- PRE-RACE FEATURE MODULES (no leakage) --------
from .track_onehot import featurize as track_onehot
from .weather_basic import featurize as weather_basic
from .history_form import featurize as history_form
from .telemetry_history_pre import featurize as telemetry_history_pre
from .quali_priors_pre import featurize as quali_priors_pre
from .strategy_priors_pre import featurize as strategy_priors_pre
from .tyre_priors_pre import featurize as tyre_priors_pre
from .dev_trend_pre import featurize as dev_trend_pre
from .reliability_risk_pre import featurize as reliability_risk_pre
from .pit_ops_risk_pre import featurize as pit_ops_risk_pre
from .traffic_overtake_pre import featurize as traffic_overtake_pre
from .driver_team_priors_pre import featurize as driver_team_priors_pre
from .pit_ops_pre import featurize as pit_ops_pre

# -------- TARGETS / LABELS (kept separate) --------
from .results_target import featurize as results_target

# Рекомендуемый порядок: контекст трассы → погода → форма/прайоры →
# телеметрия‑история → quali/стратегия/шины/тренды/надёжность/пит‑риски → трафик → прайоры → пит‑операции
FEATURIZERS = [
    ("track_onehot", track_onehot),
    ("weather_basic", weather_basic),
    ("history_form", history_form),
    ("telemetry_history_pre", telemetry_history_pre),
    ("quali_priors_pre", quali_priors_pre),
    ("strategy_priors_pre", strategy_priors_pre),
    ("tyre_priors_pre", tyre_priors_pre),
    ("dev_trend_pre", dev_trend_pre),
    ("reliability_risk_pre", reliability_risk_pre),
    ("pit_ops_risk_pre", pit_ops_risk_pre),
    ("traffic_overtake_pre", traffic_overtake_pre),
    ("driver_team_priors_pre", driver_team_priors_pre),
    ("pit_ops_pre", pit_ops_pre),
]

# Отдельно держим таргеты/лейблы, чтобы собирать их по флагу
TARGETIZERS = [
    ("results_target", results_target),
]


def featurize_pre(ctx: dict, modules: list[str] | None = None, how: str = "outer") -> pd.DataFrame:
    """Запуск только PRE‑race фичей и merge по колонке 'Driver'.

    ctx: {
      'raw_dir': Path|str,
      'year': int,
      'round': int,
      'verbose'?: bool
    }
    modules: необязательный подмножество имён из FEATURIZERS
    how: стратегия merge (по умолчанию 'outer')
    """
    frames: list[pd.DataFrame] = []
    run = [(n, f) for (n, f) in FEATURIZERS if (modules is None or n in modules)]
    for name, fn in run:
        try:
            df = fn(ctx)
            if df is not None and not df.empty:
                frames.append(df)
        except Exception as e:
            if ctx.get("verbose"):
                print(f"[pre] {name} failed: {e}")
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for df in frames[1:]:
        out = out.merge(df, on="Driver", how=how)
    return out

# Для совместимости можно вызывать просто features.featurize(ctx)
featurize = featurize_pre

__all__ = [
    # функции‑экспорты модулей
    "track_onehot",
    "weather_basic",
    "history_form",
    "telemetry_history_pre",
    "quali_priors_pre",
    "strategy_priors_pre",
    "tyre_priors_pre",
    "dev_trend_pre",
    "reliability_risk_pre",
    "pit_ops_risk_pre",
    "traffic_overtake_pre",
    "driver_team_priors_pre",
    "pit_ops_pre",
    # таргеты/лейблы
    "results_target",
    # реестры и сборщики
    "FEATURIZERS",
    "TARGETIZERS",
    "featurize_pre",
    "featurize",
]