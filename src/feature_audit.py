from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


KEY_COLS = ("Driver", "year", "round", "raceId")
GROUP_RULES = (
    ("track_is_", "track_onehot"),
    ("track_same_", "track_same"),
    ("track_", "track_profile"),
    ("weather_", "weather"),
    ("chaos_pre_", "chaos"),
    ("prac_pre_", "practice"),
    ("prac_cmp_pre_", "practice"),
    ("wknd_pre_", "weekend_delta"),
    ("sprint_pre_", "sprint"),
    ("hist_pre_", "history"),
    ("tele_pre_", "telemetry"),
    ("tele_eff_pre_", "telemetry"),
    ("quali_pre_", "quali"),
    ("qexec_pre_", "quali"),
    ("expected_stop_", "strategy"),
    ("first_stint_", "strategy"),
    ("undercut_", "strategy"),
    ("overcut_", "strategy"),
    ("double_stack_", "strategy"),
    ("pitcrew_", "pit_ops"),
    ("slowstop_", "pit_ops"),
    ("compound_mix_", "tyre"),
    ("tyre_", "tyre"),
    ("expected_deg_", "tyre"),
    ("driver_team_pre_", "driver_team"),
    ("driver_trackc_pre_", "driver_track_cluster"),
    ("reliab_", "reliability"),
    ("lap1_", "traffic"),
    ("traffic_", "traffic"),
    ("net_pass_", "traffic"),
)


def feature_group_name(col: str) -> str:
    for prefix, group in GROUP_RULES:
        if col.startswith(prefix):
            return group
    if col in KEY_COLS:
        return "key"
    if col in ("driver_trend", "team_dev_trend", "stability_delta_vs_tm"):
        return "development"
    return "other"


def _effective_feature_cols(df: pd.DataFrame, key_cols: Iterable[str]) -> list[str]:
    keys = {str(c) for c in key_cols}
    return [c for c in df.columns if c not in keys]


def column_health_report(
    df: pd.DataFrame,
    *,
    race_keys: Sequence[str] = ("year", "round"),
    key_cols: Iterable[str] = KEY_COLS,
) -> pd.DataFrame:
    cols = _effective_feature_cols(df, key_cols)
    has_race_keys = all(k in df.columns for k in race_keys)
    grouped = df.groupby(list(race_keys), dropna=False) if has_race_keys else None

    rows = []
    for col in cols:
        series = df[col]
        non_na = int(series.notna().sum())
        row = {
            "feature": col,
            "group": feature_group_name(col),
            "dtype": str(series.dtype),
            "rows": int(len(series)),
            "non_na_rows": non_na,
            "missing_rate": float(series.isna().mean()),
            "all_nan": bool(non_na == 0),
            "unique_non_na": int(series.dropna().nunique()),
        }
        if grouped is not None:
            nunique = grouped[col].nunique(dropna=False)
            any_non_na = grouped[col].apply(lambda x: bool(x.notna().any()))
            row["race_constant_share"] = float((nunique <= 1).mean())
            row["race_non_na_share"] = float(any_non_na.mean())
        else:
            row["race_constant_share"] = np.nan
            row["race_non_na_share"] = np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["all_nan", "missing_rate", "race_constant_share", "feature"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def group_health_report(column_report: pd.DataFrame) -> pd.DataFrame:
    if column_report.empty:
        return pd.DataFrame()

    out = (
        column_report.groupby("group", dropna=False)
        .agg(
            cols=("feature", "size"),
            all_nan_cols=("all_nan", "sum"),
            non_empty_cols=("all_nan", lambda s: int((~s.astype(bool)).sum())),
            mean_missing_rate=("missing_rate", "mean"),
            mean_race_constant_share=("race_constant_share", "mean"),
            mean_race_non_na_share=("race_non_na_share", "mean"),
        )
        .reset_index()
    )
    out["dead_group"] = out["non_empty_cols"] == 0
    return out.sort_values(
        ["dead_group", "mean_missing_rate", "mean_race_constant_share", "group"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


__all__ = ["KEY_COLS", "feature_group_name", "column_health_report", "group_health_report"]
