#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .practice_longrun_pre import featurize as feat_practice
from .quali_execution_pre import featurize as feat_qexec
from .weekend_helpers import current_roster
from .weekend_team_delta_pre import featurize as feat_wknd

__all__ = ["featurize"]


def _rank_pct(series: pd.Series, *, ascending: bool) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype=float)
    vals = pd.to_numeric(series, errors="coerce")
    mask = vals.notna()
    n = int(mask.sum())
    if n == 0:
        return out
    ranks = vals[mask].rank(method="average", ascending=ascending)
    if n == 1:
        out.loc[mask] = 1.0
    else:
        out.loc[mask] = 1.0 - (ranks - 1.0) / float(n - 1)
    return out


def featurize(ctx: Dict) -> pd.DataFrame:
    raw_dir = Path(ctx.get("raw_dir", "data/raw_csv"))
    year = int(ctx["year"])
    rnd = int(ctx["round"])

    drivers = current_roster(raw_dir, year, rnd, ctx.get("drivers"))
    if not drivers:
        return pd.DataFrame()

    base = pd.DataFrame({"Driver": drivers})
    qexec = feat_qexec(ctx)
    practice = feat_practice(ctx)
    wknd = feat_wknd(ctx)

    out = base.copy()
    for df in (qexec, practice, wknd):
        if df is not None and not df.empty:
            cols = [c for c in df.columns if c == "Driver" or c.startswith(("qexec_pre_", "prac_pre_", "wknd_pre_"))]
            out = out.merge(df[cols], on="Driver", how="left")

    def _col(name: str) -> pd.Series:
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce")
        return pd.Series(np.nan, index=out.index, dtype=float)

    if "qexec_pre_best_lap_s" in out.columns:
        best = _col("qexec_pre_best_lap_s")
        out["field_pre_q_gap_to_best_s"] = best - best.min() if best.notna().any() else np.nan
        med = float(best.median()) if best.notna().any() else np.nan
        out["field_pre_q_gap_to_median_s"] = best - med if np.isfinite(med) else np.nan
        out["field_pre_q_rank_pct"] = _rank_pct(best, ascending=True)
    else:
        out["field_pre_q_gap_to_best_s"] = np.nan
        out["field_pre_q_gap_to_median_s"] = np.nan
        out["field_pre_q_rank_pct"] = np.nan

    out["field_pre_prac_short_rank_pct"] = _rank_pct(_col("prac_pre_shortrun_best3_s"), ascending=True)
    out["field_pre_prac_longrun_rank_pct"] = _rank_pct(_col("prac_pre_longrun_pace_s"), ascending=True)
    out["field_pre_tm_delta_rank_pct"] = _rank_pct(_col("wknd_pre_q_tm_delta_s"), ascending=True)

    pct_cols: List[str] = [
        "field_pre_q_rank_pct",
        "field_pre_prac_short_rank_pct",
        "field_pre_prac_longrun_rank_pct",
        "field_pre_tm_delta_rank_pct",
    ]
    pct_frame = out[pct_cols].apply(pd.to_numeric, errors="coerce")
    out["field_pre_available_scores_n"] = pct_frame.notna().sum(axis=1).astype(float)
    out["field_pre_combo_rank_pct"] = pct_frame.mean(axis=1, skipna=True)

    keep = [
        "Driver",
        "field_pre_q_gap_to_best_s",
        "field_pre_q_gap_to_median_s",
        "field_pre_q_rank_pct",
        "field_pre_prac_short_rank_pct",
        "field_pre_prac_longrun_rank_pct",
        "field_pre_tm_delta_rank_pct",
        "field_pre_available_scores_n",
        "field_pre_combo_rank_pct",
    ]
    for col in keep:
        if col not in out.columns:
            out[col] = np.nan
    return out[keep]
