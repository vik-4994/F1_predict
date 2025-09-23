# src/training/featureset.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import json
import numpy as np
import pandas as pd

# =============================================================
# Public types (kept for backwards-compat with __init__.py)
# =============================================================

@dataclass(frozen=True)
class FeatureSpec:
    """Stub to keep old imports stable. Not used in this pipeline."""
    name: str
    fn: Optional[object] = None
    required: bool = False
    note: str = ""

# Convenience alias used around the codebase
FeatureMatrix = Tuple[np.ndarray, Dict[str, object]]

__all__ = [
    # core API used by dataset/engine/inference
    "select_feature_cols",
    "build_matrix",
    "FeatureScaler",
    "fit_scaler_on_df",
    "transform_with_scaler_df",
    # back-compat aliases
    "fit_scaler",
    "transform_with_scaler",
    # persistence helpers
    "save_feature_cols",
    "load_feature_cols",
    # optional helpers
    "get_feature_groups",
    # types
    "FeatureMatrix",
    "FeatureSpec",
]

# =============================================================
# Feature selection
# =============================================================

_KEY = ["Driver", "year", "round"]

# Columns we must never feed to the ranker (targets/metadata/post-factum)
_BLACKLIST = {
    # keys / identifiers
    "year", "round",
    # explicit targets / leaks
    "finish_position", "finish_pos", "finish_order", "position", "place",
    "finish_pos_eff",  # computed effective target — must be excluded
    # post-factum / outcome-like fields
    "Points", "GridPosition", "Status", "Time",
}

# extra prefixes we proactively treat as numeric even if dtype=object
# based on pre-race feature modules
_OBJECT_NUMERIC_PREFIXES: Tuple[str, ...] = (
    # driver/team priors
    "driver_team_pre_",
    # history form
    "hist_pre_",
    # telemetry priors
    "tele_pre_",
    # quali priors
    "quali_pre_",
    # traffic & overtakes / lap1
    "traffic_", "lap1_", "net_pass_",
    # track one-hot (sometimes saved as "0"/"1" strings)
    "track_is_",
)


def _coerce_object_featurelike(df: pd.DataFrame,
                               prefixes: Sequence[str] = _OBJECT_NUMERIC_PREFIXES,
                               min_numeric_share: float = 0.6) -> pd.DataFrame:
    """Make a lightweight copy where object columns that look like
    numeric feature columns (by prefix) are converted to numeric.

    If at least `min_numeric_share` of values parse as numbers, keep parsed values.
    Otherwise leave column untouched.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    obj_cols = out.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        if not any(c.startswith(p) for p in prefixes):
            continue
        s = pd.to_numeric(out[c].replace({"True": 1, "False": 0, True: 1, False: 0}), errors="coerce")
        share = float(s.notna().mean()) if len(s) else 0.0
        if share >= min_numeric_share:
            out[c] = s
    return out


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    # include numpy numbers, pandas extension ints/floats, and booleans
    num = df.select_dtypes(include=[np.number, "number", "integer", "floating", "bool", "boolean"]).columns.tolist()
    # keep order, drop dups just in case
    seen: set = set()
    out: List[str] = []
    for c in num:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def select_feature_cols(
    df: pd.DataFrame,
    *,
    drop_blacklist: bool = True,
    drop_all_nan: bool = True,
    coerce_object_featurelike: bool = True,
) -> List[str]:
    """Return ordered list of feature columns from a pre-built table (all_features).

    Policy kept intentionally permissive to avoid shrinking the feature space:
      - optionally coerce object columns with known feature prefixes → numeric
      - take all numeric/boolean columns
      - drop known keys/targets if `drop_blacklist`
      - optionally drop columns that are entirely NaN
    """
    if df is None or df.empty:
        return []

    work = _coerce_object_featurelike(df) if coerce_object_featurelike else df

    cols = _numeric_columns(work)

    if drop_blacklist:
        cols = [c for c in cols if c not in _BLACKLIST]

    if drop_all_nan:
        cols = [c for c in cols if not work[c].isna().all()]

    return cols

# =============================================================
# Matrix building
# =============================================================

def _ensure_columns(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in feature_cols:
        if c not in out.columns:
            out[c] = np.nan
    return out


def build_matrix(df: pd.DataFrame, feature_cols: Sequence[str]) -> FeatureMatrix:
    """Take a per-race slice of all_features and build an [N,F] matrix (unscaled).

    Returns (X, meta) where meta contains Driver list and optional (year, round).
    """
    if df is None or df.empty:
        return np.zeros((0, len(feature_cols)), dtype=np.float32), {}

    # normalize keys and coerce featurelike objects
    out = _ensure_columns(df, feature_cols)
    if "Driver" in out.columns:
        out["Driver"] = out["Driver"].astype(str)

    out = _coerce_object_featurelike(out)

    # Booleans → 0/1, everything → float32
    for c in feature_cols:
        if c in out.columns and out[c].dtype == "boolean":
            out[c] = out[c].astype("float32")

    X = out.loc[:, list(feature_cols)].to_numpy(dtype=np.float32, copy=False)

    meta: Dict[str, object] = {
        "drivers": out.get("Driver").tolist() if "Driver" in out.columns else None,
        "year": int(out["year"].iloc[0]) if "year" in out.columns and len(out) else None,
        "round": int(out["round"].iloc[0]) if "round" in out.columns and len(out) else None,
    }
    return X, meta

# =============================================================
# Scaler (fit on train → used in inference)
# =============================================================

@dataclass
class FeatureScaler:
    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-6

    # ---------- persistence ----------
    def save(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        obj = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "eps": float(self.eps),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f)

    @staticmethod
    def load(path: Path | str) -> "FeatureScaler":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        mean = np.asarray(obj["mean"], dtype=np.float32)
        std = np.asarray(obj["std"], dtype=np.float32)
        eps = float(obj.get("eps", 1e-6))
        return FeatureScaler(mean=mean, std=std, eps=eps)

    # ---------- transform ----------
    def transform(self, X: np.ndarray) -> np.ndarray:
        m = self.mean.astype(np.float32)
        s = self.std.astype(np.float32)
        s = np.where(s < self.eps, 1.0, s)  # guard against zero-variance
        return (X - m) / s


def fit_scaler_on_df(df: pd.DataFrame, feature_cols: Sequence[str]) -> FeatureScaler:
    """
    Fit the standardization stats on a training dataframe.
    Robustly handles NaNs/Infs and near-zero variance.
    """
    d = _ensure_columns(df, feature_cols)
    d = _coerce_object_featurelike(d)
    X = d.loc[:, list(feature_cols)].to_numpy(dtype=np.float32, copy=False)
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    # replace non-finite stats with safe constants
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where(np.isfinite(std), std, 1.0).astype(np.float32)
    # guard against tiny variance
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return FeatureScaler(mean=mean, std=std, eps=1e-6)


def transform_with_scaler_df(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    scaler: FeatureScaler,
    *,
    as_array: bool = True,
) -> np.ndarray | pd.DataFrame:
    d = _ensure_columns(df, feature_cols)
    d = _coerce_object_featurelike(d)
    X = d.loc[:, list(feature_cols)].to_numpy(dtype=np.float32, copy=False)
    # impute by mean before scaling
    imputed = np.where(np.isnan(X), scaler.mean.astype(np.float32), X)
    Xs = scaler.transform(imputed)
    if as_array:
        return Xs
    return pd.DataFrame(Xs, columns=list(feature_cols), index=d.index)

# Back-compat short aliases used in some scripts
fit_scaler = fit_scaler_on_df
transform_with_scaler = transform_with_scaler_df

# =============================================================
# Persistence helpers for feature columns
# =============================================================

def save_feature_cols(path: Path | str, feature_cols: Sequence[str]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in feature_cols:
            f.write(str(c) + "\n")


def load_feature_cols(path: Path | str) -> List[str]:
    """Load feature column list from a text file (one per line)."""
    with open(path, "r", encoding="utf-8") as f:
        # rstrip("") is invalid; use rstrip() to drop trailing newline/space
        cols = [line.rstrip() for line in f]
    # drop empties, preserve order
    return [c for c in cols if c]

# =============================================================
# Optional: simple grouping (handy for debugging/EDA)
# =============================================================

def get_feature_groups(cols: Iterable[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    def put(g: str, c: str):
        groups.setdefault(g, []).append(c)

    for c in cols:
        if c.startswith("track_is_"):
            put("track_onehot", c)
        elif c.startswith("track_same_"):
            put("track_same", c)
        elif c.startswith("weather_"):
            put("weather", c)
        elif c.startswith("strategy_") or c.startswith("expected_stop_"):
            put("strategy", c)
        elif c.startswith("tyre_") or c.startswith("compound_") or c.startswith("deg_"):
            put("tyre", c)
        elif c.startswith("pit") or "slowstop" in c:
            put("pit_ops", c)
        elif c.startswith("reliab_"):
            put("reliability", c)
        elif c.startswith("hist_"):
            put("history", c)
        elif c.startswith("tele_"):
            put("telemetry_quali", c)
        elif c.startswith("quali_"):
            put("qualifying", c)
        elif c.startswith("driver_team_pre_"):
            put("driver_team", c)
        elif c.startswith("lap1_") or c.startswith("traffic_") or c.startswith("net_pass_"):
            put("traffic_overtake", c)
        elif c.startswith("driver_") or c.startswith("team_"):
            put("driver_team_misc", c)
        else:
            put("misc", c)
    return groups
