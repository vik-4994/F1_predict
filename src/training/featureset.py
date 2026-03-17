                            
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import json
import numpy as np
import pandas as pd

                                                               
                                                           
                                                               

@dataclass(frozen=True)
class FeatureSpec:
    """Stub to keep old imports stable. Not used in this pipeline."""
    name: str
    fn: Optional[object] = None
    required: bool = False
    note: str = ""

                                            
FeatureMatrix = Tuple[np.ndarray, Dict[str, object]]


def _series_values_compatible(left: pd.Series, right: pd.Series) -> bool:
    mask = left.notna() & right.notna()
    if not bool(mask.any()):
        return True

    l = left[mask]
    r = right[mask]

    if pd.api.types.is_numeric_dtype(l) and pd.api.types.is_numeric_dtype(r):
        return bool(
            np.allclose(
                l.to_numpy(dtype=np.float64, copy=False),
                r.to_numpy(dtype=np.float64, copy=False),
                equal_nan=True,
                rtol=1e-6,
                atol=1e-8,
            )
        )

    return bool((l.astype(str) == r.astype(str)).all())


def _coalesce_series(left: pd.Series, right: pd.Series, name: str) -> pd.Series:
    if not _series_values_compatible(left, right):
        raise ValueError(f"Incompatible duplicate column '{name}'")
    out = left.copy()
    fill_mask = out.isna() & right.notna()
    if bool(fill_mask.any()):
        out.loc[fill_mask] = right.loc[fill_mask]
    return out


def sanitize_frame_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return df
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    if not out.columns.has_duplicates:
        return out

    merged_cols: List[pd.Series] = []
    merged_names: List[str] = []
    seen: set[str] = set()

    for name in out.columns:
        if name in seen:
            continue
        seen.add(name)
        idxs = [i for i, col in enumerate(out.columns) if col == name]
        base = out.iloc[:, idxs[0]].copy()
        for idx in idxs[1:]:
            base = _coalesce_series(base, out.iloc[:, idx], name=name)
        merged_cols.append(base)
        merged_names.append(name)

    sanitized = pd.concat(merged_cols, axis=1)
    sanitized.columns = merged_names
    return sanitized

__all__ = [
                                               
    "select_feature_cols",
    "build_matrix",
    "FeatureScaler",
    "fit_scaler_on_df",
    "transform_with_scaler_df",
                         
    "fit_scaler",
    "transform_with_scaler",
    "sanitize_frame_columns",
                         
    "save_feature_cols",
    "load_feature_cols",
                      
    "get_feature_groups",
           
    "FeatureMatrix",
    "FeatureSpec",
]

                                                               
                   
                                                               

_KEY = ["Driver", "year", "round"]

                                                                         
_BLACKLIST = {
                       
    "year", "round",
                              
    "finish_position", "finish_pos", "finish_order", "position", "place",
    "finish_pos_eff", "outcome_id", "result_outcome",
                                       
    "Points", "GridPosition", "Status", "Time",
}

                      
                                                                   
_SAFE_ONEHOT_PREFIXES: Tuple[str, ...] = (
    "track_is_", "team_is_", "driver_is_", "track_cluster_"
)

                                                   
_BLOCK_PATTERNS = (
    r"_last_seen_(year|round)$",                                           
    r"^(quali_pre|driver_team_pre).*_hist_n$",                           
    r"^quali_pre_pos_iqr$",                                                           
    r"^driver_team_pre_driver_finish_iqr$",
)


                                                                     
                                   
_OBJECT_NUMERIC_PREFIXES = (
    "driver_team_pre_", "driver_trackc_pre_",              
    "hist_pre_", "tele_pre_", "quali_pre_",
    "traffic_", "lap1_", "net_pass_",
    "track_is_", "track_",                                                                             
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
                                                                       
    num = df.select_dtypes(include=[np.number, "number", "integer", "floating", "bool", "boolean"]).columns.tolist()
                                        
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

    work = sanitize_frame_columns(df)
    work = _coerce_object_featurelike(work) if coerce_object_featurelike else work

    cols = _numeric_columns(work)

    if drop_blacklist:
        cols = [c for c in cols if c not in _BLACKLIST]

    if drop_all_nan:
        cols = [c for c in cols if not work[c].isna().all()]

                                       
    if cols:
        import re
        keep: List[str] = []
        for c in cols:
            bad = any(re.search(p, c) for p in _BLOCK_PATTERNS)
            if not bad:
                keep.append(c)
        cols = keep

                                                                                                   
                                            

    return cols

                                                               
                 
                                                               

def _ensure_columns(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    out = sanitize_frame_columns(df)
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

                                                   
    out = _ensure_columns(df, feature_cols)
    if "Driver" in out.columns:
        out["Driver"] = out["Driver"].astype(str)

    out = _coerce_object_featurelike(out)

                                          
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

                                                               
                                           
                                                               

@dataclass
class FeatureScaler:
    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-6

                                       
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

                                     
    def transform(self, X: np.ndarray) -> np.ndarray:
        m = self.mean.astype(np.float32)
        s = self.std.astype(np.float32)
        s = np.where(s < self.eps, 1.0, s)                               
        return (X - m) / s


def fit_scaler_on_df(df: pd.DataFrame, feature_cols: Sequence[str]) -> FeatureScaler:
    """
    Fit the standardization stats on a training dataframe.
    Robustly handles NaNs/Infs and near-zero variance.
    """
    d = _ensure_columns(df, feature_cols)
    d = _coerce_object_featurelike(d)
    X = d.loc[:, list(feature_cols)].to_numpy(dtype=np.float32, copy=False)
    finite = np.isfinite(X)
    counts = finite.sum(axis=0).astype(np.float32, copy=False)
    safe_counts = np.where(counts > 0, counts, 1.0).astype(np.float32, copy=False)

    sums = np.where(finite, X, 0.0).sum(axis=0, dtype=np.float64)
    mean = (sums / safe_counts).astype(np.float32, copy=False)

    centered = np.where(finite, X - mean, 0.0)
    var = (centered * centered).sum(axis=0, dtype=np.float64) / safe_counts
    std = np.sqrt(var).astype(np.float32, copy=False)
                                                  
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where(np.isfinite(std), std, 1.0).astype(np.float32)
                                 
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

                                                        
    if feature_cols:
        oh_mask = np.array([any(str(c).startswith(p) for p in _SAFE_ONEHOT_PREFIXES)
                            for c in feature_cols], dtype=bool)
        if oh_mask.any():
            mean = mean.copy()
            std = std.copy()
            mean[oh_mask] = 0.0
            std[oh_mask] = 1.0

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
                                   
    imputed = np.where(np.isnan(X), scaler.mean.astype(np.float32), X)
    Xs = scaler.transform(imputed)
    if as_array:
        return Xs
    return pd.DataFrame(Xs, columns=list(feature_cols), index=d.index)

                                                
fit_scaler = fit_scaler_on_df
transform_with_scaler = transform_with_scaler_df

                                                               
                                         
                                                               

def save_feature_cols(path: Path | str, feature_cols: Sequence[str]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in feature_cols:
            f.write(str(c) + "\n")


def load_feature_cols(path: Path | str) -> List[str]:
    """Load feature column list from a text file (one per line)."""
    with open(path, "r", encoding="utf-8") as f:
                                                                            
        cols = [line.rstrip() for line in f]
                                  
    return [c for c in cols if c]

                                                               
                                                     
                                                               

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
        elif c.startswith("sprint_pre_"):
            put("sprint", c)
        elif c.startswith("field_pre_"):
            put("weekend_field", c)
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
        elif c.startswith("tele_") or c.startswith("tele_eff_"):
            put("telemetry_quali", c)
        elif c.startswith("quali_") or c.startswith("qexec_"):
            put("qualifying", c)
        elif c.startswith("qevo_"):
            put("qualifying", c)
        elif c.startswith("prac_"):
            put("practice", c)
        elif c.startswith("ready_"):
            put("practice", c)
        elif c.startswith("driver_team_pre_"):
            put("driver_team", c)
        elif c.startswith("lap1_") or c.startswith("traffic_") or c.startswith("net_pass_"):
            put("traffic_overtake", c)
        elif c.startswith("driver_") or c.startswith("team_"):
            put("driver_team_misc", c)
        else:
            put("misc", c)
    return groups
