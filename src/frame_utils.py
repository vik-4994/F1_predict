from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


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


def coalesce_series(left: pd.Series, right: pd.Series, name: str) -> pd.Series:
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
            base = coalesce_series(base, out.iloc[:, idx], name=name)
        merged_cols.append(base)
        merged_names.append(name)

    sanitized = pd.concat(merged_cols, axis=1)
    sanitized.columns = merged_names
    return sanitized

def filter_feature_cols(
    feature_cols: Sequence[str],
    *,
    drop_prefixes: Iterable[str] = (),
    drop_contains: Iterable[str] = (),
    drop_exact: Iterable[str] = (),
    keep_prefixes: Iterable[str] = (),
) -> Tuple[List[str], List[str]]:
    drop_prefixes = [str(x).strip() for x in drop_prefixes if str(x).strip()]
    drop_contains = [str(x).strip() for x in drop_contains if str(x).strip()]
    drop_exact = {str(x).strip() for x in drop_exact if str(x).strip()}
    keep_prefixes = [str(x).strip() for x in keep_prefixes if str(x).strip()]

    kept: List[str] = []
    dropped: List[str] = []

    for col in feature_cols:
        should_keep = True
        if keep_prefixes:
            should_keep = any(col.startswith(prefix) for prefix in keep_prefixes)
        if should_keep and any(col.startswith(prefix) for prefix in drop_prefixes):
            should_keep = False
        if should_keep and any(token in col for token in drop_contains):
            should_keep = False
        if should_keep and col in drop_exact:
            should_keep = False

        if should_keep:
            kept.append(col)
        else:
            dropped.append(col)

    return kept, dropped


__all__ = ["coalesce_series", "sanitize_frame_columns", "filter_feature_cols"]
