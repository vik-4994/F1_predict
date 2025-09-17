from __future__ import annotations
import pandas as pd
import numpy as np

def _q_to_ms(s):
    if pd.isna(s): return np.nan
    try:
        parts = str(s).split(":")
        if len(parts)==2:
            m = int(parts[0]); sec = float(parts[1])
            return int((m*60 + sec) * 1000)
        return int(float(s) * 1000)
    except Exception:
        return np.nan

def make_qualifying_features(qualifying: pd.DataFrame) -> pd.DataFrame:
    q = qualifying.copy()
    for c in ["q1","q2","q3"]:
        if c in q.columns:
            q[c+"_ms"] = q[c].apply(_q_to_ms)
    q["best_q_ms"] = q[[c for c in ["q1_ms","q2_ms","q3_ms"] if c in q.columns]].min(axis=1)
    # Grid proxy (fallback): use qualifying.position if exists
    if "position" in q.columns:
        q["grid"] = pd.to_numeric(q["position"], errors="coerce")
    else:
        q["grid"] = q.groupby("raceId")["best_q_ms"].rank(method="first")
    return q.groupby(["raceId","driverId"], as_index=False).agg(
        grid_q=("grid","min"),
        best_q_ms=("best_q_ms","min"),
        constructorId=("constructorId","first"),
    )
