from __future__ import annotations
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

def load_artifacts(art_dir: str | Path):
    art_dir = Path(art_dir)
    bundle = joblib.load(art_dir / "model.pkl")
    meta = (art_dir / "schema.json")
    schema = None
    if meta.exists():
        import json
        schema = json.loads(meta.read_text())
    return bundle, schema

def predict_positions(model_bundle, features_df: pd.DataFrame) -> pd.DataFrame:
    model = model_bundle["model"]
    impute = model_bundle.get("impute_medians", {})
    X = features_df.copy()
    for c, m in impute.items():
        if c in X.columns:
            X[c] = X[c].fillna(m)
    preds = model.predict(X)
    return pd.DataFrame({"pred": preds})
