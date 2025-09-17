from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib
import pandas as pd

app = FastAPI()

class PredictRequest(BaseModel):
    run: str = "latest"
    rows: list[dict]

@app.post("/predict")
def predict(req: PredictRequest):
    art = Path("data/artifacts")
    run_dir = art / req.run
    if req.run == "latest":
        run_dir = (art / "latest").resolve()
    bundle = joblib.load(run_dir / "model.pkl")
    X = pd.DataFrame(req.rows)[bundle["feature_cols"]]
    for c, m in bundle.get("impute_medians", {}).items():
        if c in X.columns:
            X[c] = X[c].fillna(m)
    preds = bundle["model"].predict(X)
    order = preds.argsort().tolist()
    return {"preds": preds.tolist(), "order": order}
