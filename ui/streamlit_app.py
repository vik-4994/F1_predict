import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="F1 Ranker", layout="wide")
st.title("🏁 F1 Ranker (modular project)")

run_dir = Path("data/artifacts/latest").resolve()
if not run_dir.exists():
    st.warning("Нет артефактов. Сначала обучите модель (scripts/train_ranker.py).")
else:
    bundle = joblib.load(run_dir / "model.pkl")
    feat_dir = Path("data/features")
    split = st.selectbox("Split", ["test","valid","train"])
    fp = feat_dir / f"{split}.parquet"
    if not fp.exists():
        st.warning(f"Нет файла {fp}")
    else:
        df = pd.read_parquet(fp)
        race_ids = sorted(df["raceId"].unique().tolist())
        race_id = st.selectbox("raceId", race_ids)
        sub = df[df["raceId"]==race_id].copy()
        X = sub[bundle["feature_cols"]].copy()
        for c, m in bundle.get("impute_medians", {}).items():
            if c in X.columns:
                X[c] = X[c].fillna(m)
        sub["pred"] = bundle["model"].predict(X)
        sub = sub.sort_values("pred")
        st.dataframe(sub[["driverId","constructorId","grid","pred","finish_pos"]])
