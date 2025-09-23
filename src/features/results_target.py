# FILE: src/features/results_target.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
from .utils import read_csv_if_exists

__all__ = ["featurize"]

def featurize(ctx: dict) -> pd.DataFrame:
    raw_dir: Path = Path(ctx["raw_dir"])
    year: int = int(ctx["year"])
    rnd: int = int(ctx["round"])

    p = raw_dir / f"results_{year}_{rnd}.csv"
    df = read_csv_if_exists(p)
    if df.empty:
        return pd.DataFrame()

    out = pd.DataFrame()

    # Ключ Driver: сначала Abbreviation, иначе DriverNumber
    if "Abbreviation" in df.columns:
        out["Driver"] = df["Abbreviation"].astype(str)
    elif "Driver" in df.columns:
        out["Driver"] = df["Driver"].astype(str)
    elif "DriverNumber" in df.columns:
        out["Driver"] = df["DriverNumber"].astype(str)
    else:
        return pd.DataFrame()

    # Позиция/очки/грид/статус — если есть
    if "Position" in df.columns:
        out["finish_position"] = pd.to_numeric(df["Position"], errors="coerce")
    for col in ["Status", "Points", "GridPosition", "Time"]:
        if col in df.columns:
            out[col] = df[col]

    return out
