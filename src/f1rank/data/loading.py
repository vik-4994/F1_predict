from __future__ import annotations
from pathlib import Path
import pandas as pd

def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def load_raw(paths) -> dict[str, pd.DataFrame]:
    p = paths.raw
    files = [
        "circuits.csv", "constructors.csv", "constructor_results.csv", "constructor_standings.csv",
        "drivers.csv", "driver_standings.csv", "lap_times.csv", "pit_stops.csv", "qualifying.csv",
        "races.csv", "results.csv", "seasons.csv", "sprint_results.csv", "status.csv"
    ]
    dfs = {}
    for f in files:
        fp = p / f
        if fp.exists():
            dfs[f[:-4]] = read_csv(fp)
        else:
            dfs[f[:-4]] = None
    return dfs
