from __future__ import annotations
import pandas as pd
import numpy as np

def driver_constructor_rollups(target_df: pd.DataFrame, qual_feats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = target_df.copy()

    q_min = qual_feats[["raceId", "driverId", "constructorId"]].rename(
        columns={"constructorId": "constructorId_q"}
    )
    df = df.merge(q_min, on=["raceId", "driverId"], how="left")

    if "constructorId" in df.columns:
        df["constructorId"] = df["constructorId"].fillna(df["constructorId_q"])
    else:
        df["constructorId"] = df["constructorId_q"]

    if "constructorId_q" in df.columns:
        df.drop(columns=["constructorId_q"], inplace=True)

    df = df.sort_values("race_ts")

    def drv_roll(g):
        g = g.sort_values("race_ts").copy()
        n = np.arange(len(g))
        g["drv_prev_starts"] = n
        wins_cum = (g["finish_pos"].shift().eq(1)).fillna(False).cumsum()
        podiums_cum = (g["finish_pos"].shift().le(3)).fillna(False).cumsum()
        g["drv_prev_win_rate"] = wins_cum / np.where(n == 0, np.nan, n)
        g["drv_prev_podium_rate"] = podiums_cum / np.where(n == 0, np.nan, n)
        g["drv_prev_avg_finish"] = g["finish_pos"].shift().expanding().mean()
        g["drv_prev_last5_avg"] = g["finish_pos"].shift().rolling(5, min_periods=1).mean()
        return g

    def con_roll(g):
        g = g.sort_values("race_ts").copy()
        n = np.arange(len(g))
        g["con_prev_starts"] = n
        wins_cum = (g["finish_pos"].shift().eq(1)).fillna(False).cumsum()
        podiums_cum = (g["finish_pos"].shift().le(3)).fillna(False).cumsum()
        g["con_prev_win_rate"] = wins_cum / np.where(n == 0, np.nan, n)
        g["con_prev_podium_rate"] = podiums_cum / np.where(n == 0, np.nan, n)
        g["con_prev_avg_finish"] = g["finish_pos"].shift().expanding().mean()
        return g

    drv = df.groupby("driverId", group_keys=False).apply(drv_roll).reset_index(drop=True)

    con = (
        df.dropna(subset=["constructorId"])
          .groupby("constructorId", group_keys=False)
          .apply(con_roll)
          .reset_index(drop=True)
    )

    con = con[["raceId", "driverId", "con_prev_starts", "con_prev_win_rate", "con_prev_avg_finish", "con_prev_podium_rate"]]
    drv = drv[["raceId", "driverId", "drv_prev_starts", "drv_prev_win_rate", "drv_prev_avg_finish", "drv_prev_last5_avg", "drv_prev_podium_rate"]]
    return drv, con
