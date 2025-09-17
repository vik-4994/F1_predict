from __future__ import annotations
import pandas as pd
import numpy as np

def build_target_from_results(results: pd.DataFrame, status: pd.DataFrame, races: pd.DataFrame, qualifying: pd.DataFrame, target_cfg: dict) -> pd.DataFrame:
    # Merge status labels
    res = results.merge(status[["statusId","status"]], on="statusId", how="left")
    # Race timestamp
    races = races.copy()
    races["date"] = pd.to_datetime(races["date"], errors="coerce")
    if "time" in races.columns:
        try:
            races["time"] = pd.to_timedelta(races["time"])
            races["race_ts"] = races["date"] + races["time"].fillna(pd.Timedelta(0))
        except Exception:
            races["race_ts"] = races["date"]
    else:
        races["race_ts"] = races["date"]
    res = res.merge(races[["raceId","year","round","circuitId","race_ts"]], on="raceId", how="left")
    # Exclusions by status
    exclude = set(map(str.upper, target_cfg.get("exclude_status", ["DNS","DNQ"])))
    mask_excl = res["status"].str.upper().isin(exclude)
    res = res.loc[~mask_excl].copy()
    # DSQ handling
    dsq_mode = target_cfg.get("treat_DSQ_as", "exclude")
    is_dsq = res["status"].str.upper().eq("DSQ")
    if dsq_mode == "exclude":
        res = res.loc[~is_dsq].copy()
    elif dsq_mode == "max_pos":
        # push DSQ to the end (max position within race)
        res["_is_dsq"] = is_dsq.astype(int)
    # Target: official ordered position
    res["finish_pos"] = pd.to_numeric(res["positionOrder"], errors="coerce")
    # Grid
    res["grid"] = pd.to_numeric(res.get("grid", np.nan), errors="coerce")
    if res["grid"].isna().any() and qualifying is not None:
        q = qualifying.copy()
        q["grid_q"] = pd.to_numeric(q.get("position", np.nan), errors="coerce")
        res = res.merge(q[["raceId","driverId","grid_q"]], on=["raceId","driverId"], how="left")
        res["grid"] = res["grid"].fillna(res["grid_q"]); res.drop(columns=["grid_q"], inplace=True, errors="ignore")
    # For DSQ max_pos option, move them to tail per race
    if dsq_mode == "max_pos" and "_is_dsq" in res.columns:
        # Re-assign finish_pos ranks within each race to place DSQ last
        def reorder(group):
            # non-dsq keep their positionOrder rank; dsq go to the end preserving relative order
            non_dsq = group[~group["_is_dsq"]].sort_values("finish_pos")
            dsq = group[group["_is_dsq"]].sort_values("finish_pos")
            out = pd.concat([non_dsq, dsq], axis=0)
            out["finish_pos"] = np.arange(1, len(out)+1)
            return out
        res = res.groupby("raceId", group_keys=False).apply(reorder).reset_index(drop=True)
        res.drop(columns=["_is_dsq"], inplace=True)
    # Final columns
    cols = ["raceId","driverId","constructorId","year","round","circuitId","race_ts","grid","finish_pos","status"]
    return res[cols].sort_values(["race_ts","raceId","finish_pos"]).reset_index(drop=True)
