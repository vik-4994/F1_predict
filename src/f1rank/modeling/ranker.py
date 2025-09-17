from __future__ import annotations
import numpy as np
from typing import Tuple, Optional

# LightGBM if available; fallback to sklearn regressor
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

from sklearn.ensemble import GradientBoostingRegressor

class RankerWrapper:
    def __init__(self, params: dict, objective: str = "lambdarank"):
        self.params = params or {}
        self.objective = objective
        self.model = None
        self.fallback = False

    def fit(self, X, y, groups: Optional[np.ndarray] = None):
        if HAS_LGB:
            self.model = lgb.LGBMRanker(objective=self.objective, **self.params)
            self.model.fit(X, y, group=groups)
        else:
            # Fallback: regress finish position
            self.model = GradientBoostingRegressor(random_state=self.params.get("random_state", 42))
            self.model.fit(X, y)
            self.fallback = True
        return self

    def predict(self, X):
        return self.model.predict(X)

def group_by_race(df, group_col="raceId"):
    # returns group sizes for LightGBM group parameter
    return df.groupby(group_col).size().to_numpy()
