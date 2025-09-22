from __future__ import annotations
import numpy as np, pandas as pd
def top_features_by_residual_variance(columns: list[str], residuals: np.ndarray, k: int=10) -> pd.DataFrame:
    var = residuals.var(axis=0)
    idx = np.argsort(var)[::-1][:k]
    return pd.DataFrame({'feature':[columns[i] for i in idx], 'residual_variance':[float(var[i]) for i in idx]})
