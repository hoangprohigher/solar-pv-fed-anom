from __future__ import annotations
import numpy as np, pandas as pd

def apply_basic_qc(df: pd.DataFrame, drop_maint: bool=True, drop_precip_dew: bool=True) -> pd.DataFrame:
    out = df.copy()
    if drop_maint and 'Maintenance' in out.columns:
        out = out[out['Maintenance'] != 1]
    if drop_precip_dew and 'QA_Residual' in out.columns:
        res = out['QA_Residual'].dropna()
        if len(res) > 50:
            m, s = res.mean(), res.std()
            out = out[(out['QA_Residual'] > m-3*s) & (out['QA_Residual'] < m+3*s)]
    return out
