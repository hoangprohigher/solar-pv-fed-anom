from __future__ import annotations
import numpy as np
import pandas as pd


def compute_cell_temp(t_back: pd.Series, poa: pd.Series) -> pd.Series:
    return t_back + 2.0 * (poa / 1000.0)


def time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    dt = pd.Series(index=index, data=index)
    sod = dt.dt.hour * 3600 + dt.dt.minute * 60 + dt.dt.second
    return pd.DataFrame(
        {
            'hour': dt.dt.hour.values,
            'dow': dt.dt.dayofweek.values,
            'sod_sin': np.sin(2 * np.pi * sod / 86400),
            'sod_cos': np.cos(2 * np.pi * sod / 86400),
        },
        index=index,
    )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {'BackTemp', 'POA'}.issubset(out.columns):
        out['Tcell'] = compute_cell_temp(out['BackTemp'], out['POA'])
    for a, b in [('DHI', 'GHI'), ('DNI', 'GHI'), ('POA', 'GHI')]:
        if a in out.columns and b in out.columns:
            out[f'ratio_{a}_{b}'] = np.where(
                out[b] > 1e-3,
                out[a] / out[b],
                np.nan,
            )
    if {'Imp', 'Vmp'}.issubset(out.columns):
        out['Pm_calc'] = out['Imp'] * out['Vmp']
    out = out.join(time_features(out.index))
    out = out.dropna(
        thresh=int(0.8 * len(out.columns))
    )
    return out


def select_training_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    num = df.select_dtypes(include=['number']).copy()
    drop_cols = [
        c for c in ['Pm', 'Pm_calc', 'Imp', 'Vmp', 'Isc', 'Voc'] if c in num.columns
    ]
    X = num.drop(columns=drop_cols, errors='ignore').dropna()
    return df.loc[X.index], X


def enrich_with_module_labels(
    df: pd.DataFrame,
    module_map_path: str | None = None,
) -> pd.DataFrame:
    """Attach per-module labels (Sandia a,b; Character alpha,beta,gamma).

    If file missing or not matched, fill NaN. Values are constant per (Site, Module).
    """
    if module_map_path is None:
        return df
    import json
    import os

    if not os.path.exists(module_map_path):
        return df
    with open(module_map_path, 'r', encoding='utf-8') as f:
        mp = json.load(f)

    def _norm(s: str) -> str:
        return str(s).strip().replace('-', '').replace('_', '').lower()

    rows = []
    if 'Module' not in df.columns or 'Site' not in df.columns:
        return df
    for (site, module), grp in df.groupby(['Site', 'Module']):
        key = _norm(module)
        sandia = mp.get('sandia', {}).get(key, {})
        charac = mp.get('character', {}).get(key, {})
        add = {
            'sandia_a': sandia.get('a'),
            'sandia_b': sandia.get('b'),
            'char_alpha': charac.get('alpha'),
            'char_beta': charac.get('beta'),
            'char_gamma': charac.get('gamma'),
        }
        g2 = grp.assign(**add)
        rows.append(g2)
    if rows:
        return pd.concat(rows).sort_index()
    return df
