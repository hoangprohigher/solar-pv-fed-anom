from __future__ import annotations
import pandas as pd


def rename_columns(df: pd.DataFrame, columns_map: dict) -> pd.DataFrame:
    rev = {v: k for k, v in columns_map.items() if v in df.columns}
    return df.rename(columns=rev)


def require_columns(df: pd.DataFrame, cols: list[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
