from __future__ import annotations
from typing import Dict, Any
import pandas as pd


def _norm(s: str) -> str:
    return str(s).strip().replace('-', '').replace('_', '').lower()


def build_module_map_from_files(
    sandia_xlsx: str, characterization_xlsx: str
) -> Dict[str, Any]:
    out = {'sandia': {}, 'character': {}}
    # Sandia
    try:
        sheets = pd.read_excel(sandia_xlsx, sheet_name=None, engine='openpyxl')
        sheet = 'Data' if 'Data' in sheets else list(sheets.keys())[0]
        df = sheets[sheet]
        df.columns = [str(c).strip() for c in df.columns]
        name_col = next(
            (
                c
                for c in [
                    'Name',
                    'Model',
                    'Module',
                    'Module Name',
                    'NREL identifier',
                    'Identifier',
                    'ID',
                ]
                if c in df.columns
            ),
            df.columns[0],
        )
        for i, row in df.iterrows():
            raw = str(row[name_col])
            key = _norm(raw)
            rec = {'raw_name': raw, 'sheet': sheet, 'row_index': int(i)}
            for k in [
                'a', 'b', 'A0', 'A1', 'A2', 'A3', 'A4',
                'B0', 'B1', 'B2', 'B3', 'B4', 'B5',
            ]:
                col = next((c for c in df.columns if c.lower() == k.lower()), None)
                if col:
                    rec[k] = row.get(col)
            out['sandia'][key] = rec
    except Exception:
        pass
    # Characterization
    try:
        xls = pd.ExcelFile(characterization_xlsx, engine='openpyxl')
        for sh in xls.sheet_names:
            try:
                cdf = xls.parse(sh)
            except Exception:
                continue
            alpha = beta = gamma = None
            for col in cdf.columns:
                cl = str(col).lower()
                if 'alpha' in cl and alpha is None:
                    s = pd.to_numeric(cdf[col], errors='coerce').dropna()
                    alpha = float(s.iloc[0]) if len(s) > 0 else None
                if 'beta' in cl and beta is None:
                    s = pd.to_numeric(cdf[col], errors='coerce').dropna()
                    beta = float(s.iloc[0]) if len(s) > 0 else None
                if 'gamma' in cl and gamma is None:
                    s = pd.to_numeric(cdf[col], errors='coerce').dropna()
                    gamma = float(s.iloc[0]) if len(s) > 0 else None
            out['character'][_norm(sh)] = {
                'raw_sheet': sh,
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma,
            }
    except Exception:
        pass
    return out


def choose_record(module_name: str, module_map: Dict[str, Any]) -> Dict[str, Any]:
    key = _norm(module_name)
    return {
        'sandia': module_map.get('sandia', {}).get(key),
        'character': module_map.get('character', {}).get(key),
    }
