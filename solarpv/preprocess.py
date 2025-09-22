from __future__ import annotations
import os
import re
import pandas as pd
from .utils import load_cfg, list_csvs, timeit
from .io_schema import rename_columns
from .qc import apply_basic_qc
from .features import engineer_features, enrich_with_module_labels


@timeit
def load_site(site_cfg: dict, columns_map: dict) -> pd.DataFrame:
    dfs = []
    for f in list_csvs(site_cfg['path']):
        df = pd.read_csv(f)
        df = rename_columns(df, columns_map)
        if 'DateTime' not in df.columns:
            raise ValueError('DateTime column missing; map it in YAML (preprocess.columns.datetime)')
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df = df.set_index('DateTime').sort_index()
        df['Site'] = site_cfg['name']
        # derive Module from filename like Cocoa_mSi0166.csv
        base = os.path.basename(f)
        m = re.match(r'^[A-Za-z]+_(.+?)\.csv$', base)
        df['Module'] = m.group(1) if m else os.path.splitext(base)[0]
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No CSV under {site_cfg['path']}")
    return pd.concat(dfs)


def main(config_path: str):
    cfg = load_cfg(config_path)
    out_dir = cfg['preprocess']['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    for site in cfg['data']['sites']:
        df = load_site(site, cfg['preprocess']['columns'])
        df = apply_basic_qc(
            df,
            cfg['preprocess'].get('drop_maintenance_windows', True),
            cfg['preprocess'].get('drop_when_precip_or_dew', True),
        )
        rule = cfg['preprocess'].get('resample_rule', '')
        if rule:
            df = df.resample(rule).mean()
        df = engineer_features(df)
        # attach per-module labels from configs/module_map.json if exists
        mp_json = os.path.join('configs', 'module_map.json')
        df = enrich_with_module_labels(df, mp_json)
        out_path = os.path.join(out_dir, f"{site['name']}.parquet")
        df.to_parquet(out_path)
        print(f"[Preprocess] Saved {out_path} ({len(df)} rows)")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    a = p.parse_args()
    main(a.config)
