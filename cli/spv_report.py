from __future__ import annotations
import os
from solarpv.utils import load_cfg
from solarpv.report import simple_report

def main(config_path: str):
    cfg = load_cfg(config_path)
    sdir = os.path.join(cfg['preprocess']['out_dir'], 'scores')
    site = cfg['data']['sites'][0]['name']
    sp = os.path.join(sdir, f"{site}_scores.parquet")
    out_html = os.path.join(cfg['report']['out_dir'], f"{site}_summary.html")
    simple_report(sp, out_html)
    print(f"[Report] {out_html}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    a = p.parse_args()
    main(a.config)
