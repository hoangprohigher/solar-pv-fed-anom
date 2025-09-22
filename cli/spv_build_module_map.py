from __future__ import annotations
import os, json
from solarpv.utils import load_cfg
from solarpv.module_map import build_module_map_from_files

def main(config_path: str, out_json: str):
    cfg = load_cfg(config_path)
    mp = build_module_map_from_files(cfg['data']['sandia_coeffs_xlsx'], cfg['data']['characterization_xlsx'])
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json,'w',encoding='utf-8') as f:
        json.dump(mp, f, indent=2, ensure_ascii=False)
    print(f"[ModuleMap] Wrote {out_json}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", default="configs/module_map.json")
    a = p.parse_args()
    main(a.config, a.out)
