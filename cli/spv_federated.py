from __future__ import annotations
import os
import joblib
from solarpv.utils import load_cfg
from solarpv.fedavg import fedavg


def main(config_path: str):
    cfg = load_cfg(config_path)
    mdir = os.path.join(cfg['preprocess']['out_dir'], 'models')
    weight_sets = []
    for site in cfg['data']['sites']:
        w = joblib.load(os.path.join(mdir, f"{site['name']}_vae_weights.pkl"))
        weight_sets.append(w)
    gw = fedavg(weight_sets)
    joblib.dump(gw, os.path.join(mdir, 'global_vae_weights.pkl'))
    msg = f"[FedAvg] Wrote global_vae_weights.pkl from {len(weight_sets)} clients"
        print(msg)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    a = p.parse_args()
    main(a.config)
