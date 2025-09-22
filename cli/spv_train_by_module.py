from __future__ import annotations
import os
import json
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from solarpv.utils import load_cfg
from solarpv.features import select_training_matrix
from solarpv.vae import build_vae_keras
from solarpv.svdd import fit_svdd


def train_one(
    dfX: pd.DataFrame,
    columns: list[str],
    cfg: dict,
    out_dir: str,
    prefix: str,
):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(dfX[columns].values)
    vae, enc = build_vae_keras(
        Xs.shape[1],
        cfg['model']['hidden_dim'],
        cfg['model']['latent_dim'],
        cfg['model']['learning_rate'],
    )
    vae.fit(
        Xs,
        Xs,
        epochs=cfg['model']['epochs_local'],
        batch_size=cfg['model']['batch_size'],
        verbose=0,
    )
    Xrec = vae.predict(Xs, verbose=0)
    resid = Xs - Xrec
    oc = fit_svdd(resid, cfg['svdd']['nu'], cfg['svdd']['gamma'])
    joblib.dump(scaler, os.path.join(out_dir, f"{prefix}_scaler.pkl"))
    joblib.dump(oc, os.path.join(out_dir, f"{prefix}_svdd.pkl"))
    joblib.dump(vae.get_weights(), os.path.join(out_dir, f"{prefix}_vae_weights.pkl"))
    with open(os.path.join(out_dir, f"{prefix}_columns.json"), 'w') as f:
        json.dump(columns, f)
    print(f"[TrainModule] Saved artifacts: {prefix}")

def main(config_path: str):
    cfg = load_cfg(config_path)
    out_dir = os.path.join(cfg['preprocess']['out_dir'], 'models')
    os.makedirs(out_dir, exist_ok=True)
    for site in cfg['data']['sites']:
        pq = os.path.join(cfg['preprocess']['out_dir'], f"{site['name']}.parquet")
        base = pd.read_parquet(pq)
        full_df, X = select_training_matrix(base)
        columns = list(X.columns)
        if 'Module' not in base.columns:
            # Fallback: train one model for whole site
            train_one(X, columns, cfg, out_dir, prefix=f"{site['name']}")
            continue
        for module, df_mod in base.loc[X.index].groupby('Module'):
            dfX = X.loc[df_mod.index]
            prefix = f"{site['name']}__{module}".replace('/', '_')
            train_one(dfX, columns, cfg, out_dir, prefix)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    a = p.parse_args()
    main(a.config)
