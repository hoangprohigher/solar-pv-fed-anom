from __future__ import annotations
import os
import json
import joblib
import pandas as pd
from solarpv.utils import load_cfg
from solarpv.features import select_training_matrix
from solarpv.svdd import score_svdd, pred_svdd
from tensorflow.keras import layers, Model


def rebuild_vae(input_dim: int, hidden_dim: int, latent_dim: int, weights):
    inp = layers.Input(shape=(input_dim,))
    h = layers.Dense(hidden_dim, activation='relu')(inp)
    z_mean = layers.Dense(latent_dim)(h)
    z = z_mean
    dh = layers.Dense(hidden_dim, activation='relu')(z)
    out = layers.Dense(input_dim)(dh)
    vae = Model(inp, out)
    vae.set_weights(weights)
    return vae


def main(config_path: str, site: str, save_scores: bool, module: str | None = None):
    cfg = load_cfg(config_path)
    mdir = os.path.join(cfg['preprocess']['out_dir'], 'models')
    pq = os.path.join(cfg['preprocess']['out_dir'], f"{site}.parquet")
    df = pd.read_parquet(pq)
    df, X = select_training_matrix(df)
    prefix = f"{site}__{module}" if module else site
    with open(os.path.join(mdir, f"{prefix}_columns.json"), 'r') as f:
        columns = json.load(f)
    scaler = joblib.load(os.path.join(mdir, f"{prefix}_scaler.pkl"))
    try:
        weights = joblib.load(os.path.join(mdir, 'global_vae_weights.pkl'))
    except Exception:
        weights = joblib.load(os.path.join(mdir, f"{prefix}_vae_weights.pkl"))
    vae = rebuild_vae(
        len(columns),
        cfg['model']['hidden_dim'],
        cfg['model']['latent_dim'],
        weights,
    )
    Xs = scaler.transform(X[columns].values)
    Xrec = vae.predict(Xs, verbose=0)
    resid = Xs - Xrec
    oc = joblib.load(os.path.join(mdir, f"{prefix}_svdd.pkl"))
    scores = score_svdd(oc, resid)
    labels = pred_svdd(oc, resid)
    out = df.assign(score=scores, is_anom=(labels == -1).astype(int))
    if save_scores:
        os.makedirs(os.path.join(cfg['preprocess']['out_dir'], 'scores'), exist_ok=True)
        tag = f"{site}__{module}" if module else site
        outp = os.path.join(
            cfg['preprocess']['out_dir'],
            'scores',
            f"{tag}_scores.parquet",
        )
        out[['score', 'is_anom']].to_parquet(outp)
        print(f"[Score] Saved {outp}")
    else:
        print(out[['score', 'is_anom']].head())


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--site", required=True)
    p.add_argument("--module", required=False)
    p.add_argument("--save-scores", action="store_true")
    a = p.parse_args()
    main(a.config, a.site, a.save_scores, a.module)
