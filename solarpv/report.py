from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt


def simple_report(scores_parquet: str, out_html: str):
    df = pd.read_parquet(scores_parquet)
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    fig = plt.figure()
    df['score'].plot(title='Anomaly score')
    png = out_html.replace('.html', '.png')
    fig.savefig(png, bbox_inches='tight')
    with open(out_html, 'w', encoding='utf-8') as f:
        img = os.path.basename(png)
        f.write(f"<h1>Solar PV Anomaly Report</h1><img src='{img}'/>")
