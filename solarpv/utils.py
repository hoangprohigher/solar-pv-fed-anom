from __future__ import annotations
import os
import glob
import yaml
import time
import logging
import random
import numpy as np

log = logging.getLogger("solarpv")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def load_cfg(path: str) -> dict:
    with open(path,'r',encoding='utf-8') as f:
        return yaml.safe_load(f)

def list_csvs(folder: str) -> list[str]:
    return sorted(glob.glob(os.path.join(folder, '*.csv')))

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def timeit(fn):
    def wrap(*a, **k):
        t0=time.time()
        r=fn(*a, **k)
        log.info(f"{fn.__name__} took {time.time()-t0:.2f}s")
        return r
    return wrap
