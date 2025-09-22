from __future__ import annotations
from solarpv.preprocess import main
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    a = p.parse_args()
    main(a.config)
