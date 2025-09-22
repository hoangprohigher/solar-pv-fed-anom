from __future__ import annotations
import numpy as np

def fedavg(weight_sets: list[list[np.ndarray]]) -> list[np.ndarray]:
    if not weight_sets:
        raise ValueError("No client weights")
    out = []
    for i in range(len(weight_sets[0])):
        stack = np.stack([ws[i] for ws in weight_sets], axis=0)
        out.append(stack.mean(axis=0))
    return out
