from __future__ import annotations

import numpy as np
from sklearn.svm import OneClassSVM

def fit_svdd(residuals: np.ndarray, nu: float = 0.1, gamma: str = 'scale') -> OneClassSVM:
    oc = OneClassSVM(
        kernel='rbf', nu=nu, gamma=gamma
    )
    oc.fit(residuals)
    return oc


def score_svdd(model: OneClassSVM, residuals: np.ndarray):
    return model.decision_function(residuals)


def pred_svdd(model: OneClassSVM, residuals: np.ndarray):
    return model.predict(residuals)
