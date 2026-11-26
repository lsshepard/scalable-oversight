"""Learned estimate of sigma^2: how well can 'the original is dirty' be predicted from
one view?  sigma^2 is the Brier-score gain of the Bayes predictor over the base rate;
the held-out Brier gain of any fitted classifier estimates it from below."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def view_features(batch: dict, which: str = "1") -> np.ndarray:
    """Features readable from a view: errors, tells, untold errors, length, marker."""
    K, R = batch["K" + which], batch["R" + which]
    return np.column_stack([K, R, K - R, batch["L" + which], batch["M" + which]]).astype(float)


def brier_gain(y_true, p_hat, base_rate) -> float:
    y = np.asarray(y_true, float)
    return float(np.mean((y - base_rate) ** 2) - np.mean((y - p_hat) ** 2))


def fit_discriminators(train: dict, test: dict, seed: int) -> dict:
    """Fit a logistic regression and a small MLP on the train batch; return held-out
    Brier gains over the (train) base rate."""
    Xtr, ytr = view_features(train), (train["T"] > 0).astype(int)
    Xte, yte = view_features(test), (test["T"] > 0).astype(int)
    base = ytr.mean()
    out = {}
    models = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=2000)),
        "mlp": make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(32, 32), alpha=1e-4,
                                           max_iter=400, early_stopping=True,
                                           random_state=seed)),
    }
    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        out[name] = brier_gain(yte, mdl.predict_proba(Xte)[:, 1], base)
    return out
