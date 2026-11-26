"""Menu of supervisor strategies.

Every strategy maps what is readable from the view -- K wrong lines, R of them with a
tell, marker bit M -- to a real report F.  ``belief`` is an :class:`ExactModel` of the
(publicly known) generating process, used by the Bayesian strategies; no strategy ever
sees T or Delta.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from .model import ExactModel

REPORT_MAX = 30.0   # larger than any possible error count (<= 20 + 2 + 8 lines)


def honest(K, R, M, belief):
    """Report every error found."""
    return np.asarray(K, float)


def hide_one(K, R, M, belief):
    """Report max(count - 1, 0): hide one error, never report a negative count."""
    return np.maximum(np.asarray(K, float) - 1.0, 0.0)


def shift_minus_one(K, R, M, belief):
    """Exact uniform shift: report count - 1 (so -1 on an error-free view)."""
    return np.asarray(K, float) - 1.0


def hide_unrecognized(K, R, M, belief):
    """'Hide all recognised-genuine': report the errors that carry a tell (certainly
    planted) plus the posterior-mean number of plants among the rest.  Equivalently
    F = K - E[T | view]: every error believed genuine is withheld."""
    return np.asarray(K, float) - belief.lookup(belief.ET, K, R, M)


def posterior_shade_50(K, R, M, belief, theta=0.5):
    """Posterior-mean shading: withhold a fraction theta of the believed-genuine
    errors, F = K - theta E[T | view]."""
    return np.asarray(K, float) - theta * belief.lookup(belief.ET, K, R, M)


def posterior_shade_10(K, R, M, belief):
    """Mild posterior-mean shading (theta = 0.1): a small step away from honesty."""
    return posterior_shade_50(K, R, M, belief, theta=0.1)


def always_zero(K, R, M, belief):
    return np.zeros(np.shape(K), float)


def report_max(K, R, M, belief):
    return np.full(np.shape(K), REPORT_MAX)


def over_one(K, R, M, belief):
    """Harmless over-reporting: count + 1."""
    return np.asarray(K, float) + 1.0


def marker_hide_one(K, R, M, belief):
    """Hide one error exactly when the benign-content marker is present."""
    K = np.asarray(K, float)
    return np.where(np.asarray(M) == 1, np.maximum(K - 1.0, 0.0), K)


MENU: Dict[str, Callable] = {
    "honest": honest,
    "hide_one": hide_one,
    "shift_minus_one": shift_minus_one,
    "hide_unrecognized": hide_unrecognized,
    "posterior_shade_50": posterior_shade_50,
    "posterior_shade_10": posterior_shade_10,
    "always_zero": always_zero,
    "report_max": report_max,
    "over_one": over_one,
}

MARKER_MENU: Dict[str, Callable] = dict(MENU, marker_hide_one=marker_hide_one)


def evaluate_menu(batch: dict, belief: ExactModel, reward, menu=None):
    """Monte Carlo cost matrix on one batch of paired views.

    Returns (names, C) with C[d, s] = mean cost to supervisor 1 when it plays d and
    supervisor 2 plays s.  The same samples are used for every cell (common random
    numbers), so differences between cells are paired.
    """
    from .mechanism import cost
    menu = menu or MENU
    names = list(menu)
    F1 = [menu[n](batch["K1"], batch["R1"], batch["M1"], belief) for n in names]
    F2 = [menu[n](batch["K2"], batch["R2"], batch["M2"], belief) for n in names]
    C = np.empty((len(names), len(names)))
    for d in range(len(names)):
        for s in range(len(names)):
            C[d, s] = cost(reward, F1[d], F2[s], batch["D1"], batch["D2"], batch["T"]).mean()
    return names, C


def deviation_gains(C: np.ndarray) -> np.ndarray:
    """G[s, d] = C[s, s] - C[d, s]: what a supervisor saves by unilaterally switching
    from the symmetric profile s to d.  Positive means the deviation is profitable."""
    return np.diag(C)[:, None] - C.T
