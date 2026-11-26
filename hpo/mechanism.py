"""The hidden-perturbation mechanism's charge to a supervisor."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Reward:
    """cost_i = beta (V_i - V_j)^2 + lam max(Delta_i - F_i, 0)^2 + eps F_i
                - b 1{T > 0} (T - V_i),          V = F - Delta.

    beta : weight on adjusted agreement
    lam  : weight on the planted floor
    eps  : per-unit reporting charge
    b    : outside stake (bonus per unit of concealed genuine misconduct)
    """
    beta: float = 1.0
    lam: float = 1.0
    eps: float = 0.0
    b: float = 0.0
    name: str = ""

    def label(self) -> str:
        return self.name or f"beta={self.beta:g},lam={self.lam:g},b={self.b:g},eps={self.eps:g}"


def cost(rw: Reward, F_i, F_j, D_i, D_j, T):
    """Per-sample cost to supervisor i (vectorised)."""
    F_i = np.asarray(F_i, float); F_j = np.asarray(F_j, float)
    V_i, V_j = F_i - D_i, F_j - D_j
    c = rw.beta * (V_i - V_j) ** 2 + rw.lam * np.maximum(D_i - F_i, 0.0) ** 2
    c = c + rw.eps * F_i - rw.b * (np.asarray(T) > 0) * (T - V_i)
    return c
