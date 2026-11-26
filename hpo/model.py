"""Exact finite generative model of what a supervisor can learn from a view.

For deciding how many errors are genuine, a view reduces to the summary
``s = (K, R)`` = (number of wrong lines, number of those carrying a tell), plus the
marker bit in marker mode.  The rest of the text (operands, positions, length) is
independent of T given s, and of the other view given (T, s) -- see README, "Why the
summary is enough".  The original reduces to T for the same reason.

Everything the theory needs is computed exactly from this finite model:
``sigma2`` (Brier gain for predicting "dirty" from a view), ``gamma`` (one minus the
squared maximal correlation between view and original), ``c_k`` and ``q0``.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binom

from .ledger import LedgerConfig


class ExactModel:
    def __init__(self, cfg: LedgerConfig):
        self.cfg = cfg
        self.pT = np.asarray(cfg.p_genuine, dtype=float)
        self.pD = cfg.plant_pmf()
        self.p = float(cfg.recognizability)
        self.marker = bool(cfg.marker)
        nT, nD = len(self.pT), len(self.pD)
        self.Tvals = np.arange(nT)

        table = {}                                   # state -> P(s | T) row
        for t in range(nT):
            for d in range(nD):
                for r in range(d + 1):
                    pr = self.pD[d] * binom.pmf(r, d, self.p)
                    if pr <= 0:
                        continue
                    s = (t + d, r, int(t > 0) if self.marker else 0)
                    table.setdefault(s, np.zeros(nT))[t] += pr
        self.states = sorted(table)
        self.index = {s: i for i, s in enumerate(self.states)}
        self.K = np.array([s[0] for s in self.states], dtype=float)
        self.R = np.array([s[1] for s in self.states], dtype=float)
        self.M = np.array([s[2] for s in self.states], dtype=float)
        self.PsT = np.array([table[s] for s in self.states]).T          # [T, S] = P(s|T)
        self.joint = self.pT[:, None] * self.PsT                        # P(T, s)
        self.Ps = self.joint.sum(0)                                     # P(s)
        self.PTs = self.joint / self.Ps                                 # [T, S] = P(T|s)
        self.p_dirty = 1.0 - self.PTs[0]                                # P(T>0 | s)
        self.ET = self.Tvals @ self.PTs                                 # E[T | s]
        # opponent-view kernel J(s, s') = sum_T P(T) P(s|T) P(s'|T)
        self.J = np.einsum("t,ts,tu->su", self.pT, self.PsT, self.PsT)

    # ---- quantities appearing in the theory ------------------------------------
    @property
    def p0(self) -> float:
        """P(original clean)."""
        return float(self.pT[0])

    @property
    def q0(self) -> float:
        """P(zero plants)."""
        return float(self.pD[0])

    @property
    def m(self) -> int:
        """Largest possible genuine score."""
        return int(self.Tvals[-1])

    @property
    def sigma2(self) -> float:
        """Brier gain over the base rate of the Bayes predictor of 1{dirty} from a
        view: Var(P(dirty | Z)) = p0 (1 - p0) - E Var(dirty | Z)."""
        return float(self.Ps @ (self.p_dirty - (1 - self.p0)) ** 2)

    @property
    def view_operator_eigenvalues(self) -> np.ndarray:
        """Eigenvalues of K(z, z') = sum_u P(u|z) P(z'|u), sorted descending.  Its
        non-zero spectrum equals that of the |T| x |T| matrix
        A(t, t') = sum_z P(z|t) P(t'|z), which is what we diagonalise."""
        A = self.PsT @ self.PTs.T
        ev = np.sort(np.linalg.eigvals(A).real)[::-1]
        return ev

    @property
    def gamma(self) -> float:
        """1 - (squared maximal correlation between a view and the original)
        = 1 - second eigenvalue of the view-to-view Markov operator."""
        return float(1.0 - self.view_operator_eigenvalues[1])

    def view_operator(self) -> np.ndarray:
        """The full S x S operator K(z, z') (used in tests to cross-check gamma)."""
        return self.PTs.T @ self.PsT

    def c_k(self) -> dict:
        """c_k = P(clean | view shows k errors), marginalising over tells."""
        out = {}
        for k in sorted(set(int(k) for k in self.K)):
            sel = self.K == k
            out[k] = float(self.joint[0, sel].sum() / self.Ps[sel].sum())
        return out

    # ---- lookups used by strategies ---------------------------------------------
    def lookup(self, values: np.ndarray, K, R, M=None, default=0.0) -> np.ndarray:
        """values[state] for arrays of observed (K, R[, M]); `default` off-support."""
        K = np.asarray(K); R = np.asarray(R)
        M = np.zeros_like(K) if (M is None or not self.marker) else np.asarray(M)
        kmax, rmax = int(self.K.max()) + 1, int(self.R.max()) + 1
        tab = np.full((kmax + 1, rmax + 1, 2), default, dtype=float)
        tab[self.K.astype(int), self.R.astype(int), self.M.astype(int)] = values
        return tab[np.clip(K, 0, kmax), np.clip(R, 0, rmax), M]
