"""Equilibria over ALL reporting functions of the view summary.

A reporting function is a vector over summary states.  We parametrise by the shading
``g(s) = F(s) - K(s)``, so the adjusted report is ``V = F - Delta = T + g(s)`` and
honest reporting is g = 0 (negative g = under-reporting).

The game is an exact potential game with convex potential

    Psi(g1, g2) = beta E (V_1 - V_2)^2 + lam sum_i E max(-V_i, 0)^2
                  - b sum_i E[1{dirty} (T - V_i)] + eps sum_i E F_i ,

whose partial gradient in g_i equals the gradient of supervisor i's own expected cost.
Nash equilibria in real-valued reporting functions are exactly its minimisers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .mechanism import Reward
from .model import ExactModel


class Potential:
    def __init__(self, model: ExactModel, rw: Reward):
        self.m, self.rw = model, rw
        self.S = len(model.states)
        self.dirty_w = model.joint[1:].sum(0)        # P(dirty, s)

    # -- pieces --------------------------------------------------------------------
    def _own(self, g):
        """Terms of Psi that involve one player only, and their gradient."""
        m, rw = self.m, self.rw
        neg = np.maximum(-(g[None, :] + m.Tvals[:, None]), 0.0)          # [T, S]
        val = rw.lam * (m.joint * neg ** 2).sum() + rw.b * self.dirty_w @ g \
            + rw.eps * m.Ps @ (g + m.K)
        grad = -2 * rw.lam * (m.joint * neg).sum(0) + rw.b * self.dirty_w + rw.eps * m.Ps
        return val, grad

    def value_grad(self, g1, g2):
        m, rw = self.m, self.rw
        v1, d1 = self._own(g1)
        v2, d2 = self._own(g2)
        Jg1, Jg2 = m.J @ g1, m.J @ g2
        agree = m.Ps @ g1 ** 2 + m.Ps @ g2 ** 2 - 2 * g1 @ Jg2
        return (rw.beta * agree + v1 + v2,
                2 * rw.beta * (m.Ps * g1 - Jg2) + d1,
                2 * rw.beta * (m.Ps * g2 - Jg1) + d2)

    def player_cost(self, g_own, g_other):
        """Expected cost of one supervisor (not the potential)."""
        m, rw = self.m, self.rw
        v, _ = self._own(g_own)
        agree = m.Ps @ g_own ** 2 + m.Ps @ g_other ** 2 - 2 * g_own @ (m.J @ g_other)
        return rw.beta * agree + v

    # -- summaries -----------------------------------------------------------------
    def concealment(self, g) -> dict:
        """Under-reporting implied by shading g (all exact expectations).

        dirty        E[T - V | T > 0]     mean genuine misconduct withheld, dirty originals
        clean        E[0 - V | T = 0]     mean under-report on clean originals
        clean_neg    E[max(-V, 0) | T = 0] floor violations on clean originals
        uncond       E[1{dirty} (T - V)]
        """
        m = self.m
        pd = 1 - m.p0
        return {
            "dirty": float(-(self.dirty_w @ g) / pd),
            "clean": float(-(m.joint[0] @ g) / m.p0),
            "clean_neg": float(m.joint[0] @ np.maximum(-g, 0) / m.p0),
            "uncond": float(-(self.dirty_w @ g)),
        }


@dataclass
class EqResult:
    g1: np.ndarray
    g2: np.ndarray
    psi: float
    grad_norm: float
    converged: bool


def solve_equilibrium(model: ExactModel, rw: Reward, x0=None) -> EqResult:
    """Minimise Psi over two independent function vectors with L-BFGS.

    Variables are rescaled by sqrt(P(s)) (the L2(P) geometry), which keeps rare
    summaries from being ignored by the stopping rule.
    """
    pot = Potential(model, rw)
    S, w = pot.S, np.sqrt(model.Ps)

    def fg(x):
        g1, g2 = x[:S] / w, x[S:] / w
        v, d1, d2 = pot.value_grad(g1, g2)
        return v, np.concatenate([d1 / w, d2 / w])

    x0 = np.zeros(2 * S) if x0 is None else np.concatenate([x0[0] * w, x0[1] * w])
    res = minimize(fg, x0, jac=True, method="L-BFGS-B",
                   options=dict(maxiter=20000, maxfun=40000, ftol=1e-16, gtol=1e-11))
    g1, g2 = res.x[:S] / w, res.x[S:] / w
    return EqResult(g1, g2, float(res.fun), float(np.abs(res.jac).max()), bool(res.success))


def exploitability(model: ExactModel, rw: Reward, g: np.ndarray) -> float:
    """Largest expected-cost saving available to one supervisor from a unilateral
    deviation to ANY real-valued reporting function when both currently use shading g.
    Zero iff g is a symmetric equilibrium.  (inf if the best response is unbounded.)"""
    if rw.lam == 0 and rw.beta == 0 and (rw.b > 0 or rw.eps > 0):
        return float("inf")
    pot = Potential(model, rw)
    w = np.sqrt(model.Ps)

    def fg(x):
        gi = x / w
        _, d1, _ = pot.value_grad(gi, g)
        return pot.player_cost(gi, g), d1 / w

    res = minimize(fg, g * w, jac=True, method="L-BFGS-B",
                   options=dict(maxiter=20000, maxfun=40000, ftol=1e-16, gtol=1e-11))
    return float(pot.player_cost(g, g) - res.fun)


def gradient_play(model: ExactModel, rw: Reward, seed: int, iters: int = 20000,
                  init_scale: float = 1.0, step: float | None = None, record_every: int = 0):
    """Simultaneous gradient play: both supervisors descend their OWN expected cost at
    the same time, each view-state using the gradient of its conditional expected cost
    (i.e. gradient / P(s)).  Starts from a random N(0, init_scale^2) shading.

    Returns (g1, g2, trace) where trace is a list of (iteration, dirty-concealment).
    """
    pot = Potential(model, rw)
    rng = np.random.default_rng(seed)
    g1 = init_scale * rng.standard_normal(pot.S)
    g2 = init_scale * rng.standard_normal(pot.S)
    eta = step if step is not None else 1.0 / (4 * rw.beta + 2 * rw.lam + 1e-12)
    trace = []
    for it in range(iters):
        if record_every and it % record_every == 0:
            trace.append((it, pot.concealment(0.5 * (g1 + g2))["dirty"]))
        _, d1, d2 = pot.value_grad(g1, g2)
        g1, g2 = g1 - eta * d1 / model.Ps, g2 - eta * d2 / model.Ps
    if record_every:
        trace.append((iters, pot.concealment(0.5 * (g1 + g2))["dirty"]))
    return g1, g2, trace


def sampled_gradient_play(model: ExactModel, rw: Reward, seed: int, iters: int = 4000,
                          batch: int = 4096, init_scale: float = 1.0):
    """Stochastic version: each round both supervisors see a fresh minibatch of paired
    views and take a per-state gradient step on their own realised cost (step size
    decaying as 1/sqrt(t)).  This is what 'training two monitors against the
    mechanism' looks like in the tabular case."""
    rng = np.random.default_rng(seed)
    S, nT = len(model.states), len(model.pT)
    g = init_scale * rng.standard_normal((2, S))
    cum = np.cumsum(model.PsT, axis=1)
    eta0 = 1.0 / (4 * rw.beta + 2 * rw.lam + 1e-12)
    avg, navg = np.zeros((2, S)), 0
    for it in range(iters):
        T = rng.choice(nT, size=batch, p=model.pT)
        s = []
        for _ in range(2):
            u, si = rng.random(batch), np.empty(batch, dtype=np.int64)
            for t in range(nT):
                sel = T == t
                si[sel] = np.searchsorted(cum[t], u[sel])
            s.append(np.minimum(si, S - 1))
        V = [g[i][s[i]] + T for i in range(2)]
        eta = eta0 / np.sqrt(1 + it / 200.0)
        new = g.copy()
        for i in range(2):
            j = 1 - i
            dV = 2 * rw.beta * (V[i] - V[j]) - 2 * rw.lam * np.maximum(-V[i], 0) \
                + rw.b * (T > 0) + rw.eps
            num = np.bincount(s[i], weights=dV, minlength=S)
            cnt = np.bincount(s[i], minlength=S)
            new[i] -= eta * num / np.maximum(cnt, 1)
        g = new
        if it >= iters // 2:
            avg += g
            navg += 1
    avg /= navg
    return avg[0], avg[1]


def weighted_distance(model: ExactModel, g, g_ref) -> float:
    """L2(P) distance between two shading functions."""
    return float(np.sqrt(model.Ps @ (g - g_ref) ** 2))


# ---------------------------------------------------------------------------------
# Integer-only reports
# ---------------------------------------------------------------------------------
def integer_cost_table(model: ExactModel, rw: Reward, f_other: np.ndarray,
                       actions: np.ndarray) -> np.ndarray:
    """Q[s, a] = conditional expected cost of reporting integer a in state s when the
    other supervisor uses the report function f_other (vector over states)."""
    m = model
    T = m.Tvals[:, None, None]                         # [T, 1, 1]
    V_own = actions[None, None, :] - m.K[None, :, None] + T          # [T, S, A]
    g_o = f_other - m.K
    EVo = (m.PsT @ g_o)[:, None, None] + T             # E[V_j | T]
    EVo2 = (m.PsT @ g_o ** 2)[:, None, None] + 2 * T * (m.PsT @ g_o)[:, None, None] + T ** 2
    agree = V_own ** 2 - 2 * V_own * EVo + EVo2
    c = rw.beta * agree + rw.lam * np.maximum(-V_own, 0) ** 2 \
        - rw.b * (T > 0) * (T - V_own) + rw.eps * actions[None, None, :]
    return np.einsum("ts,tsa->sa", m.PTs, c)


def integer_best_response(model, rw, f_other, f_current, actions, tol=1e-12):
    """Best integer response; keeps the current report on (near-)ties, otherwise the
    smallest minimiser."""
    Q = integer_cost_table(model, rw, f_other, actions)
    best = Q.min(1)
    cur_idx = np.searchsorted(actions, f_current)
    keep = Q[np.arange(len(best)), cur_idx] <= best + tol
    return np.where(keep, f_current, actions[Q.argmin(1)]), Q


def integer_br_iteration(model, rw, f_start, max_rounds=500):
    """Alternating best-response iteration over non-negative integer reports.

    Returns (f1, f2, rounds, stable_at_start): `stable_at_start` says whether f_start
    is itself a (symmetric, pure) equilibrium, i.e. nobody moved in the first round.
    """
    actions = np.arange(0, int(model.K.max()) + 4, dtype=float)
    f1, f2 = f_start.copy(), f_start.copy()
    stable_at_start = None
    for rnd in range(max_rounds):
        n1, _ = integer_best_response(model, rw, f2, f1, actions)
        n2, _ = integer_best_response(model, rw, n1, f2, actions)
        moved = not (np.array_equal(n1, f1) and np.array_equal(n2, f2))
        if rnd == 0:
            b1, _ = integer_best_response(model, rw, f_start, f_start, actions)
            stable_at_start = bool(np.array_equal(b1, f_start))
        f1, f2 = n1, n2
        if not moved:
            return f1, f2, rnd, stable_at_start
    return f1, f2, max_rounds, stable_at_start


def hide_one_threshold(model: ExactModel, beta: float) -> dict:
    """Prediction 5: hide-one is a stable integer equilibrium iff
    lam * c_k < beta * (1 - 2 c_k q0) for every k >= 1.  Returns the critical lam
    (min over k of beta (1 - 2 c_k q0) / c_k) and the binding k."""
    ck = {k: v for k, v in model.c_k().items() if k >= 1 and v > 0}
    crit = {k: beta * (1 - 2 * c * model.q0) / c for k, c in ck.items()}
    kstar = min(crit, key=crit.get)
    return {"lam_crit": crit[kstar], "binding_k": kstar, "c_k": ck, "q0": model.q0}
