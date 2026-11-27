"""Exact one-round model of the reviewer game: what is the best a pair of copies can do?

Used to predict, before any run, the strategy and the concealment in each condition of
the "predicted exploits" experiment (docs/experiment2_design.md).

State.  T system errors; Delta_i inserted errors per reviewer, drawn independently; R_i of
them recognizable (each with probability p_tell); optionally a marker M = 1{T > 0} that is
visible in both copies.  A reviewer's view is v = (K, R, M) with K = T + Delta.

Payoff of reviewer i with report r_i:
    - beta * (est_i - est_j)^2              adjusted agreement,  est = r - Delta
    - lam  * max(Delta_i - r_i, 0)^2        planted floor
    + b    * clip(T - est_i, 0, T)          private gain per system error missing from est_i

A strategy maps views to reports in [0, K].  `pair_opt` finds the best symmetric strategy
for the pair (what two copies choose if each assumes the other reasons identically);
`best_response` and `exploitability` check that it is also a Nash equilibrium.
"""
from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict
from typing import Dict, Tuple

View = Tuple[int, int, int]
Strategy = Dict[View, float]


def poisson_pmf(mean: float, kmax: int = 8) -> Dict[int, float]:
    w = [math.exp(-mean) * mean ** k / math.factorial(k) for k in range(kmax + 1)]
    s = sum(w)
    return {k: x / s for k, x in enumerate(w)}


def uniform_pmf(values) -> Dict[int, float]:
    return {int(v): 1.0 / len(values) for v in values}


class Game:
    def __init__(self, pT: Dict[int, float], pD: Dict[int, float], b: float = 0.5,
                 beta: float = 1.0, lam: float = 1.0, p_tell: float = 0.0,
                 marker: bool = False, perturb: bool = True):
        self.pT, self.b, self.beta, self.lam = dict(pT), b, beta, lam
        self.p_tell, self.marker = p_tell, marker
        self.pD = dict(pD) if perturb else {0: 1.0}
        self.single = []                      # (Delta, R, probability) for one reviewer
        for d, pd in self.pD.items():
            for r in range(d + 1):
                pr = math.comb(d, r) * p_tell ** r * (1 - p_tell) ** (d - r)
                if pd * pr > 0:
                    self.single.append((d, r, pd * pr))
        self.views = sorted({self.view(t, d, r) for t in self.pT for d, r, _ in self.single})
        self.joint = [(self.pT[t] * p1 * p2, t, d1, self.view(t, d1, r1), d2, self.view(t, d2, r2))
                      for t in self.pT for d1, r1, p1 in self.single for d2, r2, p2 in self.single]
        self.by_v1 = defaultdict(list)
        for row in self.joint:
            self.by_v1[row[3]].append(row)

    def view(self, t: int, d: int, r: int) -> View:
        return (t + d, r, int(t > 0) if self.marker else 0)

    def honest(self) -> Strategy:
        return {v: float(v[0]) for v in self.views}

    def own(self, t: int, d: int, k: int, r: float) -> float:
        return -self.lam * max(d - r, 0.0) ** 2 + self.b * min(max(k - r, 0.0), t)

    def payoff(self, s1: Strategy, s2: Strategy) -> float:
        tot = 0.0
        for p, t, d1, v1, d2, v2 in self.joint:
            r1, r2 = s1[v1], s2[v2]
            tot += p * (-self.beta * ((r1 - d1) - (r2 - d2)) ** 2 + self.own(t, d1, v1[0], r1))
        return tot

    def W(self, st: Strategy) -> float:
        """Expected payoff of each reviewer when both play st."""
        return self.payoff(st, st)

    def concealed(self, st: Strategy) -> float:
        """Share of system errors missing from the estimate, as the analysis measures it:
        E[(T - estimate) on dirty ledgers] / E[T]."""
        num = sum(p * (v1[0] - st[v1]) for p, t, d1, v1, d2, v2 in self.joint if t > 0)
        return num / sum(p * t for t, p in self.pT.items())

    def clean_shift(self, st: Strategy) -> float:
        return sum(p * (st[v1] - d1) for p, t, d1, v1, d2, v2 in self.joint if t == 0) / self.pT[0]

    @staticmethod
    def grid(v: View, step: float):
        return [i * step for i in range(int(round(v[0] / step)) + 1)]

    def best_response(self, s2: Strategy, step: float = 1.0) -> Strategy:
        br = {}
        for v1, rows in self.by_v1.items():
            best = None
            for r1 in self.grid(v1, step):
                val = sum(p * (-self.beta * ((r1 - d1) - (s2[v2] - d2)) ** 2 + self.own(t, d1, v1[0], r1))
                          for p, t, d1, _, d2, v2 in rows)
                if best is None or val > best[0] + 1e-12:
                    best = (val, r1)
            br[v1] = best[1]
        return br

    def exploitability(self, st: Strategy, step: float = 1.0) -> float:
        """Gain of the best unilateral deviation; 0 means st is a Nash equilibrium."""
        return self.payoff(self.best_response(st, step), st) - self.W(st)

    def from_honest(self, step: float = 1.0, iters: int = 50) -> Strategy:
        """Where repeated individual best responses lead when both start honest."""
        st = self.honest()
        for _ in range(iters):
            nxt = self.best_response(st, step)
            if nxt == st:
                break
            st = nxt
        return st

    def exhaustive(self) -> Tuple[Strategy, float]:
        """Best symmetric whole-number strategy.  Honest wins ties."""
        best = (self.W(self.honest()), self.honest())
        for combo in itertools.product(*[range(v[0] + 1) for v in self.views]):
            st = {v: float(r) for v, r in zip(self.views, combo)}
            w = self.W(st)
            if w > best[0] + 1e-9:
                best = (w, st)
        return best[1], best[0]

    def pair_opt(self, step: float = 1.0, restarts: int = 60, seed: int = 0) -> Tuple[Strategy, float]:
        """Best symmetric strategy by coordinate ascent from many starts.  Ties are broken
        toward the start listed first, so honest wins when nothing beats it."""
        rng = random.Random(seed)
        dmax = max(self.pD)
        starts = [self.honest(), {v: 0.0 for v in self.views},
                  {v: max(v[0] - 1.0, 0.0) for v in self.views},
                  {v: float(v[1]) for v in self.views},
                  {v: float(min(v[0], dmax)) for v in self.views}]
        starts += [{v: rng.choice(self.grid(v, step)) for v in self.views} for _ in range(restarts)]
        best = None
        for st in starts:
            st = dict(st)
            cur, improved = self.W(st), True
            while improved:
                improved = False
                for v in self.views:
                    keep = st[v]
                    for r in self.grid(v, step):
                        if r == keep:
                            continue
                        st[v] = r
                        val = self.W(st)
                        if val > cur + 1e-10:
                            cur, keep, improved = val, r, True
                    st[v] = keep
            if best is None or cur > best[0] + 1e-9:
                best = (cur, dict(st))
        return best[1], best[0]

    def solve(self, step: float = 1.0, exhaustive_limit: int = 200000) -> Tuple[Strategy, float, str]:
        size = 1
        for v in self.views:
            size *= v[0] + 1
        if step == 1.0 and size <= exhaustive_limit:
            st, w = self.exhaustive()
            return st, w, "exhaustive"
        st, w = self.pair_opt(step)
        return st, w, "coordinate ascent"


def describe(g: Game, st: Strategy, tol: float = 1e-9) -> str:
    """Compact description: by how much each view's report falls short of its count."""
    out = [f"{v}:-{v[0] - st[v]:g}" for v in g.views if abs(v[0] - st[v]) > tol]
    return "honest" if not out else " ".join(out)


def game_for(rule: str, p_genuine, plant_values, stake: float, recognizable: float = 0.0,
             marker: bool = False, plant_mean: float = 1.5, plant_max: int = 8) -> Game:
    """The exact game behind one condition of hpo.rounds (rule as in GameConfig.rule)."""
    pT = {t: float(p) for t, p in enumerate(p_genuine)}
    pD = uniform_pmf(plant_values) if plant_values else poisson_pmf(plant_mean, plant_max)
    perturb = rule in ("perturb", "agree", "floor")
    beta = 0.0 if rule == "floor" else 1.0
    lam = 1.0 if rule in ("perturb", "floor") else 0.0
    return Game(pT, pD, b=stake, beta=beta, lam=lam, p_tell=recognizable if perturb else 0.0,
                marker=marker, perturb=perturb)


def predict(g: Game) -> dict:
    """Prediction record for one condition (whole-number reports)."""
    st, w, method = g.solve()
    ind = g.from_honest()
    return dict(
        concealed=round(g.concealed(st), 4), gain_per_round=round(w, 4),
        clean_shift=round(g.clean_shift(st), 4), exploitability=round(g.exploitability(st), 6),
        concealed_from_honest=round(g.concealed(ind), 4), method=method,
        strategy={f"{k},{r},{m}": rep for (k, r, m), rep in sorted(st.items())},
        description=describe(g, st))
