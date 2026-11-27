"""Exact single-round model of the reviewer game, for design deliberation.

State: T genuine errors, Delta_i planted errors per reviewer (iid), R_i of them carry a tell
(each with prob p_tell), optional marker M = 1{T>0} visible in every copy.
View v = (K, R, M), K = T + Delta.  Strategy: view -> report >= 0.
Payoff_i = -beta*(adj_i - adj_j)^2 - lam*max(Delta_i - r_i, 0)^2 + b*clip(K_i - r_i, 0, T),
adj = r - Delta.  (Stake: b per genuine error missing from the adjusted report, capped at T.)
"""
import itertools, math, random
from collections import defaultdict

def poisson(m, kmax=8):
    w = [math.exp(-m) * m**k / math.factorial(k) for k in range(kmax + 1)]
    s = sum(w); return {k: x / s for k, x in enumerate(w)}

def uniform(vals):
    return {v: 1 / len(vals) for v in vals}

class Game:
    def __init__(s, pT, pD, b=0.5, beta=1.0, lam=1.0, p_tell=0.0, marker=False, perturb=True):
        s.pT, s.b, s.beta, s.lam, s.p_tell, s.marker = pT, b, beta, lam, p_tell, marker
        s.pD = pD if perturb else {0: 1.0}
        # single-reviewer conditional: P(Delta, R | nothing) and view given T
        s.single = []       # (Delta, R, prob)
        for d, pd in s.pD.items():
            for r in range(d + 1):
                pr = math.comb(d, r) * p_tell**r * (1 - p_tell)**(d - r)
                if pd * pr > 0: s.single.append((d, r, pd * pr))
        s.views = sorted({s.view(t, d, r) for t in pT for d, r, _ in s.single})
        # joint table: list of (prob, T, d1, v1, d2, v2)
        s.joint = [(pT[t] * p1 * p2, t, d1, s.view(t, d1, r1), d2, s.view(t, d2, r2))
                   for t in pT for d1, r1, p1 in s.single for d2, r2, p2 in s.single]
        s.by_v1 = defaultdict(list)
        for row in s.joint: s.by_v1[row[3]].append(row)
    def view(s, t, d, r):
        return (t + d, r, int(t > 0) if s.marker else 0)
    def honest(s): return {v: float(v[0]) for v in s.views}
    def own(s, t, d, k, r):
        return -s.lam * max(d - r, 0.0)**2 + s.b * min(max(k - r, 0.0), t)
    def payoff(s, s1, s2):
        tot = 0.0
        for p, t, d1, v1, d2, v2 in s.joint:
            r1, r2 = s1[v1], s2[v2]
            tot += p * (-s.beta * ((r1 - d1) - (r2 - d2))**2 + s.own(t, d1, v1[0], r1))
        return tot
    def W(s, st): return s.payoff(st, st)
    def concealed(s, st):
        num = sum(p * (v1[0] - st[v1]) for p, t, d1, v1, d2, v2 in s.joint if t > 0)
        return num / sum(s.pT[t] * t for t in s.pT)
    def clean_shift(s, st):
        return sum(p * (st[v1] - d1) for p, t, d1, v1, d2, v2 in s.joint if t == 0) / s.pT[0]
    def grid(s, v, step):
        n = int(round(v[0] / step)); return [i * step for i in range(n + 1)]
    def best_response(s, s2, step=1.0):
        br = {}
        for v1, rows in s.by_v1.items():
            best = None
            for r1 in s.grid(v1, step):
                val = sum(p * (-s.beta * ((r1 - d1) - (s2[v2] - d2))**2 + s.own(t, d1, v1[0], r1))
                          for p, t, d1, _, d2, v2 in rows)
                if best is None or val > best[0] + 1e-12: best = (val, r1)
            br[v1] = best[1]
        return br
    def exploitability(s, st, step=1.0):
        return s.payoff(s.best_response(st, step), st) - s.W(st)
    def pair_opt(s, step=1.0, restarts=30, seed=0):
        """Best symmetric strategy for the pair: coordinate ascent on W with restarts."""
        rng = random.Random(seed)
        starts = [s.honest(), {v: 0.0 for v in s.views}, {v: max(v[0] - 1.0, 0.0) for v in s.views},
                  {v: float(v[1]) for v in s.views}]
        dmax = max(s.pD)
        starts.append({v: float(min(v[0], dmax)) for v in s.views})
        for _ in range(restarts):
            starts.append({v: rng.choice(s.grid(v, step)) for v in s.views})
        best = None
        for st in starts:
            st = dict(st); cur = s.W(st); improved = True
            while improved:
                improved = False
                for v in s.views:
                    keep = st[v]
                    for r in s.grid(v, step):
                        if r == keep: continue
                        st[v] = r; val = s.W(st)
                        if val > cur + 1e-10: cur, keep, improved = val, r, True
                    st[v] = keep
            if best is None or cur > best[0] + 1e-10: best = (cur, dict(st))
        return best[1], best[0]
    def exhaustive(s):
        best = None
        for combo in itertools.product(*[range(v[0] + 1) for v in s.views]):
            st = {v: float(r) for v, r in zip(s.views, combo)}
            w = s.W(st)
            if best is None or w > best[0] + 1e-12: best = (w, st)
        return best[1], best[0]

def describe(g, st, tol=1e-9):
    """Compact description: gap K - r per view."""
    out = []
    for v in g.views:
        gap = v[0] - st[v]
        if abs(gap) > tol: out.append(f"{v}:-{gap:g}")
    return "honest" if not out else " ".join(out)


# ------------------------------------------------- tables quoted in docs/experiment2_design.md
if __name__ == "__main__":
    PT = {0: 0.6, 1: 0.3, 2: 0.1}

    def row(name, g, step=1.0, exhaustive=False):
        st, w = g.exhaustive() if exhaustive else g.pair_opt(step, restarts=80)
        ind = g.honest()
        for _ in range(50):
            nxt = g.best_response(ind, step)
            if nxt == ind:
                break
            ind = nxt
        print(f"{name:46s} concealed {g.concealed(st):5.2f}  gain/round {w:+.3f}  "
              f"from-honest {g.concealed(ind):4.2f}  {describe(g, st)[:80]}")

    print("A. Setup of the confirmatory run (Poisson(1.5) insertions), whole-number reports")
    P15 = poisson(1.5, 8)
    row("plain agreement, no insertion", Game(PT, P15, perturb=False))
    row("insertion, adjusted agreement only", Game(PT, P15, lam=0.0))
    row("insertion, floor only", Game(PT, P15, beta=0.0))
    row("insertion, both", Game(PT, P15))

    print()
    print("B. Proposed setup (1, 2 or 3 insertions per copy, equally likely), whole-number reports")
    U = uniform([1, 2, 3])
    row("1 plain agreement, no insertion", Game(PT, U, perturb=False), exhaustive=True)
    row("2 insertion, adjusted agreement only", Game(PT, U, lam=0.0), exhaustive=True)
    row("3 insertion, floor only", Game(PT, U, beta=0.0), exhaustive=True)
    row("4 insertion, both (the mechanism)", Game(PT, U), exhaustive=True)
    for p in (0.5, 0.9, 0.99, 1.0):
        row(f"   mechanism, {p:.0%} of insertions recognizable", Game(PT, U, p_tell=p))
    row("9 mechanism, header marks dirty ledgers", Game(PT, U, marker=True))
    row("10 mechanism, header re-randomized (= 4)", Game(PT, U), exhaustive=True)

    print()
    print("C. Where recognizability starts to pay (proposed setup, both parts)")
    for p in (0.9, 0.93, 0.95, 0.97):
        row(f"   {p:.0%} recognizable", Game(PT, U, p_tell=p))

    print()
    print("D. Proposed setup with reports in steps of 0.25")
    for name, kw in (("adjusted agreement only", dict(lam=0.0)), ("floor only", dict(beta=0.0)), ("both", {})):
        row("   " + name, Game(PT, U, **kw), step=0.25)
