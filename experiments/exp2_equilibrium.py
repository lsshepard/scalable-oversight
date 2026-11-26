"""Experiment 2: equilibrium over all real-valued reporting functions of the summary.

(a) For each reward: minimise the convex potential with L-BFGS; run simultaneous
    gradient play (exact gradients, random initialisations) and sampled (minibatch)
    gradient play, 5 seeds each; report under-reporting on dirty and clean originals.
(b) Prediction 1: under agreement only with a stake the potential is unbounded below;
    best-response and gradient dynamics conceal more and more.
(c) Prediction 3: sweep stake b, beta and lam and compare concealment with the bound
    b (sigma^2 / (beta gamma) + m^2 / (lam p0)).
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import (BETA, LAM, P_DEFAULT, SEEDS, SERIES, STAKE, dump_json,
                                outdir, reward_grid, style)
from hpo.equilibrium import (Potential, exploitability, gradient_play, sampled_gradient_play,
                             solve_equilibrium, weighted_distance)
from hpo.ledger import LedgerConfig
from hpo.mechanism import Reward
from hpo.model import ExactModel
from hpo.stats import mean_ci


def bounded(rw: Reward) -> bool:
    """Psi is bounded below iff the floor is on, or nothing pushes reports down."""
    return rw.lam > 0 or (rw.b == 0 and rw.eps == 0)


def theory_bound(model: ExactModel, rw: Reward) -> float:
    return rw.b * (model.sigma2 / (rw.beta * model.gamma) + model.m ** 2 / (rw.lam * model.p0))


def table_a(model):
    rows = []
    for rw in reward_grid():
        pot = Potential(model, rw)
        row = dict(reward=rw.name, beta=rw.beta, lam=rw.lam, b=rw.b, eps=rw.eps,
                   bounded=bounded(rw),
                   honest_exploitability=exploitability(model, rw, np.zeros(pot.S)))
        if bounded(rw):
            eq = solve_equilibrium(model, rw)
            c = pot.concealment(0.5 * (eq.g1 + eq.g2))
            row.update(eq_exploitability=exploitability(model, rw, 0.5 * (eq.g1 + eq.g2)), psi=eq.psi, lbfgs_grad=eq.grad_norm, lbfgs_dirty=c["dirty"],
                       lbfgs_clean=c["clean"], lbfgs_clean_floor_viol=c["clean_neg"],
                       lbfgs_asym=weighted_distance(model, eq.g1, eq.g2))
            for label, play in (("gp", lambda s: gradient_play(model, rw, s, iters=30000)[:2]),
                                ("sgp", lambda s: sampled_gradient_play(model, rw, s))):
                res = [play(s) for s in SEEDS]
                dirty = [pot.concealment(0.5 * (a + b))["dirty"] for a, b in res]
                clean = [pot.concealment(0.5 * (a + b))["clean"] for a, b in res]
                dist = [weighted_distance(model, 0.5 * (a + b), 0.5 * (eq.g1 + eq.g2)) for a, b in res]
                psi_gap = [pot.value_grad(a, b)[0] - eq.psi for a, b in res]
                for nm, v in (("dirty", dirty), ("clean", clean), ("dist", dist), ("psi_gap", psi_gap)):
                    m, ci = mean_ci(v)
                    row[f"{label}_{nm}"], row[f"{label}_{nm}_ci"] = float(m), float(ci)
        else:
            # unbounded: record how far gradient play has run after a fixed horizon
            res = [gradient_play(model, rw, s, iters=30000)[:2] for s in SEEDS]
            dirty = [pot.concealment(0.5 * (a + b))["dirty"] for a, b in res]
            m, ci = mean_ci(dirty)
            row.update(gp_dirty=float(m), gp_dirty_ci=float(ci))
        rows.append(row)
    return pd.DataFrame(rows)


def unbounded_demo(model, d):
    """Prediction 1 with a stake: concealment along the dynamics, no floor vs floor."""
    rows = []
    horizons = [0, 1000, 2000, 4000, 8000, 16000, 32000]
    for name, rw in (("agreement only", Reward(BETA, 0.0, 0.0, STAKE)),
                     ("agreement + floor", Reward(BETA, LAM, 0.0, STAKE))):
        pot = Potential(model, rw)
        for seed in SEEDS:
            _, _, tr = gradient_play(model, rw, seed, iters=horizons[-1], init_scale=0.0,
                                     step=1.0 / (4 * BETA + 2 * LAM), record_every=1000)
            for it, c in tr:
                if it in horizons:
                    rows.append(dict(reward=name, seed=seed, iteration=it, concealment_dirty=c))
        # along the uniform shift g = -x the potential is linear in x without a floor
        for x in (0, 1, 10, 100, 1000):
            g = -x * np.ones(pot.S)
            rows.append(dict(reward=name, seed=-1, iteration=-1, shift=x,
                             psi=pot.value_grad(g, g)[0]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(d, "unbounded_demo.csv"), index=False)
    # exact best-response ladder under agreement only: each round adds b P(dirty|s)/(2 beta)
    rw = Reward(BETA, 0.0, 0.0, STAKE)
    g = np.zeros(len(model.states))
    ladder = []
    pot = Potential(model, rw)
    for rnd in range(0, 201):
        if rnd in (0, 1, 10, 50, 100, 200):
            ladder.append(dict(round=rnd, concealment_dirty=pot.concealment(g)["dirty"]))
        g = (model.J @ g) / model.Ps - rw.b * model.p_dirty / (2 * rw.beta)
    pd.DataFrame(ladder).to_csv(os.path.join(d, "agreement_only_br_ladder.csv"), index=False)

    plot_unbounded(df, d)
    return df, pd.DataFrame(ladder)


def bound_sweep(model, d):
    rows = []
    grid = [0.25, 0.5, 1.0, 2.0, 4.0, 16.0, 64.0]
    sweeps = [("b", [0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8])] + [("beta", grid), ("lam", grid)]
    for var, vals in sweeps:
        for v in vals:
            kw = dict(beta=BETA, lam=LAM, b=STAKE, eps=0.0)
            kw[var] = v
            rw = Reward(**kw)
            eq = solve_equilibrium(model, rw)
            pot = Potential(model, rw)
            c = pot.concealment(0.5 * (eq.g1 + eq.g2))
            rows.append(dict(sweep=var, beta=rw.beta, lam=rw.lam, b=rw.b,
                             concealment_dirty=c["dirty"], concealment_uncond=c["uncond"],
                             under_clean=c["clean"], bound=theory_bound(model, rw),
                             agreement_part=rw.b * model.sigma2 / (rw.beta * model.gamma),
                             floor_part=rw.b * model.m ** 2 / (rw.lam * model.p0),
                             grad=eq.grad_norm))
    df = pd.DataFrame(rows)
    df["ratio_dirty_to_bound"] = df.concealment_dirty / df.bound.replace(0, np.nan)
    df.to_csv(os.path.join(d, "bound_sweep.csv"), index=False)

    plot_bound(df, d)
    return df


def plot_unbounded(df, d):
    style()
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    tr = df[df.iteration >= 0].groupby(["reward", "iteration"]).concealment_dirty.mean().reset_index()
    for i, name in enumerate(("agreement only", "agreement + floor")):
        t = tr[tr.reward == name]
        ax.plot(t.iteration, t.concealment_dirty, marker="o", color=SERIES[i], label=name)
    ax.set_xlabel("gradient-play iteration (both supervisors start honest)")
    ax.set_ylabel("genuine errors concealed per dirty original")
    ax.set_title(f"Stake b={STAKE:g}: concealment diverges without the floor")
    ax.legend(loc="upper left")
    fig.savefig(os.path.join(d, "fig_unbounded.png"))
    plt.close(fig)


def plot_bound(df, d):
    style()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    labels = {"b": "stake b", "beta": "agreement weight beta", "lam": "floor weight lambda"}
    for ax, var in zip(axes, ("b", "beta", "lam")):
        t = df[(df.sweep == var) & (df.b > 0)]
        h1, = ax.plot(t[var], t.bound, marker="s", color=SERIES[1])
        h0, = ax.plot(t[var], t.concealment_dirty, marker="o", color=SERIES[0])
        ax.set_xlabel(labels[var])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
    axes[0].set_ylabel("genuine errors concealed")
    fig.suptitle("Equilibrium concealment vs. the bound b(sigma^2/(beta gamma) + m^2/(lambda p0))",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.legend([h1, h0], ["theory bound", "equilibrium concealment per dirty original (exact)"],
               loc="upper right", ncol=2, fontsize=8, bbox_to_anchor=(0.99, 0.93))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(d, "fig_bound_sweep.png"))
    plt.close(fig)


def main():
    d = outdir("equilibrium")
    model = ExactModel(LedgerConfig(recognizability=P_DEFAULT))
    dump_json(dict(p=P_DEFAULT, sigma2=model.sigma2, gamma=model.gamma, p0=model.p0,
                   q0=model.q0, m=model.m, n_states=len(model.states)),
              os.path.join(d, "model_constants.json"))
    ta = table_a(model)
    ta.to_csv(os.path.join(d, "equilibrium_by_reward.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(ta[["reward", "bounded", "honest_exploitability", "eq_exploitability", "psi", "lbfgs_dirty", "lbfgs_clean", "gp_dirty", "gp_dirty_ci",
              "gp_clean", "gp_dist", "sgp_dirty", "sgp_dirty_ci", "sgp_dist"]].round(4).to_string())
    demo, ladder = unbounded_demo(model, d)
    print(demo[demo.iteration >= 0].groupby(["reward", "iteration"]).concealment_dirty.mean())
    print(demo[demo.iteration < 0][["reward", "shift", "psi"]])
    print(ladder)
    bs = bound_sweep(model, d)
    print(bs.round(4).to_string())


if __name__ == "__main__":
    main()
