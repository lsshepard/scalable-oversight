"""Experiment 3: recognizability of plants (prediction 4).

For p in P_SWEEP: exact sigma^2 and gamma; a fitted discriminator's held-out Brier
gain (5 seeds, trained on features parsed from generated ledger text); equilibrium
concealment with a stake (L-BFGS, cross-checked by gradient play from 5 random
initialisations).  A dense exact grid in p checks smoothness.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import BETA, LAM, P_SWEEP, SEEDS, SERIES, STAKE, dump_json, outdir, style
from experiments.exp2_equilibrium import theory_bound
from hpo.discriminator import fit_discriminators
from hpo.equilibrium import Potential, gradient_play, solve_equilibrium
from hpo.ledger import LedgerConfig, sample_text_batch
from hpo.mechanism import Reward
from hpo.model import ExactModel
from hpo.stats import mean_ci

N_TRAIN = N_TEST = 10000
RW = Reward(BETA, LAM, 0.0, STAKE)


def exact_row(p):
    model = ExactModel(LedgerConfig(recognizability=p))
    pot = Potential(model, RW)
    eq = solve_equilibrium(model, RW)
    c = pot.concealment(0.5 * (eq.g1 + eq.g2))
    gam = model.gamma
    return model, dict(p=p, sigma2=model.sigma2, gamma=gam,
                       sigma2_over_gamma=model.sigma2 / gam if gam > 1e-9 else np.inf,
                       concealment_dirty=c["dirty"], concealment_uncond=c["uncond"],
                       under_clean=c["clean"],
                       bound=theory_bound(model, RW) if gam > 1e-9 else np.inf,
                       lbfgs_grad=eq.grad_norm)


def plot(df, dense, d):
    style()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    ax = axes[0]
    ax.plot(dense.p, dense.concealment_dirty, color=SERIES[0], label="exact equilibrium (L-BFGS)")
    ax.errorbar(df.p, df.gp_concealment_dirty, yerr=df.gp_concealment_dirty_ci, fmt="o",
                color=SERIES[1], mec="white", mew=1, label="gradient play, 5 seeds", zorder=3)
    ax.set_xlabel("recognizability p (share of plants with a tell)")
    ax.set_ylabel("genuine errors concealed per dirty original")
    ax.set_title("Concealment vs. recognizability")
    ax.legend(loc="upper left")
    ax = axes[1]
    fin = dense[np.isfinite(dense.sigma2_over_gamma)]
    ax.plot(fin.sigma2_over_gamma, fin.concealment_dirty, color=SERIES[0], label="exact equilibrium")
    pts = df[np.isfinite(df.sigma2_over_gamma)]
    ax.plot(pts.sigma2_over_gamma, pts.concealment_dirty, "o", color=SERIES[0], mec="white", mew=1)
    for r in pts[pts.p.isin([0.0, 0.8, 0.95])].itertuples():
        ax.annotate(f"p={r.p:g}", (r.sigma2_over_gamma, r.concealment_dirty),
                    textcoords="offset points", xytext=(5, -9), fontsize=7, color="#666666")
    ax.set_xscale("log")
    ax.set_xlabel("sigma^2 / gamma  (log scale; infinite at p = 1)")
    ax.set_ylabel("genuine errors concealed per dirty original")
    ax.set_title("Concealment vs. sigma^2 / gamma")
    ax = axes[2]
    ax.plot(dense.p, dense.sigma2, color=SERIES[0], label="exact sigma^2")
    ax.errorbar(df.p, df.brier_gain_mlp, yerr=df.brier_gain_mlp_ci, fmt="o", color=SERIES[1],
                mec="white", mew=1, label="MLP Brier gain (held out)", zorder=3)
    ax.errorbar(df.p, df.brier_gain_logreg, yerr=df.brier_gain_logreg_ci, fmt="s", color=SERIES[2],
                mec="white", mew=1, label="logistic Brier gain (held out)", zorder=3)
    ax.set_xlabel("recognizability p")
    ax.set_ylabel("Brier gain over base rate")
    ax.set_title("Discriminator estimate of sigma^2")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "fig_recognizability.png"))
    plt.close(fig)



def main():
    d = outdir("recognizability")
    rows = []
    for p in P_SWEEP:
        model, row = exact_row(p)
        pot = Potential(model, RW)
        gains = {"logreg": [], "mlp": []}
        for seed in SEEDS:
            cfg = LedgerConfig(recognizability=p)
            tr = sample_text_batch(cfg, N_TRAIN, np.random.default_rng(2000 + seed))
            te = sample_text_batch(cfg, N_TEST, np.random.default_rng(3000 + seed))
            for k, v in fit_discriminators(tr, te, seed).items():
                gains[k].append(v)
        for k, v in gains.items():
            m, ci = mean_ci(v)
            row[f"brier_gain_{k}"], row[f"brier_gain_{k}_ci"] = float(m), float(ci)
        gp = [pot.concealment(0.5 * sum(gradient_play(model, RW, s, iters=40000)[:2]))["dirty"]
              for s in SEEDS]
        m, ci = mean_ci(gp)
        row["gp_concealment_dirty"], row["gp_concealment_dirty_ci"] = float(m), float(ci)
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(d, "recognizability_sweep.csv"), index=False)

    grid = sorted(set(np.round(np.linspace(0, 1, 101), 2)) | {0.995, 0.999, 0.9999, 0.99999})
    dense = pd.DataFrame([exact_row(p)[1] for p in grid])
    dense.to_csv(os.path.join(d, "recognizability_dense_exact.csv"), index=False)
    dc, dp = np.diff(dense.concealment_dirty), np.diff(dense.p)
    smooth = dict(monotone_increasing=bool(np.all(dc > -1e-7)),
                  largest_step=float(dc.max()), largest_step_at_p=float(dense.p[dc.argmax() + 1]),
                  max_slope=float((dc / dp).max()), max_slope_at_p=float(dense.p[(dc / dp).argmax() + 1]),
                  gap_between_p_0p99999_and_1=float(dc[-1]),
                  concealment_at_p0=float(dense.concealment_dirty.iloc[0]),
                  concealment_at_p1=float(dense.concealment_dirty.iloc[-1]))
    dump_json(smooth, os.path.join(d, "smoothness.json"))
    print(smooth)

    plot(df, dense, d)


if __name__ == "__main__":
    main()
