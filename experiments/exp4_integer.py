"""Experiment 4: integer-only reports (prediction 5).

No tells (p = 0), no stake.  For a grid of floor weights lam, start both supervisors
at hide-one, F(k) = max(k - 1, 0), and run alternating best-response iteration over
non-negative integer reports.  Prediction: hide-one is a stable equilibrium iff
lam c_k < beta (1 - 2 c_k q0) for all k >= 1.  With real-valued reports it never is.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import SERIES, dump_json, outdir, style
from hpo.equilibrium import (Potential, hide_one_threshold, integer_best_response,
                             integer_br_iteration, solve_equilibrium)
from hpo.ledger import LedgerConfig
from hpo.mechanism import Reward
from hpo.model import ExactModel


def plot(df, thr, d):
    style()
    t = df[df.beta == 1.0]
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    ax.plot(t.lam, t.best_unilateral_gain, marker="o", ms=3.5, color=SERIES[0],
            label="integer reports: best unilateral gain at hide-one")
    ax.plot(t.lam, t.hide_one_psi / 2 - t.real_eq_psi / 2, color=SERIES[1], ls="-",
            label="real reports: potential drop available from hide-one")
    ax.axvline(thr["lam_crit"], color="#666666", lw=1, ls="--")
    ax.annotate(f"predicted threshold\nlambda = {thr['lam_crit']:.3f}", (thr["lam_crit"], ax.get_ylim()[1] * 0.55),
                textcoords="offset points", xytext=(6, 0), color="#666666", fontsize=8)
    ax.set_xlabel("floor weight lambda (beta = 1, no tells, no stake)")
    ax.set_ylabel("expected cost saved by deviating")
    ax.set_title("Hide-one survives integer reports below the threshold")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.savefig(os.path.join(d, "fig_integer.png"))
    plt.close(fig)



def main():
    d = outdir("integer")
    model = ExactModel(LedgerConfig(recognizability=0.0))
    hide = np.maximum(model.K - 1, 0)
    actions = np.arange(0, int(model.K.max()) + 4, dtype=float)
    rows = []
    for beta in (0.5, 1.0, 2.0):
        thr = hide_one_threshold(model, beta)
        lams = sorted(set(np.round(np.arange(0.05, 2.5001, 0.05) * beta, 4))
                      | {round(thr["lam_crit"] - 1e-3, 6), round(thr["lam_crit"] + 1e-3, 6)})
        for lam in lams:
            rw = Reward(beta, lam)
            pot = Potential(model, rw)
            f1, f2, rounds, stable = integer_br_iteration(model, rw, hide)
            br, Q = integer_best_response(model, rw, hide, hide, actions)
            cur = Q[np.arange(len(hide)), hide.astype(int)]
            gain = float(model.Ps @ (cur - Q.min(1)))           # best unilateral gain at hide-one
            # real-valued reports: slope of own cost at hide-one, and where L-BFGS goes from there
            g_hide = hide - model.K
            _, grad, _ = pot.value_grad(g_hide, g_hide)
            eq = solve_equilibrium(model, rw, x0=(g_hide, g_hide))
            rows.append(dict(
                beta=beta, lam=lam, lam_crit=thr["lam_crit"],
                predicted_stable=bool(lam < thr["lam_crit"]), hide_one_stable=stable,
                agree=bool((lam < thr["lam_crit"]) == stable),
                best_unilateral_gain=gain,
                states_that_deviate=";".join(str(int(k)) for k in model.K[br != hide]),
                br_rounds=rounds,
                limit_is_honest=bool(np.array_equal(f1, model.K) and np.array_equal(f2, model.K)),
                limit_is_hide_one=bool(np.array_equal(f1, hide) and np.array_equal(f2, hide)),
                limit_concealment_dirty=pot.concealment(0.5 * (f1 + f2) - model.K)["dirty"],
                limit_f1=" ".join(str(int(v)) for v in f1),
                real_grad_norm_at_hide_one=float(np.sqrt(((grad / model.Ps) ** 2) @ model.Ps)),
                real_eq_from_hide_one_concealment=pot.concealment(0.5 * (eq.g1 + eq.g2))["dirty"],
                real_eq_psi=eq.psi, hide_one_psi=pot.value_grad(g_hide, g_hide)[0]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(d, "integer_hide_one.csv"), index=False)
    thr = hide_one_threshold(model, 1.0)
    dump_json(dict(thr, all_grid_points_agree=bool(df.agree.all()), n_grid=len(df)),
              os.path.join(d, "threshold.json"))
    print("threshold (beta=1):", thr["lam_crit"], "binding k:", thr["binding_k"],
          "c_1:", thr["c_k"][1], "q0:", thr["q0"])
    print("prediction matches at every grid point:", bool(df.agree.all()), f"({len(df)} points)")
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(df[df.beta == 1.0][["lam", "predicted_stable", "hide_one_stable", "best_unilateral_gain",
                              "states_that_deviate", "br_rounds", "limit_is_honest",
                              "limit_concealment_dirty", "real_grad_norm_at_hide_one",
                              "real_eq_from_hide_one_concealment"]].round(4).to_string())

    plot(df, thr, d)


if __name__ == "__main__":
    main()
