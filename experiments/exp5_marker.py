"""Experiment 5: a marker that survives perturbation (prediction 6).

In marker mode the parity of the ledger id (benign content the perturbation never
touches) is odd iff the original is dirty.  We compare, with and without the marker:
 * Monte Carlo cost of 'hide one error when the marker is present' and the gain from
   deviating to it from honesty (text-generated ledgers, 10 seeds);
 * the exact equilibrium over all reporting functions for several stakes, together
   with the mechanism penalty (agreement + floor) the supervisors actually pay.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from experiments.common import BETA, LAM, MENU_SEEDS, P_DEFAULT, STAKE, dump_json, outdir
from experiments.exp1_menu import run_env, to_markdown
from hpo.equilibrium import Potential, gradient_play, solve_equilibrium
from hpo.ledger import LedgerConfig
from hpo.mechanism import Reward
from hpo.model import ExactModel
from hpo.stats import mean_ci
from hpo.strategies import MARKER_MENU
from experiments.common import SEEDS


def main():
    d = outdir("marker")
    rewards = [Reward(BETA, LAM, 0.0, 0.0, "both"), Reward(BETA, LAM, 0.0, STAKE, "both+stake")]
    summs = []
    for marker in (True, False):
        cfg = LedgerConfig(recognizability=P_DEFAULT, marker=marker)
        lg, sm = run_env(cfg, MARKER_MENU, rewards, tag=f"marker={'on' if marker else 'off'}")
        lg.to_csv(os.path.join(d, f"marker_{'on' if marker else 'off'}_all_cells.csv"), index=False)
        summs.append(sm)
    sm = pd.concat(summs)
    sm.to_csv(os.path.join(d, "marker_menu_summary.csv"), index=False)
    with open(os.path.join(d, "marker_menu_tables.md"), "w") as fh:
        fh.write(to_markdown(sm))
    print(to_markdown(sm))
    # the headline cells: honest -> marker_hide_one
    cells = []
    for marker in ("on", "off"):
        lg = pd.read_csv(os.path.join(d, f"marker_{marker}_all_cells.csv"))
        c = lg[(lg.profile == "honest") & (lg.deviation == "marker_hide_one")]
        cells.append(c)
    cells = pd.concat(cells)
    cells.to_csv(os.path.join(d, "honest_to_marker_hide_one.csv"), index=False)
    print(cells.to_string())

    rows = []
    for marker in (True, False):
        model = ExactModel(LedgerConfig(recognizability=P_DEFAULT, marker=marker))
        for b in (0.0, 0.001, 0.01, 0.05, STAKE, 0.8):
            rw = Reward(BETA, LAM, 0.0, b)
            pot = Potential(model, rw)
            eq = solve_equilibrium(model, rw)
            g = 0.5 * (eq.g1 + eq.g2)
            c = pot.concealment(g)
            penalty = pot.player_cost(g, g) + b * c["uncond"]        # agreement + floor only
            gp = [pot.concealment(0.5 * sum(gradient_play(model, rw, s, iters=30000)[:2]))["dirty"]
                  for s in SEEDS]
            m, ci = mean_ci(gp)
            rows.append(dict(marker=marker, b=b, sigma2=model.sigma2, gamma=model.gamma,
                             concealment_dirty=c["dirty"], under_clean=c["clean"],
                             mechanism_penalty_paid=penalty,
                             gp_concealment_dirty=float(m), gp_concealment_dirty_ci=float(ci)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(d, "marker_equilibrium.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(df.round(5).to_string())


if __name__ == "__main__":
    main()
