"""Experiment 1: strategy menu, no training.

For every reward, estimate by Monte Carlo on text-generated ledgers the expected cost
of each strategy when both supervisors use it and the gain from unilaterally deviating
to every other strategy.  Predictions 1, 2 and the 'only truth is free' half of 3.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from experiments.common import MENU_SEEDS, P_DEFAULT, outdir, reward_grid
from hpo.equilibrium import Potential
from hpo.ledger import LedgerConfig, sample_text_batch
from hpo.model import ExactModel
from hpo.stats import mean_ci
from hpo.strategies import MENU, deviation_gains, evaluate_menu

N_PER_SEED = 10000


def run_env(cfg: LedgerConfig, menu, rewards, seeds=MENU_SEEDS, n=N_PER_SEED, tag=""):
    """Returns (long dataframe of all cells, per-strategy summary dataframe)."""
    belief = ExactModel(cfg)
    names = list(menu)
    C = np.empty((len(rewards), len(seeds), len(names), len(names)))
    for si, seed in enumerate(seeds):
        batch = sample_text_batch(cfg, n, np.random.default_rng(1000 + seed))
        for ri, rw in enumerate(rewards):
            _, C[ri, si] = evaluate_menu(batch, belief, rw, menu)
    long_rows, summ_rows = [], []
    for ri, rw in enumerate(rewards):
        G = np.stack([deviation_gains(C[ri, si]) for si in range(len(seeds))])
        Cm, Cci = mean_ci(C[ri])
        Gm, Gci = mean_ci(G)
        pot = Potential(belief, rw)
        for s, sn in enumerate(names):
            for d, dn in enumerate(names):
                long_rows.append(dict(env=tag, reward=rw.name, profile=sn, deviation=dn,
                                      cost_dev_vs_profile=Cm[d, s], cost_ci=Cci[d, s],
                                      gain=Gm[s, d], gain_ci=Gci[s, d]))
            others = [d for d in range(len(names)) if d != s]
            best = max(others, key=lambda d: Gm[s, d])
            # not an equilibrium: some deviation's gain has its CI wholly above zero;
            # strict: every deviation's gain has its CI wholly below zero;
            # otherwise weak (exact ties) or unresolved (CI straddles zero)
            profitable = [names[d] for d in others if Gm[s, d] - Gci[s, d] > 1e-12]
            ties = [names[d] for d in others if abs(Gm[s, d]) <= 1e-12 and Gci[s, d] <= 1e-12]
            g = menu[sn](belief.K.astype(int), belief.R.astype(int), belief.M.astype(int),
                         belief) - belief.K
            summ_rows.append(dict(
                env=tag, reward=rw.name, strategy=sn,
                sym_cost=Cm[s, s], sym_cost_ci=Cci[s, s], sym_cost_exact=pot.player_cost(g, g),
                best_deviation=names[best], best_gain=Gm[s, best], best_gain_ci=Gci[s, best],
                menu_equilibrium=len(profitable) == 0,
                strict=all(Gm[s, d] + Gci[s, d] < -1e-12 for d in others),
                costless_alternatives=";".join(ties)))
    return pd.DataFrame(long_rows), pd.DataFrame(summ_rows)


def to_markdown(summ: pd.DataFrame) -> str:
    out = []
    for (env, rw), df in summ.groupby(["env", "reward"], sort=False):
        out.append(f"\n**{rw}** ({env})\n")
        out.append("| strategy (both use it) | expected cost | best unilateral deviation | gain from it | equilibrium in menu |")
        out.append("|---|---|---|---|---|")
        for r in df.itertuples():
            eq = "no" if not r.menu_equilibrium else ("yes (strict)" if r.strict else "yes (weak or unresolved)")
            out.append(f"| {r.strategy} | {r.sym_cost:.4f} ± {r.sym_cost_ci:.4f} | {r.best_deviation} "
                       f"| {r.best_gain:+.4f} ± {r.best_gain_ci:.4f} | {eq} |")
    return "\n".join(out) + "\n"


def main():
    d = outdir("menu")
    rewards = reward_grid()
    longs, summs = [], []
    for p in (P_DEFAULT, 0.0):
        cfg = LedgerConfig(recognizability=p)
        lg, sm = run_env(cfg, MENU, rewards, tag=f"p={p:g}")
        longs.append(lg); summs.append(sm)
    lg, sm = pd.concat(longs), pd.concat(summs)
    lg.to_csv(os.path.join(d, "menu_all_cells.csv"), index=False)
    sm.to_csv(os.path.join(d, "menu_summary.csv"), index=False)
    with open(os.path.join(d, "menu_tables.md"), "w") as fh:
        fh.write(to_markdown(sm))
    print(to_markdown(sm[sm.env == f"p={P_DEFAULT:g}"]))
    print("max |MC - exact| symmetric cost:", (sm.sym_cost - sm.sym_cost_exact).abs().max())


if __name__ == "__main__":
    main()
