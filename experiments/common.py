"""Shared settings for all experiments."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hpo.mechanism import Reward

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# Default mechanism parameters used throughout unless a sweep says otherwise.
BETA, LAM, STAKE, EPS = 1.0, 1.0, 0.2, 0.02
P_DEFAULT = 0.5                     # recognizability used for the menu / equilibrium tables
MENU_SEEDS = list(range(10))        # Monte Carlo replications (unit of the CI)
SEEDS = list(range(5))              # everything else that is random
P_SWEEP = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]


def reward_grid():
    """agreement only / floor only / both, x stake off/on, x reporting charge off/on."""
    out = []
    for kind, (beta, lam) in (("agreement_only", (BETA, 0.0)), ("floor_only", (0.0, LAM)),
                              ("both", (BETA, LAM))):
        for b in (0.0, STAKE):
            for eps in (0.0, EPS):
                name = kind + ("+stake" if b else "") + ("+eps" if eps else "")
                out.append(Reward(beta, lam, eps, b, name))
    return out


def outdir(name: str) -> str:
    d = os.path.join(RESULTS, name)
    os.makedirs(d, exist_ok=True)
    return d


def dump_json(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=float)


# Figure style: thin marks, recessive grid, fixed categorical order, single y-axis.
SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17"]
INK, INK2 = "#191919", "#666666"


def style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.edgecolor": "#cccccc", "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#ebebeb", "grid.linewidth": 0.8,
        "lines.linewidth": 2.0, "lines.markersize": 5, "legend.frameon": False,
        "axes.prop_cycle": plt.cycler(color=SERIES),
    })
