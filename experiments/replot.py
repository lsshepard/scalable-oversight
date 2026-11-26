"""Redraw every figure from the CSVs under results/ without recomputing anything."""
import json
import os

import pandas as pd

from experiments import exp2_equilibrium, exp3_recognizability, exp4_integer
from experiments.common import outdir

if __name__ == "__main__":
    d = outdir("equilibrium")
    exp2_equilibrium.plot_unbounded(pd.read_csv(os.path.join(d, "unbounded_demo.csv")), d)
    exp2_equilibrium.plot_bound(pd.read_csv(os.path.join(d, "bound_sweep.csv")), d)
    d = outdir("recognizability")
    exp3_recognizability.plot(pd.read_csv(os.path.join(d, "recognizability_sweep.csv")),
                              pd.read_csv(os.path.join(d, "recognizability_dense_exact.csv")), d)
    d = outdir("integer")
    with open(os.path.join(d, "threshold.json")) as fh:
        thr = json.load(fh)
    exp4_integer.plot(pd.read_csv(os.path.join(d, "integer_hide_one.csv")), thr, d)
