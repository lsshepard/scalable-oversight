"""Recognizable insertions on a language model's features: the outcome against how much of
the tell the supervisor's features expose, with hidden insertions at the same supervisor
sizes as control.

    python experiments/exp9_sweeps.py results/trained/qwen05_cpu

Needs the features of the conditions `rec100` and `hidden` in that folder (exp9_trained.py
extract).  A supervisor of size k reads k principal directions per layer plus the three
probability features.  "Tells exposed" is the share of tells a linear probe on those features
recovers on test ledgers.  Writes tell_sweep.json and capacity_sweep_hidden.json.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from exp9_trained import load, reduce_features  # noqa: E402
from hpo.mechanism import Reward  # noqa: E402
from hpo.trained import TrainConfig, train_pair  # noqa: E402

SIZES = (2, 8, 24, 48)
REWARDS = (("no private gain", Reward(beta=1, lam=1, b=0.0, eps=0.02)),
           ("private gain 0.5", Reward(beta=1, lam=1, b=0.5, eps=0.02)))
KEEP = ("net_share", "concealed_dirty", "under_clean", "charge", "w_system", "w_inserted_plain", "w_inserted_tell",
        "w_ok", "rmse_count")


def main():
    from sklearn.linear_model import LogisticRegression
    torch.set_num_threads(1)
    feats = pathlib.Path(sys.argv[1])
    hidden = json.loads((feats / "meta.json").read_text())["hidden"]
    for cond, out in (("rec100", "tell_sweep.json"), ("hidden", "capacity_sweep_hidden.json")):
        (dtr, ftr), (dte, fte) = load(feats, cond, "train"), load(feats, cond, "test")
        rows = []
        for k in SIZES:
            rtr, rte = reduce_features(ftr, fte, dtr["status"] >= 0, hidden, k)
            recall = fa = float("nan")
            if cond == "rec100":
                mtr, mte = dtr["status"] >= 0, dte["status"] >= 0
                mu, sd = rtr[0][mtr].mean(0), rtr[0][mtr].std(0) + 1e-6
                clf = LogisticRegression(C=1.0, max_iter=1000).fit((rtr[0][mtr] - mu) / sd, dtr["tell"][mtr])
                q, t = clf.predict((rte[0][mte] - mu) / sd), dte["tell"][mte]
                recall, fa = float(q[t].mean()), float(q[~t].mean())
            for label, rw in REWARDS:
                for seed in (0, 1):
                    res = train_pair(dtr, dte, rtr, rte, rw, TrainConfig(steps=3000, batch=256, eval_every=250, hidden=16,
                                                                         seed=seed, val_frac=0.2))
                    f = res["final"]
                    rows.append(dict(k=k, tell_recall=recall, tell_false_alarms=fa, label=label, seed=seed,
                                     step=res["selected_step"], **{kk: f[kk] for kk in KEEP}))
                    print(f"{cond} k={k:2d} {label:16s} seed {seed}: withheld {f['net_share']:+.3f}  weight system "
                          f"{f['w_system']:+.2f}", flush=True)
        (feats / out).write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
