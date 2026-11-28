"""Summary figure and table for section 8: frozen heads, adapter (gradient), adapter (pay only),
each against the exact equilibrium, in the four conditions the adapter runs cover.

    python experiments/fig_e2e.py            # reads results/trained/main/{analysis/table.csv,e2e.json}
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "results" / "trained" / "main"
CONDS = [("plain agreement", "Plain agreement"),
         ("both terms", "Both terms\n(the mechanism)"),
         ("both terms, insertions recognizable", "Both terms,\ninsertions recognizable"),
         ("both terms, no stake", "Both terms,\nno stake")]
CLIP = -0.6


def load():
    frozen = {}
    for r in csv.DictReader(open(ROOT / "analysis" / "table.csv")):
        if r["estimator"] == "analytic":
            frozen[r["label"]] = (float(r["net_share_mean"]), float(r["exact_net_share"] or "nan"))
    e2e = defaultdict(list)
    for r in json.load(open(ROOT / "e2e.json")):
        e2e[(r["label"], "reinforce" if "pay only" in r["estimator"] else "analytic")].append(r)
    return frozen, e2e


def main():
    frozen, e2e = load()
    series = [("Frozen features + head (gradient)", "#257de0", "o"),
              ("Adapter (gradient)", "#f47230", "s"),
              ("Adapter (pay only, RL)", "#00b894", "D")]
    rows = []
    for key, _ in CONDS:
        vals = [frozen[key][0]]
        seeds = []
        for est in ("analytic", "reinforce"):
            rs = e2e.get((key, est), [])
            vals.append(float(np.mean([r["final"]["net_share"] for r in rs])) if rs else np.nan)
            seeds.append(len(rs))
        rows.append(dict(cond=key, vals=vals, exact=frozen[key][1], seeds=seeds,
                         wsys=[float(np.mean([r["final"]["w_system"] for r in e2e[(key, est)]]))
                               for est in ("analytic", "reinforce")],
                         rmse=[float(np.mean([r["final"]["rmse_count"] for r in e2e[(key, est)]]))
                               for est in ("analytic", "reinforce")]))

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    x = np.arange(len(CONDS))
    off = [-0.22, 0, 0.22]
    for j, (name, col, mk) in enumerate(series):
        y = np.array([r["vals"][j] for r in rows])
        yc = np.maximum(y, CLIP)
        ax.scatter(x + off[j], yc, s=46, color=col, marker=mk, label=name, zorder=3,
                   edgecolor="white", linewidth=1.5)
        for xi, yi, yo in zip(x + off[j], yc, y):
            if yo < CLIP:
                ax.annotate(f"{yo:.1f}, off scale", (xi, yi), xytext=(0, -14), textcoords="offset points",
                            ha="center", fontsize=8, color="#444")
                ax.annotate("", (xi, yi - 0.08), xytext=(xi, yi - 0.0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2))
    for xi, r in zip(x, rows):
        ax.plot([xi - 0.36, xi + 0.36], [r["exact"]] * 2, color="#222", lw=2, zorder=2,
                label="Exact equilibrium" if xi == 0 else None)
    ax.axhline(0, color="#bbb", lw=0.8, zorder=1)
    ax.set_xticks(x, [c[1] for c in CONDS], fontsize=9)
    ax.set_ylabel("Share of the system's errors withheld (net)")
    ax.set_ylim(CLIP - 0.25, 1.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#eee", lw=0.8)
    ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.17))
    fig.tight_layout()
    out = ROOT / "analysis"
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_trained_summary.{ext}", bbox_inches="tight")

    lines = ["| Condition | Frozen head | Adapter, gradient | Adapter, pay only | Exact |", "|---|---|---|---|---|"]
    for (key, _), r in zip(CONDS, rows):
        v = r["vals"]
        lines.append(f"| {key[0].upper() + key[1:]} | {v[0]:.2f} | {v[1]:.2f} | {v[2]:.2f} | {r['exact']:.2f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    for r in rows:
        print(r["cond"], "seeds", r["seeds"], "w_system", [round(w, 2) for w in r["wsys"]], "rmse", [round(w, 2) for w in r["rmse"]])


if __name__ == "__main__":
    main()
