"""Analysis of the trained-supervisor runs (experiments/exp9_trained.py).

    python experiments/analyze_trained.py results/trained/qwen15/train_*.json --out results/trained/qwen15/analysis

Writes table.md / table.csv (trained pairs, mean and range over seeds, next to the exact
equilibrium for a supervisor with perfect perception) and three figures:
  fig_trained_main      concealed share by condition, trained against exact
  fig_trained_dials     recognizability and stake sweeps, trained against exact
  fig_trained_weights   what one line of each kind adds to the trained report
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
from collections import defaultdict

import numpy as np

BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"
INK, INK2, GRID, AXIS, SURFACE = "#191919", "#666666", "#e7e7e7", "#c2c2c2", "#ffffff"
METRICS = ("net_share", "concealed_share", "concealed_dirty", "under_clean", "charge", "w_system", "w_inserted_plain",
           "w_inserted_tell", "rmse_count")


def load(paths):
    rows = []
    for p in paths:
        rows += json.loads(pathlib.Path(p).read_text())
    groups = defaultdict(list)
    for r in rows:
        groups[(r["label"], r["estimator"])].append(r)
    out = []
    for (label, est), rs in groups.items():
        g = dict(label=label, estimator=est, cond=rs[0]["cond"], kind=rs[0]["kind"], b=rs[0]["b"], eps=rs[0]["eps"],
                 seeds=len(rs), exact=rs[0]["exact"], traces=[r["trace"] for r in rs],
                 gap=float(np.nanmean([r["shading_rms_gap"] for r in rs])))
        for m in METRICS:
            v = np.array([r["final"][m] for r in rs], float)
            g[m] = dict(mean=float(np.nanmean(v)), lo=float(np.nanmin(v)), hi=float(np.nanmax(v)))
        out.append(g)
    return out


def fmt(d, digits=3):
    if d["mean"] != d["mean"]:
        return "n/a"
    return f"{d['mean']:.{digits}f}" + (f" [{d['lo']:.{digits}f}, {d['hi']:.{digits}f}]" if d["hi"] - d["lo"] > 5e-4 else "")


def write_table(groups, out):
    md = ["| Condition | Estimator | Seeds | Share of system errors withheld: trained | exact | Withheld per dirty original: trained | exact | "
          "Charge per round: trained | exact | Weight of a system error | of an inserted error | of an inserted error with tell | "
          "Count error (RMSE) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for g in groups:
        ex = g["exact"]
        e = (lambda k: f"{ex[k]:.3f}") if ex.get("bounded") else (lambda k: "unbounded" if g["kind"] != "supervised" else "")
        md.append(f"| {g['label']} | {g['estimator']} | {g['seeds']} | {fmt(g['net_share'])} | {e('net_share')} | "
                  f"{fmt(g['concealed_dirty'])} | {e('concealed_dirty')} | {fmt(g['charge'])} | {e('charge')} | "
                  f"{fmt(g['w_system'], 2)} | {fmt(g['w_inserted_plain'], 2)} | {fmt(g['w_inserted_tell'], 2)} | {fmt(g['rmse_count'], 2)} |")
        rows.append(dict(label=g["label"], estimator=g["estimator"], cond=g["cond"], kind=g["kind"], b=g["b"], eps=g["eps"],
                         seeds=g["seeds"], shading_rms_gap=g["gap"],
                         **{f"{m}_{s}": g[m][s] for m in METRICS for s in ("mean", "lo", "hi")},
                         **{f"exact_{k}": ex.get(k) for k in ("net_share", "concealed_share", "concealed_dirty", "under_clean", "charge")}))
    (out / "table.md").write_text("\n".join(md) + "\n")
    with (out / "table.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return "\n".join(md)


def _style(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS); ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", length=0, colors=INK2, labelsize=8.5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)


def _save(fig, path):
    fig.savefig(path, dpi=200, facecolor=SURFACE); fig.savefig(path.with_suffix(".pdf"), facecolor=SURFACE)


def fig_main(groups, path, estimator):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    order = ["plain agreement", "agreement only", "floor only", "both terms", "both terms, insertions recognizable"]
    short = ["Plain\nagreement", "Agreement\nonly", "Floor\nonly", "Both terms\n(the mechanism)", "Both terms,\ninsertions\nrecognizable"]
    gs = {g["label"]: g for g in groups if g["estimator"] == estimator}
    present = [(l, s) for l, s in zip(order, short) if l in gs]
    if not present:
        return
    fig, ax = plt.subplots(figsize=(1.55 * len(present) + 1.2, 3.6)); _style(ax)
    for i, (label, _) in enumerate(present):
        g = gs[label]; d = g["net_share"]
        clip = lambda v: float(np.clip(v, 0.0, 1.3))
        d = dict(mean=clip(d["mean"]), lo=clip(d["lo"]), hi=clip(d["hi"]), raw=d["mean"])
        ax.bar(i, d["mean"], width=0.36, color=BLUE, linewidth=0, zorder=2)
        if d["hi"] - d["lo"] > 5e-4:
            ax.plot([i, i], [d["lo"], d["hi"]], color=INK, linewidth=1.0, zorder=3)
        ax.text(i, max(d["hi"], d["mean"]) + 0.035, f"{d['raw']:.2f}" if d["raw"] <= 1.3 else "> 1.3\n(still falling)",
                ha="center", va="bottom", fontsize=9, color=INK)
        if g["exact"].get("bounded"):
            ax.plot([i - 0.27, i + 0.27], [g["exact"]["net_share"]] * 2, color=ORANGE, linewidth=2.2, zorder=4,
                    solid_capstyle="round")
        else:
            ax.text(i, 0.6, "exact:\nno bound", ha="center", va="center", fontsize=8, color=SURFACE, zorder=5)
    ax.set_xticks(range(len(present))); ax.set_xticklabels([s for _, s in present], fontsize=8, color=INK2)
    ax.set_ylim(0, 1.52); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0, 1.25]); ax.set_xlim(-0.6, len(present) - 0.4)
    ax.set_ylabel("System errors withheld,\nas a share of those present", fontsize=9, color=INK2)
    ax.legend(handles=[Line2D([0], [0], color=BLUE, linewidth=6, label="trained pair"),
                       Line2D([0], [0], color=ORANGE, linewidth=2.2, label="exact equilibrium")],
              loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2, frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.tight_layout(); _save(fig, path)


def fig_dials(groups, path, estimator):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gs = [g for g in groups if g["estimator"] == estimator and g["kind"] == "mechanism" and g["exact"].get("bounded")]
    rec = sorted([g for g in gs if g["b"] == 0.5 and (g["cond"] == "hidden" or g["cond"].startswith("rec"))
                  and "recognizable" in g["label"] and "%" in g["label"]], key=lambda g: g["label"])
    rec = sorted(rec, key=lambda g: 0 if g["cond"] == "hidden" else int(g["cond"][3:]))
    stk = sorted([g for g in gs if g["label"].startswith("both terms, stake")], key=lambda g: g["b"])
    panels = [(rec, [0 if g["cond"] == "hidden" else int(g["cond"][3:]) for g in rec], "Share of inserted errors that carry a tell (%)"),
              (stk, [g["b"] for g in stk], "Supervisor's private gain per concealed error")]
    panels = [p for p in panels if p[0]]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.4), squeeze=False)
    for ax, (gl, xs, xlabel) in zip(axes[0], panels):
        _style(ax)
        ax.plot(xs, [g["exact"]["net_share"] for g in gl], color=ORANGE, linewidth=2, zorder=2, label="exact equilibrium")
        ax.errorbar(xs, [g["net_share"]["mean"] for g in gl],
                    yerr=[[g["net_share"]["mean"] - g["net_share"]["lo"] for g in gl],
                          [g["net_share"]["hi"] - g["net_share"]["mean"] for g in gl]],
                    fmt="o", color=BLUE, markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.2, elinewidth=1,
                    ecolor=INK, zorder=3, label="trained pair")
        ax.set_xlabel(xlabel, fontsize=9, color=INK2); ax.set_ylim(-0.04, 1.3); ax.set_xticks(xs)
    axes[0][0].set_ylabel("System errors withheld,\nas a share of those present", fontsize=9, color=INK2)
    axes[0][0].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    fig.tight_layout(); _save(fig, path)


def fig_weights(groups, path, estimator):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    want = ["plain agreement, no stake", "both terms, no stake", "both terms, no stake, insertions recognizable",
            "plain agreement", "both terms", "both terms, insertions recognizable"]
    short = ["Plain\nagreement", "Both terms", "Both terms,\nrecognizable", "Plain\nagreement", "Both terms", "Both terms,\nrecognizable"]
    gs = {g["label"]: g for g in groups if g["estimator"] == estimator}
    present = [(l, s) for l, s in zip(want, short) if l in gs]
    if not present:
        return
    fig, ax = plt.subplots(figsize=(1.5 * len(present) + 1.4, 3.6)); _style(ax)
    for i, (label, _) in enumerate(present):
        g = gs[label]
        ins = g["w_inserted_tell"] if g["cond"].startswith("rec") and g["w_inserted_tell"]["mean"] == g["w_inserted_tell"]["mean"] \
            else g["w_inserted_plain"]
        for dx, d, color in ((-0.17, g["w_system"], BLUE), (0.17, ins, ORANGE)):
            if d["mean"] != d["mean"]:
                continue
            ax.bar(i + dx, d["mean"], width=0.3, color=color, linewidth=0, zorder=2)
            ax.text(i + dx, max(d["mean"], 0) + 0.03, f"{d['mean']:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_xticks(range(len(present))); ax.set_xticklabels([s for _, s in present], fontsize=8, color=INK2)
    ax.set_ylabel("Added to the report by one such line", fontsize=9, color=INK2); ax.set_ylim(-0.35, 1.25)
    ax.legend(handles=[Line2D([0], [0], color=BLUE, linewidth=6, label="an error of the audited system"),
                       Line2D([0], [0], color=ORANGE, linewidth=6, label="an inserted error")],
              loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.tight_layout(); _save(fig, path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--estimator", default="analytic")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    groups = load(a.files)
    print(write_table(groups, out))
    fig_main(groups, out / "fig_trained_main.png", a.estimator)
    fig_dials(groups, out / "fig_trained_dials.png", a.estimator)
    fig_weights(groups, out / "fig_trained_weights.png", a.estimator)
    print(f"\nwritten to {out}/")
