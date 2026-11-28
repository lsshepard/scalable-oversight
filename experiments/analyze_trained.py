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
                 gap=float(np.mean([x for x in (r["shading_rms_gap"] for r in rs) if x == x] or [float("nan")])))
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
    """Left: no stake (a reporting charge of 0.02 is the only reason to report less).
    Right: stake 0.5.  Shares above 1 (reports below zero) are drawn at 1; the table has them."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    panels = [("No private gain from concealment",
               ["plain agreement, no stake", "both terms, no stake", "both terms, no stake, insertions recognizable"],
               ["Plain\nagreement", "Both terms\n(the mechanism)", "Both terms,\ninsertions\nrecognizable"]),
              ("Private gain of 0.5 per concealed error",
               ["plain agreement", "agreement only", "floor only", "both terms", "both terms, insertions recognizable"],
               ["Plain\nagreement", "Agreement\nonly", "Floor\nonly", "Both terms\n(the mechanism)",
                "Both terms,\ninsertions\nrecognizable"])]
    gs = {g["label"]: g for g in groups if g["estimator"] == estimator}
    panels = [(t, [(l, s) for l, s in zip(ls, ss) if l in gs]) for t, ls, ss in panels]
    panels = [(t, pr) for t, pr in panels if pr]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(sum(1.45 * len(pr) + 0.9 for _, pr in panels), 3.9), squeeze=False,
                             gridspec_kw=dict(width_ratios=[len(pr) + 0.6 for _, pr in panels]))
    clip = lambda v: float(np.clip(v, 0.0, 1.0))
    for ax, (title, present) in zip(axes[0], panels):
        _style(ax)
        for i, (label, _) in enumerate(present):
            g = gs[label]; d = g["net_share"]
            unbounded = not g["exact"].get("bounded")
            ax.bar(i, clip(d["mean"]), width=0.36, color=BLUE, linewidth=0, zorder=2)
            if d["hi"] - d["lo"] > 5e-4 and not unbounded:
                ax.plot([i, i], [clip(d["lo"]), clip(d["hi"])], color=INK, linewidth=1.0, zorder=3)
            txt = "no limit" if unbounded else (f"{d['mean']:.2f}" if d["mean"] <= 1 else "1.00")
            ax.text(i, clip(max(d["hi"], d["mean"])) + 0.035, txt, ha="center", va="bottom", fontsize=9, color=INK)
            if not unbounded:
                ax.plot([i - 0.27, i + 0.27], [clip(g["exact"]["net_share"])] * 2, color=ORANGE, linewidth=2.2, zorder=4,
                        solid_capstyle="round")
        ax.set_xticks(range(len(present))); ax.set_xticklabels([s for _, s in present], fontsize=8, color=INK2)
        ax.set_ylim(0, 1.14); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xlim(-0.6, len(present) - 0.4)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=8)
    axes[0][0].set_ylabel("Share of the system's errors\nwithheld from the report", fontsize=9, color=INK2)
    fig.legend(handles=[Line2D([0], [0], color=BLUE, linewidth=6, label="trained pair"),
                        Line2D([0], [0], color=ORANGE, linewidth=2.2, label="exact equilibrium, perfect perception")],
               loc="upper right", bbox_to_anchor=(0.995, 1.0), ncol=2, frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); _save(fig, path)


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
    """What one line of each kind adds to the trained report (least-squares weights)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    panels = [("No private gain from concealment",
               ["plain agreement, no stake", "both terms, no stake", "both terms, no stake, insertions recognizable"]),
              ("Private gain of 0.5 per concealed error",
               ["plain agreement", "both terms", "both terms, insertions recognizable"])]
    short = ["Plain\nagreement", "Both terms\n(the mechanism)", "Both terms,\ninsertions\nrecognizable"]
    gs = {g["label"]: g for g in groups if g["estimator"] == estimator}
    panels = [(t, [(l, s) for l, s in zip(ls, short) if l in gs]) for t, ls in panels]
    panels = [(t, pr) for t, pr in panels if pr]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.9), squeeze=False, sharey=True)
    for ax, (title, present) in zip(axes[0], panels):
        _style(ax)
        for i, (label, _) in enumerate(present):
            g = gs[label]
            tell = g["cond"].startswith("rec") and g["w_inserted_tell"]["mean"] == g["w_inserted_tell"]["mean"]
            ins = g["w_inserted_tell"] if tell else g["w_inserted_plain"]
            for dx, d, color in ((-0.17, g["w_system"], BLUE), (0.17, ins, ORANGE)):
                if d["mean"] != d["mean"]:
                    continue
                ax.bar(i + dx, d["mean"], width=0.3, color=color, linewidth=0, zorder=2)
                ax.text(i + dx, max(d["mean"], 0) + 0.03, f"{d['mean']:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.set_xticks(range(len(present))); ax.set_xticklabels([s for _, s in present], fontsize=8, color=INK2)
        ax.set_ylim(-0.35, 1.22); ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=8)
    axes[0][0].set_ylabel("Added to the report by one such line", fontsize=9, color=INK2)
    fig.legend(handles=[Line2D([0], [0], color=BLUE, linewidth=6, label="an error of the audited system"),
                        Line2D([0], [0], color=ORANGE, linewidth=6, label="an inserted error")],
               loc="upper right", bbox_to_anchor=(0.995, 1.0), ncol=2, frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); _save(fig, path)


def fig_tell(folder, path):
    """Recognizable insertions, private gain 0.5: share withheld against the share of tells the
    supervisor's features expose, with the exact curve and hidden insertions at the same sizes."""
    import sys
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from hpo.ledger import LedgerConfig
    from hpo.mechanism import Reward
    from hpo.trained import exact_prediction
    folder = pathlib.Path(folder)
    if not (folder / "tell_sweep.json").exists():
        return
    tell = [r for r in json.loads((folder / "tell_sweep.json").read_text()) if r["label"] == "private gain 0.5"]
    ctrl = [r for r in json.loads((folder / "capacity_sweep_hidden.json").read_text()) if r["label"] == "private gain 0.5"] \
        if (folder / "capacity_sweep_hidden.json").exists() else []
    ks = sorted({r["k"] for r in tell})
    x = [100 * np.mean([r["tell_recall"] for r in tell if r["k"] == k]) for k in ks]
    ps = np.concatenate([np.linspace(0.70, 0.95, 11), np.linspace(0.96, 1.0, 9)])
    ex = [exact_prediction(LedgerConfig(p_genuine=(0.6, 0.3, 0.1), recognizability=float(p)), Reward(beta=1, lam=1, b=0.5, eps=0.02))
          for p in ps]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    ax = axes[0]; _style(ax)
    ax.plot(100 * ps, [e["net_share"] for e in ex], color=ORANGE, linewidth=2, zorder=2, label="exact equilibrium")
    for k, xi in zip(ks, x):
        ys = [r["net_share"] for r in tell if r["k"] == k]
        ax.plot([xi] * len(ys), ys, linestyle="none", marker="o", markersize=6.5, color=BLUE, markeredgecolor=SURFACE,
                markeredgewidth=1.2, zorder=4, label="trained pair, insertions carry a tell" if k == ks[0] else None)
        yc = [r["net_share"] for r in ctrl if r["k"] == k]
        if yc:
            ax.plot([xi] * len(yc), yc, linestyle="none", marker="o", markersize=6, markerfacecolor=SURFACE, markeredgecolor=AQUA,
                    markeredgewidth=1.6, zorder=3, label="same supervisor, hidden insertions" if k == ks[0] else None)
    ax.set_xlabel("Share of tells the supervisor's features expose (%)", fontsize=9, color=INK2)
    ax.set_ylabel("Share of the system's errors\nwithheld from the report", fontsize=9, color=INK2)
    ax.set_xlim(72, 101.5); ax.set_ylim(0, 1.3)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    ax = axes[1]; _style(ax)
    for j, (key, color, label) in enumerate((("w_system", BLUE, "an error of the audited system"),
                                             ("w_inserted_tell", ORANGE, "an inserted error (with tell)"))):
        ys = [np.mean([r[key] for r in tell if r["k"] == k]) for k in ks]
        ax.plot(x, ys, color=color, linewidth=2, marker="o", markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.2,
                zorder=3, label=label)
    ax.set_xlabel("Share of tells the supervisor's features expose (%)", fontsize=9, color=INK2)
    ax.set_ylabel("Added to the report by one such line", fontsize=9, color=INK2)
    ax.set_xlim(72, 101.5); ax.set_ylim(0, 1.1)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left")
    fig.tight_layout(); _save(fig, path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--estimator", default="analytic")
    ap.add_argument("--sweeps", default="", help="folder with tell_sweep.json (adds fig_trained_tell)")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    groups = load(a.files)
    print(write_table(groups, out))
    fig_main(groups, out / "fig_trained_main.png", a.estimator)
    fig_dials(groups, out / "fig_trained_dials.png", a.estimator)
    fig_weights(groups, out / "fig_trained_weights.png", a.estimator)
    if a.sweeps:
        fig_tell(a.sweeps, out / "fig_trained_tell.png")
    print(f"\nwritten to {out}/")
