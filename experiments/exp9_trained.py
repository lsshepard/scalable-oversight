"""Supervisors trained on payments (hpo/trained.py).

    # 1. features: a language model reads every view once
    python experiments/exp9_trained.py extract --model Qwen/Qwen2.5-0.5B-Instruct \
        --conds hidden plain rec100 --n-train 8000 --n-test 2000 --out results/trained/qwen05
    # 2. train pairs of report heads on payments, compare with the exact equilibrium
    python experiments/exp9_trained.py train --feats results/trained/qwen05 --plan main
    # check of the training code with perfect perception (no language model, CPU, seconds)
    python experiments/exp9_trained.py synth

Conditions (what the views look like):
    hidden    each copy gets its own Poisson(1.5) inserted errors, indistinguishable
    plain     no insertion; both supervisors see the same ledger
    recNNN    as hidden, but NNN% of inserted errors carry a visible tell (doubled space)
    marker    as hidden, and the ledger number is odd exactly when the original is dirty
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from hpo.ledger import LedgerConfig  # noqa: E402
from hpo.mechanism import Reward  # noqa: E402

P_GENUINE = (0.6, 0.3, 0.1)


def ledger_cfg(cond: str) -> LedgerConfig:
    if cond in ("hidden", "plain"):
        return LedgerConfig(p_genuine=P_GENUINE)
    if cond.startswith("rec"):
        return LedgerConfig(p_genuine=P_GENUINE, recognizability=int(cond[3:]) / 100)
    if cond == "marker":
        return LedgerConfig(p_genuine=P_GENUINE, marker=True)
    raise ValueError(cond)


def cmd_extract(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from hpo.trained import lm_features, sample_views
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).eval()
    nl = model.config.num_hidden_layers
    layers = args.layers or [nl // 3, 2 * nl // 3, nl]
    meta = dict(model=args.model, adapter=args.adapter, layers=layers, hidden=model.config.hidden_size,
                p_genuine=P_GENUINE, seed=args.seed)
    for cond in args.conds:
        for split, n, seed in (("train", args.n_train, args.seed), ("test", args.n_test, args.seed + 1)):
            f = out / f"{cond}_{split}.npz"
            if f.exists():
                print(f"{f} exists, skipped"); continue
            t0 = time.time()
            data = sample_views(ledger_cfg(cond), n, seed, plain=cond == "plain")
            V = data["D"].shape[1]
            texts = [t for row in data["texts"] for t in row]
            X, H = lm_features(texts, tok, model, layers, args.batch, device, progress=True)
            X = X.reshape(n, V, *X.shape[1:]); H = H.reshape(n, V, H.shape[-1])
            np.savez(f, X=X, H=H, texts=np.array(texts[:6]), **{k: data[k] for k in ("T", "D", "K", "R", "M", "status", "tell")})
            print(f"{f}: {n} originals x {V} views, X {X.shape}, {time.time() - t0:.0f}s", flush=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


def load(feats: pathlib.Path, cond: str, split: str):
    z = np.load(feats / f"{cond}_{split}.npz")
    data = {k: z[k] for k in ("T", "D", "K", "R", "M", "status", "tell")}
    return data, (z["X"], z["H"])


def reward(kind: str, b: float, eps: float) -> Reward:
    """mechanism = adjusted agreement + planted floor; the other two drop one term."""
    beta, lam = dict(mechanism=(1.0, 1.0), agree_only=(1.0, 0.0), floor_only=(0.0, 1.0))[kind]
    return Reward(beta=beta, lam=lam, b=b, eps=eps, name=kind)


# A small reporting charge selects one equilibrium among those that differ only by harmless
# over-reporting (docs/theory.tex, corollary on reward maximizers).
EPS = 0.02
# (label, condition of the views, reward terms, stake b, reporting charge eps)
PLANS = {
    "main": [("plain agreement", "plain", "mechanism", 0.5, EPS),
             ("agreement only", "hidden", "agree_only", 0.5, EPS),
             ("floor only", "hidden", "floor_only", 0.5, EPS),
             ("both terms", "hidden", "mechanism", 0.5, EPS),
             ("both terms, insertions recognizable", "rec100", "mechanism", 0.5, EPS)],
    "nostake": [("plain agreement, no stake", "plain", "mechanism", 0.0, EPS),
                ("both terms, no stake", "hidden", "mechanism", 0.0, EPS),
                ("both terms, no stake, insertions recognizable", "rec100", "mechanism", 0.0, EPS)],
    "recog": [(f"both terms, {p}% recognizable", f"rec{p:03d}" if p else "hidden", "mechanism", 0.5, EPS)
              for p in (0, 50, 90, 100)],
    "stake": [(f"both terms, stake {b:g}", "hidden", "mechanism", b, EPS) for b in (0.1, 0.25, 0.5, 1.0)],
    "marker": [("both terms, surviving signal", "marker", "mechanism", 0.5, EPS)],
    # perception ceiling: the same head trained on the true count of wrong lines (uses labels)
    "ceiling": [(f"trained on true counts ({c})", c, "supervised", 0.0, 0.0) for c in ("hidden", "rec100", "plain")],
}


def shading_distance(learned: dict, exact: dict) -> float:
    """Root mean square gap between learned and exact shading, weighted by how often each
    view summary (K, R, M) occurs in the test set."""
    num = den = 0.0
    for key, (val, cnt) in learned.items():
        if key in exact:
            num += cnt * (val - exact[key]) ** 2; den += cnt
    return float(np.sqrt(num / den)) if den else float("nan")


def run_plan(plan, get_data, tc_kwargs, seeds, device, log=print):
    from hpo.trained import TrainConfig, exact_prediction, train_pair
    rows = []
    for label, cond, kind, b, eps in plan:
        sup = kind == "supervised"
        rw = Reward(name="supervised") if sup else reward(kind, b, eps)
        ex = dict(bounded=False) if sup else exact_prediction(ledger_cfg(cond), rw, plain=cond == "plain")
        (train, ftrain), (test, ftest) = get_data(cond, "train"), get_data(cond, "test")
        for est, seed in [(e, s) for e in tc_kwargs.get("estimators", ["analytic"]) for s in seeds]:
            if sup and est != tc_kwargs.get("estimators", ["analytic"])[0]:
                continue
            tc = TrainConfig(seed=seed, estimator=est, supervised=sup,
                             **{k: v for k, v in tc_kwargs.items() if k != "estimators"})
            t0 = time.time()
            log(f"== {label} | views: {cond} | {kind} b={b:g} eps={eps:g} | {est} seed {seed}")
            res = train_pair(train, test, ftrain, ftest, rw, tc, device, log=log)
            fin = res["final"]
            dist = shading_distance(fin["shading"], ex["shading"]) if ex.get("bounded") else float("nan")
            row = dict(label=label, cond=cond, kind=kind, b=b, eps=eps, estimator=est, seed=seed,
                       seconds=time.time() - t0, exact=ex, final={k: v for k, v in fin.items() if k != "shading"},
                       shading=fin["shading"], shading_rms_gap=dist,
                       trace=[{k: v for k, v in t.items() if k != "shading"} for t in res["trace"]])
            rows.append(row)
            if ex.get("bounded"):
                log(f"   exact: concealed/dirty {ex['concealed_dirty']:+.3f}  share {ex['concealed_share']:.3f}  "
                    f"clean {ex['under_clean']:+.3f}  charge {ex['charge']:.3f}   | rms gap of shading {dist:.3f}")
            elif not sup:
                log("   exact: unbounded (reports fall without limit)")
    return rows


def cmd_synth(args):
    """The training code against the exact equilibrium, with perfect perception."""
    from hpo.trained import sample_views, synthetic_features
    cache = {}
    def get(cond, split):
        if (cond, split) not in cache:
            n, seed = (args.n_train, args.seed) if split == "train" else (args.n_test, args.seed + 1)
            d = sample_views(ledger_cfg(cond), n, seed, plain=cond == "plain")
            cache[(cond, split)] = (d, synthetic_features(d, seed=seed))
        return cache[(cond, split)]
    plan = [r for name in args.plans for r in PLANS[name]]
    rows = run_plan(plan, get, dict(steps=args.steps, estimators=args.estimators, lr=args.lr, batch=args.batch,
                                    sigma=args.sigma), range(args.seeds), "cpu")
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(rows))


def cmd_train(args):
    import torch
    feats = pathlib.Path(args.feats)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache = {}
    def get(cond, split):
        if (cond, split) not in cache:
            cache[(cond, split)] = load(feats, cond, split)
        return cache[(cond, split)]
    plan = [r for name in args.plans for r in PLANS[name]]
    rows = run_plan(plan, get, dict(steps=args.steps, estimators=args.estimators, lr=args.lr, batch=args.batch,
                                    sigma=args.sigma, weight_decay=args.weight_decay), range(args.seeds), device)
    out = pathlib.Path(args.out or feats / f"train_{'_'.join(args.plans)}.json")
    out.write_text(json.dumps(rows))
    print(f"written {out}")


def cmd_e2e(args):
    """Train the language model itself (adapter + head) on the payments."""
    import torch
    from hpo.trained import exact_prediction, sample_views
    from hpo.trained_e2e import E2EConfig, train_e2e
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(out.read_text()) if out.exists() else []
    done = {(r["label"], r["seed"]) for r in rows}
    plan = [r for name in args.plans for r in PLANS[name] if not args.only or any(o == r[0] for o in args.only)]
    for label, cond, kind, b, eps in plan:
        sup = kind == "supervised"
        rw = Reward(name="supervised") if sup else reward(kind, b, eps)
        ex = dict(bounded=False) if sup else exact_prediction(ledger_cfg(cond), rw, plain=cond == "plain")
        train = sample_views(ledger_cfg(cond), args.n_train, args.seed, plain=cond == "plain")
        test = sample_views(ledger_cfg(cond), args.n_test, args.seed + 1, plain=cond == "plain")
        for seed in range(args.seeds):
            if (label, seed) in done:
                continue
            print(f"== {label} | views: {cond} | {kind} b={b:g} eps={eps:g} | end to end, seed {seed}", flush=True)
            t0 = time.time()
            cfg = E2EConfig(steps=args.steps, batch=args.batch, seed=seed, rank=args.rank, supervised=sup,
                            lr_adapter=args.lr_adapter, eval_every=args.eval_every, n_eval=args.n_test)
            res = train_e2e(args.model, train, test, rw, cfg, device, log=lambda m: print(m, flush=True))
            fin = res["final"]
            rows.append(dict(label=label, cond=cond, kind=kind, b=b, eps=eps, estimator="end to end", seed=seed,
                             model=args.model, seconds=time.time() - t0, exact=ex,
                             final={k: v for k, v in fin.items() if k != "shading"}, shading=fin["shading"],
                             shading_rms_gap=shading_distance(fin["shading"], ex["shading"]) if ex.get("bounded") else float("nan"),
                             trace=[{k: v for k, v in t.items() if k != "shading"} for t in res["trace"]]))
            out.write_text(json.dumps(rows))
            if ex.get("bounded"):
                print(f"   exact: concealed/dirty {ex['concealed_dirty']:+.3f}  share {ex['concealed_share']:.3f}  "
                      f"clean {ex['under_clean']:+.3f}  charge {ex['charge']:.3f}", flush=True)
    print(f"written {out}")


def _train_args(p):
    p.add_argument("--plans", nargs="+", default=["main"], choices=list(PLANS))
    p.add_argument("--steps", type=int, default=4000); p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--estimators", nargs="+", default=["analytic"], choices=["analytic", "reinforce"])
    p.add_argument("--lr", type=float, default=2e-3); p.add_argument("--batch", type=int, default=512)
    p.add_argument("--sigma", type=float, default=0.3); p.add_argument("--out", default="")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sy = sub.add_parser("synth"); _train_args(sy)
    sy.add_argument("--n-train", type=int, default=20000); sy.add_argument("--n-test", type=int, default=5000)
    sy.add_argument("--seed", type=int, default=0)
    tr = sub.add_parser("train"); _train_args(tr)
    tr.add_argument("--feats", required=True); tr.add_argument("--device", default="")
    tr.add_argument("--weight-decay", type=float, default=0.0)
    e = sub.add_parser("extract")
    e.add_argument("--model", required=True); e.add_argument("--adapter", default="")
    e.add_argument("--conds", nargs="+", default=["hidden"])
    e.add_argument("--n-train", type=int, default=8000); e.add_argument("--n-test", type=int, default=2000)
    e.add_argument("--layers", type=int, nargs="*"); e.add_argument("--batch", type=int, default=8)
    e.add_argument("--device", default=""); e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", required=True)
    ee = sub.add_parser("e2e")
    ee.add_argument("--model", required=True); ee.add_argument("--plans", nargs="+", default=["main"], choices=list(PLANS))
    ee.add_argument("--only", nargs="*", default=[], help="labels of the plan rows to run")
    ee.add_argument("--n-train", type=int, default=4000); ee.add_argument("--n-test", type=int, default=400)
    ee.add_argument("--steps", type=int, default=1500); ee.add_argument("--batch", type=int, default=16)
    ee.add_argument("--seeds", type=int, default=1); ee.add_argument("--seed", type=int, default=0)
    ee.add_argument("--rank", type=int, default=8); ee.add_argument("--lr-adapter", type=float, default=2e-4)
    ee.add_argument("--eval-every", type=int, default=150)
    ee.add_argument("--device", default=""); ee.add_argument("--out", required=True)
    for q in (e, sy, tr, ee):
        q.add_argument("--threads", type=int, default=0, help="CPU threads for torch (0 = default)")
    args = ap.parse_args()
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    dict(extract=cmd_extract, synth=cmd_synth, train=cmd_train, e2e=cmd_e2e)[args.cmd](args)
