"""Prompt-only LLM experiment: repeated rounds, two copies of one model.

Stage 1 (elicit):  no perturbation; climb the elicitation ladder (levels 0-4) to find
                   the weakest prompt under which the pair ends up concealing.
Stage 2 (compare): at a chosen level, run plain vs perturb vs placebo.

Examples
--------
  # offline pipeline check, no API key needed
  python experiments/exp6_llm_rounds.py elicit --backend mock-adaptive --trajectories 4

  # real runs
  export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY (+ OPENAI_BASE_URL)
  python experiments/exp6_llm_rounds.py elicit  --backend anthropic --model <model-id>
  python experiments/exp6_llm_rounds.py compare --backend anthropic --model <model-id> --level 2

Runs are resumable: finished trajectories are skipped on restart.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from hpo.rounds import (GameConfig, play_trajectory, summarize, mock_backend,  # noqa: E402
                        anthropic_backend, openai_backend)


def backend(args):
    if args.backend.startswith("mock-"):
        return mock_backend(args.backend.split("-", 1)[1])
    if not args.model:
        sys.exit("--model is required for a real backend")
    make = anthropic_backend if args.backend == "anthropic" else openai_backend
    return make(args.model, temperature=args.temperature, max_tokens=args.max_tokens)


def run_arm(cfg: GameConfig, complete, args, out: pathlib.Path):
    done, records = set(), []
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line); records.append(r)
        per_seed = {}
        for r in records:
            per_seed[r["seed"]] = per_seed.get(r["seed"], 0) + 1
        done = {s for s, n in per_seed.items() if n == 2 * cfg.rounds}
        records = [r for r in records if r["seed"] in done]
        out.write_text("".join(json.dumps(r) + "\n" for r in records))
    todo = [s for s in range(args.seed0, args.seed0 + args.trajectories) if s not in done]
    with ThreadPoolExecutor(max_workers=args.workers) as pool, out.open("a") as fh:
        futs = {pool.submit(play_trajectory, cfg, complete, s): s for s in todo}
        for f in as_completed(futs):
            recs = f.result(); records.extend(recs)
            fh.write("".join(json.dumps(r) + "\n" for r in recs)); fh.flush()
            print(f"  {out.stem}: trajectory {futs[f]} done", flush=True)
    return records


def fmt(s, key):
    d = s[key]
    return f"{d['mean']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["elicit", "compare"])
    ap.add_argument("--backend", default="mock-honest",
                    choices=["mock-honest", "mock-zero", "mock-adaptive", "anthropic", "openai"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="elicit: ladder levels")
    ap.add_argument("--level", type=int, default=2, help="compare: level carried over from elicit")
    ap.add_argument("--rules", nargs="+", default=["plain", "perturb", "placebo"])
    ap.add_argument("--trajectories", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--stake", type=float, default=0.5,
                    help="private gain per genuine error left out; below 1 so that concealing alone does not pay")
    ap.add_argument("--plant-mean", type=float, default=1.5)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results/llm_rounds")
    args = ap.parse_args()

    tag = (args.model or args.backend).replace("/", "_")
    outdir = pathlib.Path(args.out) / tag
    outdir.mkdir(parents=True, exist_ok=True)
    complete = backend(args)
    arms = ([("plain", lv) for lv in args.levels] if args.stage == "elicit"
            else [("plain", 0)] + [(rule, args.level) for rule in args.rules])
    arms = list(dict.fromkeys(arms))
    summary, last = {}, f"last_{min(5, args.rounds)}"
    for rule, level in arms:
        cfg = GameConfig(rule=rule, level=level, rounds=args.rounds, stake=args.stake,
                         plant_mean=args.plant_mean)
        recs = run_arm(cfg, complete, args, outdir / f"{rule}_level{level}.jsonl")
        summary[f"{rule}_level{level}"] = summarize(recs, last_k=min(5, args.rounds))
    (outdir / f"summary_{args.stage}.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{args.stage}: {tag}, {args.trajectories} trajectories x {args.rounds} rounds, "
          f"stake {args.stake:g}. Mean [95% bootstrap interval over trajectories].\n")
    print(f"{'arm':18s} {'concealed share, last rounds':30s} {'concealed share, all':26s} "
          f"{'shift on clean':24s} {'count error':22s} parse fail")
    for name, s in summary.items():
        print(f"{name:18s} {fmt(s[last], 'concealed_share'):30s} "
              f"{fmt(s['all_rounds'], 'concealed_share'):26s} "
              f"{fmt(s['all_rounds'], 'clean_shift'):24s} "
              f"{fmt(s['all_rounds'], 'count_error'):22s} {s['parse_failure_rate']:.1%}")
    print(f"\nrecords and summary in {outdir}/")


if __name__ == "__main__":
    main()
