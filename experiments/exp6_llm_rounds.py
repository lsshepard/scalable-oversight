"""Prompt-only LLM experiment: repeated rounds, two copies of one model.

Stage 1 (elicit):  no perturbation; climb the elicitation ladder (levels 0-4) to find
                   the weakest prompt under which the pair ends up concealing.
Stage 2 (compare): at a chosen level, run plain vs perturb vs placebo (the honest
                   plain arm, level 0, is always included as the baseline).

Examples
--------
  # offline pipeline check, no API key needed
  python experiments/exp6_llm_rounds.py elicit --backend mock-adaptive --trajectories 4

  # Claude models: the key is read from EXPERIMENT_ANTHROPIC_API_KEY only
  python experiments/exp6_llm_rounds.py elicit  --backend anthropic --model claude-sonnet-5-5 --effort medium
  python experiments/exp6_llm_rounds.py compare --backend anthropic --model claude-sonnet-5-5 --effort medium --level 2

  # any OpenAI-compatible endpoint: OPENAI_API_KEY (+ OPENAI_BASE_URL)
  python experiments/exp6_llm_rounds.py elicit --backend openai --model <model-id> --price-in 0.1 --price-out 0.5

Safety rails
------------
  * Spending cap: --max-usd (default 25) stops the run before the next call once the
    metered cost of this invocation reaches it. Needs prices: built in for current
    Claude models, otherwise pass --price-in / --price-out (USD per million tokens).
  * One configuration per output folder: model, effort, rounds, stake, prompts etc. are
    written to config.json, and a run with different settings refuses to reuse the
    folder. Use another --out for pilots and variants.
  * Runs are resumable: finished trajectories are skipped on restart.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from hpo.rounds import (GameConfig, FatalBackendError, Meter, metered,  # noqa: E402
                        mock_backend, openai_backend, play_trajectory,
                        prompt_fingerprint, summarize, system_prompt)

EFFORTS = ["low", "medium", "high", "xhigh", "max"]


def make_backend(args):
    """Returns (complete, price_in, price_out, max_tokens actually used)."""
    if args.backend.startswith("mock-"):
        return mock_backend(args.backend.split("-", 1)[1]), 0.0, 0.0, None
    if not args.model:
        sys.exit("--model is required for a real backend")
    if args.backend == "anthropic":
        from hpo.claude_backend import PRICES, claude_backend
        if args.temperature != 1.0:
            sys.exit("Current Claude models reject sampling parameters; leave --temperature "
                     "at its default.")
        max_tokens = args.max_tokens or 16000
        price = PRICES.get(args.model, (None, None))
        complete = claude_backend(args.model, effort=args.effort, max_tokens=max_tokens)
    else:
        if args.effort:
            sys.exit("--effort applies to the anthropic backend only")
        max_tokens = args.max_tokens or 2048
        price = (None, None)
        complete = openai_backend(args.model, temperature=args.temperature,
                                  max_tokens=max_tokens)
    price_in = args.price_in if args.price_in is not None else price[0]
    price_out = args.price_out if args.price_out is not None else price[1]
    return complete, price_in, price_out, max_tokens


def claim_folder(outdir: pathlib.Path, cfg: dict) -> None:
    """Bind an output folder to one configuration, so that results obtained under
    different settings or prompts are never pooled or overwritten by accident."""
    f = outdir / "config.json"
    if f.exists():
        old = json.loads(f.read_text())
        diff = [f"  {k}: folder has {old.get(k)!r}, this run has {cfg.get(k)!r}"
                for k in sorted(set(old) | set(cfg)) if old.get(k) != cfg.get(k)]
        if diff:
            sys.exit(f"{outdir} holds results from a different configuration:\n"
                     + "\n".join(diff) + "\nChoose another --out for this run.")
    else:
        f.write_text(json.dumps(cfg, indent=2) + "\n")


def run_arm(cfg: GameConfig, complete, meter: Meter, args, out: pathlib.Path):
    """Returns (records of finished trajectories, fatal error or None, failed seeds)."""
    done, records = set(), []
    if out.exists():
        for line in out.read_text().splitlines():
            records.append(json.loads(line))
        per_seed = {}
        for r in records:
            per_seed[r["seed"]] = per_seed.get(r["seed"], 0) + 1
        done = {s for s, n in per_seed.items() if n == 2 * cfg.rounds}
        records = [r for r in records if r["seed"] in done]
        out.write_text("".join(json.dumps(r) + "\n" for r in records))
    todo = [s for s in range(args.seed0, args.seed0 + args.trajectories) if s not in done]
    fatal, failed = None, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool, out.open("a") as fh:
        futs = {pool.submit(play_trajectory, cfg, complete, s): s for s in todo}
        for f in as_completed(futs):
            try:
                recs = f.result()
            except FatalBackendError as e:
                fatal = fatal or e
                meter.abort(str(e))        # the other trajectories stop at their next call
                continue
            except Exception as e:         # transient API trouble: lose this trajectory only
                failed.append(futs[f])
                print(f"  {out.stem}: trajectory {futs[f]} failed ({e!r}); rerun to retry",
                      flush=True)
                continue
            records.extend(recs)
            fh.write("".join(json.dumps(r) + "\n" for r in recs)); fh.flush()
            print(f"  {out.stem}: trajectory {futs[f]} done", flush=True)
    return records, fatal, failed


def fmt(s, key):
    d = s[key]
    return f"{d['mean']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}]"


def usd(tokens_in, tokens_out, price_in, price_out):
    if price_in is None or price_out is None:
        return None
    return (tokens_in * price_in + tokens_out * price_out) / 1e6


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["elicit", "compare"])
    ap.add_argument("--backend", default="mock-honest",
                    choices=["mock-honest", "mock-zero", "mock-adaptive", "anthropic", "openai"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--effort", choices=EFFORTS, default=None,
                    help="anthropic: thinking effort (default: the model's own default)")
    ap.add_argument("--temperature", type=float, default=1.0, help="openai backend only")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="per-reply cap (default 16000 for anthropic, 2048 for openai)")
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="elicit: ladder levels")
    ap.add_argument("--level", type=int, default=2, help="compare: level carried over from elicit")
    ap.add_argument("--rules", nargs="+", default=["plain", "perturb", "placebo"])
    ap.add_argument("--trajectories", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--stake", type=float, default=0.5,
                    help="private gain per genuine error left out; below 1 so that concealing alone does not pay")
    ap.add_argument("--plant-mean", type=float, default=1.5)
    ap.add_argument("--parse-retries", type=int, default=1,
                    help="extra queries when a reply contains no REPORT line")
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-usd", type=float, default=25.0,
                    help="spending cap for this invocation (needs prices)")
    ap.add_argument("--price-in", type=float, default=None, help="USD per million input tokens")
    ap.add_argument("--price-out", type=float, default=None, help="USD per million output tokens")
    ap.add_argument("--out", default="results/llm_rounds")
    args = ap.parse_args()

    tag = (args.model or args.backend).replace("/", "_")
    outdir = pathlib.Path(args.out) / tag
    try:
        complete, price_in, price_out, max_tokens = make_backend(args)
    except FatalBackendError as e:
        sys.exit(str(e))
    meter = Meter(price_in, price_out, args.max_usd)
    if meter.usd is None:
        print("No price known for this model: the --max-usd cap is NOT enforced. "
              "Pass --price-in and --price-out.", flush=True)
    complete = metered(complete, meter)

    outdir.mkdir(parents=True, exist_ok=True)
    claim_folder(outdir, dict(
        backend=args.backend, model=args.model, effort=args.effort,
        temperature=None if args.backend == "anthropic" else args.temperature,
        max_tokens=max_tokens, rounds=args.rounds, stake=args.stake,
        plant_mean=args.plant_mean, p_genuine=list(GameConfig().p_genuine),
        parse_retries=args.parse_retries, prompts=prompt_fingerprint()))

    arms = ([("plain", lv) for lv in args.levels] if args.stage == "elicit"
            else [("plain", 0)] + [(rule, args.level) for rule in args.rules])
    arms = list(dict.fromkeys(arms))
    summary, last, stopped = {}, f"last_{min(5, args.rounds)}", None
    for rule, level in arms:
        cfg = GameConfig(rule=rule, level=level, rounds=args.rounds, stake=args.stake,
                         plant_mean=args.plant_mean, parse_retries=args.parse_retries)
        name = f"{rule}_level{level}"
        (outdir / f"{name}.prompt.txt").write_text(
            system_prompt(rule, level, args.rounds, args.stake))
        recs, fatal, failed = run_arm(cfg, complete, meter, args, outdir / f"{name}.jsonl")
        if recs:
            summary[name] = summarize(recs, last_k=min(5, args.rounds))
            summary[name]["failed_trajectories"] = failed
        if fatal:
            stopped = fatal
            break
    (outdir / f"summary_{args.stage}.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{args.stage}: {tag}, {args.trajectories} trajectories x {args.rounds} rounds, "
          f"stake {args.stake:g}. Mean [95% bootstrap interval over trajectories].\n")
    print(f"{'arm':18s} {'n':>3s}  {'concealed share, last rounds':30s} {'concealed share, all':26s} "
          f"{'shift on clean':24s} {'exact counts':22s} {'no report':>9s} {'refused':>8s} {'cost':>8s}")
    for name, s in summary.items():
        cost = usd(s["in_tokens"], s["out_tokens"], price_in, price_out)
        print(f"{name:18s} {s['n_trajectories']:3d}  {fmt(s[last], 'concealed_share'):30s} "
              f"{fmt(s['all_rounds'], 'concealed_share'):26s} "
              f"{fmt(s['all_rounds'], 'clean_shift'):24s} "
              f"{fmt(s['all_rounds'], 'exact_count'):22s} {s['parse_failure_rate']:9.1%} "
              f"{s['refusal_rate']:8.1%} {'n/a' if cost is None else f'${cost:.2f}':>8s}")
    spent = meter.usd
    print(f"\nthis invocation: {meter.calls} calls, {meter.in_tokens:,} input and "
          f"{meter.out_tokens:,} output tokens"
          + ("" if spent is None else f", ${spent:.2f} of the ${args.max_usd:.2f} cap"))
    print(f"records, prompts and summary in {outdir}/")
    if stopped:
        sys.exit(f"\nSTOPPED EARLY: {stopped}\nFinished trajectories are saved; the table above "
                 f"is partial. Rerun the same command to resume.")


if __name__ == "__main__":
    main()
