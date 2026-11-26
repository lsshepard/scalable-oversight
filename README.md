# hidden-perturbation-oversight

First empirical experiment for *Honest Reporting through Hidden Perturbations*
(scalable AI oversight): "synthetic ledgers".

A supervised model produces a log U with an unknown number T(U) of genuine errors.
For each of two supervisors the designer privately plants Delta_i extra errors,
producing a view Z_i. Supervisor i reports F_i; the adjusted report is
V_i = F_i - Delta_i, and the supervisor is charged

    cost_i = beta (V_i - V_j)^2 + lambda max(Delta_i - F_i, 0)^2 [+ eps F_i]
             [- b 1{T > 0} (T - V_i)]        (outside stake in concealment)

This repository tests six predictions of the theory on a fully synthetic task where
everything can be computed exactly. Results, tables and verdicts: **[REPORT.md](REPORT.md)**.

## Layout

| path | what |
|---|---|
| `hpo/ledger.py` | text generator: originals, planted views, tells (`recognizability`), marker mode; `parse_view` (what a supervisor can read) |
| `hpo/model.py` | exact finite model of the view summary; sigma^2, gamma, c_k, q0 |
| `hpo/mechanism.py` | the cost function (`Reward`, `cost`) |
| `hpo/strategies.py` | strategy menu and Monte Carlo cost / deviation-gain matrices |
| `hpo/equilibrium.py` | convex potential, L-BFGS equilibrium, gradient play (exact and sampled), exploitability, integer best-response iteration, hide-one threshold |
| `hpo/discriminator.py` | logistic / MLP "is the original dirty?" classifiers, Brier gain |
| `hpo/supervisor.py` | `Supervisor.report(view_text) -> float` interface, rule-based supervisor, LLM prompt templates (honest and collusive). **The LLM backend is not implemented**: no model API key or GPU was available |
| `experiments/exp1_menu.py` | strategy menu under 12 rewards (predictions 1, 2, 3) |
| `experiments/exp2_equilibrium.py` | exact / learned equilibrium, unboundedness demo, bound sweep (1, 3) |
| `experiments/exp3_recognizability.py` | recognizability sweep and discriminator (4) |
| `experiments/exp4_integer.py` | integer-report hide-one check (5) |
| `experiments/exp5_marker.py` | marker check (6) |
| `results/` | CSV / JSON / PNG outputs and `run_all.log` |
| `tests/` | pytest suite (generator, exact model, mechanism, solvers, interface) |

## Reproduce

    pip install numpy scipy pandas scikit-learn matplotlib pytest
    python -m pytest -q                 # ~30 s
    python -m experiments.run_all       # ~15 min on 1 core; rewrites results/
    python -m experiments.replot        # redraw figures from saved CSVs only

All seeds are fixed (`experiments/common.py`). Monte Carlo tables use 10 seeds x 10,000
paired views; everything else random uses 5 seeds. Intervals are 95% Student-t
intervals across seeds.

## The synthetic ledger

    # ledger 840311
    01. 56 - 34 = 22
    02. 13 + 16 = 29
    ...
    14. 12 - 10 =  3        <- wrong, and carries the tell (doubled space after '=')

* 12-20 base steps; T in {0,1,2} genuine errors with probabilities 0.8 / 0.15 / 0.05.
* Each view receives Delta ~ Poisson(1.5) (truncated at 8) plants from **the same
  function** (`apply_error`) that makes genuine errors: either corrupt a correct line
  or insert a new wrong line.
* `recognizability` p: each plant independently carries the tell with probability p.
* `marker=True`: the ledger id is odd iff the original is dirty (the perturbation never
  touches the header). With `marker=False` the parity is an independent coin.

### Why the summary is enough

A supervisor's view reduces to s = (K, R) = (wrong lines, wrong lines with a tell),
plus the marker bit in marker mode. Operands, positions and ledger length are
independent of T given s and independent across the two views' plants, so by Jensen's
inequality any reporting function of the full text can be replaced by its average
given s without raising the convex potential. Equilibria over functions of s are
therefore equilibria over functions of the text. `tests/test_ledger.py` checks the
distributional claims on generated text. For the same reason gamma is computed on the
reduced model (original = T); the marker experiment is exactly the case where benign
content stops being independent of T.
