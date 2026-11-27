# Supervisors trained on their payments: results

Status 2026-11-27, midnight. Written by Claude. Work in progress: plain agreement on the
language model's features is still running. Choices that Liam has not confirmed are in section 7.
The experiments with prompted language models are in `REPORT_LLM.md`.

## 1. Summary

**Question.** In the prompted experiments the reviewers read the payment rule. Here nothing
is stated. Two supervisors are trained at the same time, each only on what it is paid. Do
they end up where the theory says, including where it says the mechanism fails?

**With perfect perception** (each supervisor is told which lines are wrong; 17 conditions,
three seeds each) the trained pairs sit on the exact equilibrium everywhere:

| Condition (private gain 0.5 per concealed error unless stated) | Share of system errors withheld: trained | Exact equilibrium |
|---|---|---|
| Plain agreement | 1.21 | 1.21 |
| Agreement only | 85 and falling | no bound |
| Floor only | 0.31 | 0.30 |
| Both terms (the mechanism) | 0.21 | 0.20 |
| Both terms, 50% / 90% / 100% of insertions recognizable | 0.27 / 0.65 / 1.20 | 0.26 / 0.66 / 1.21 |
| Both terms, private gain 0.1 / 0.25 / 1.0 | 0.055 / 0.11 / 0.39 | 0.053 / 0.11 / 0.39 |
| Both terms, surviving signal | 1.07 | 1.08 |
| No private gain: both terms | 0.013 | 0.015 |
| No private gain: plain agreement / recognizable insertions | 1.01 / 1.00 | 1.01 / 1.01 |

Shares above 1 mean reports below zero.

**With a language model's perception** (the supervisor reads a frozen Qwen2.5-0.5B model's
activations on the ledger text; hidden insertions; 1,500 training ledgers; three seeds):

| Condition | Share withheld: trained | Exact, perfect perception |
|---|---|---|
| Both terms, private gain 0.5 | 0.19 [0.17, 0.20] | 0.20 |
| Floor only, private gain 0.5 | 0.30 [0.28, 0.33] | 0.30 |
| Agreement only, private gain 0.5 | reports fall without limit | no bound |
| Both terms, no private gain | -0.40 [-0.47, -0.36] (over-reports) | 0.015 |

Trained with no label, a supervisor adds 0.97 to its report for each error of the audited
system and 0.94 for each inserted error. When every inserted error carries a tell, the share
withheld at a private gain of 0.5 rises from 0.19 to between 0.49 and 0.90, depending on how
much of the tell the supervisor's features expose (79% to 99.8%), close to the exact curve
(section 4.2). Plain agreement on these features: pending.

**What is weak.** With perfect perception this is close to the simulation in `REPORT.md`.
The language-model version uses a small frozen model on CPU; its perception is noisy (count
error 0.45 even when trained on true counts), which moves the outcome away from the exact
prediction at low stakes. The misalignment is a term in the training reward. The supervisor's
size was chosen after looking at test results. Section 6 lists every limit.

## 2. Setup

**Game.** As in every earlier experiment: ledgers with 0, 1 or 2 errors of the audited system
(probabilities 0.6, 0.3, 0.1); each supervisor gets its own copy with Poisson(1.5) inserted
errors made by the same error generator; the designer knows how many it inserted. Variants:
no insertion and identical copies (plain agreement); a share p of inserted errors carries a
visible tell (a doubled space after "="); the ledger number is odd exactly when the original
has an error (surviving signal).

**Supervisor.** A report head that reads per-line features of its copy and outputs a real
number: `F = w . sum_l phi(x_l) + rho(sum_l phi(x_l), psi(header))` (`hpo/trained.py`).
- *Perfect perception*: each line's features say whether it is wrong and whether it carries
  the tell; the header gives the parity of the ledger number; six noise columns are added.
- *Language model*: Qwen2.5-0.5B-Instruct, frozen, reads the rendered ledger once. A line's
  features are the hidden state at its last token in layers 8, 16 and 24 and three numbers
  from the model's own next-token probabilities for the stated result. For training on CPU
  the hidden states are reduced to 8 principal directions per layer (fitted without labels).

**Training.** Two supervisors with separate parameters are updated together. Each minimizes
its own realised cost, taking the other's report as given:

    cost_i = (V_i - V_j)^2 + max(Delta_i - F_i, 0)^2 + 0.02 F_i - b 1{T > 0} (T - V_i),   V = F - Delta.

The first two terms are the mechanism (each can be switched off), 0.02 F is a small reporting
charge, and b is the supervisor's private gain per concealed error. Two estimators:
`analytic` uses the gradient of the realised cost in the own report; `reinforce` submits a
noisy report (standard deviation 0.05), observes only the payment, and uses the payment of
the un-noised report as baseline. With the language-model features, 20% of the training
ledgers are held out and training stops at the step where the pair's own cost on them is
lowest.

**Exact prediction.** `hpo/equilibrium.py` solves for the equilibrium of the same cost over
all real-valued reporting functions of what a view shows (number of wrong lines, number with
a tell, parity), assuming perfect perception.

**Measures.** Share withheld = sum over dirty ledgers of (T - V) divided by the sum of T,
averaged over the two supervisors. Weight of a line = least-squares coefficient of the
report on the number of lines of that kind in the view.

## 3. Results with perfect perception

12,000 training and 4,000 test ledgers per condition, 3,000 steps, three seeds; brackets give
the range over seeds. Full table: `results/trained/perfect_perception/analysis/table.md`.
Figures: `fig_trained_main`, `fig_trained_dials`, `fig_trained_weights` in the same folder.

| Condition | Share withheld: trained | Exact | Charge per round: trained | Exact | Weight of a system error | Weight of an inserted error |
|---|---|---|---|---|---|---|
| Plain agreement | 1.209 | 1.208 | 0.028 | 0.027 | -0.17 | no insertions |
| Agreement only | 84.8 and falling | no bound | 19.4 | no bound | -1.66 | -1.68 |
| Floor only | 0.313 [0.300, 0.331] | 0.295 | 0.052 | 0.039 | 0.83 | 0.82 |
| Both terms | 0.206 [0.199, 0.215] | 0.202 | 0.037 | 0.032 | 0.94 | 0.94 |
| Both terms, 50% recognizable | 0.265 [0.259, 0.276] | 0.258 | 0.051 | 0.043 | 0.88 | 0.89 (1.00 with tell) |
| Both terms, 90% recognizable | 0.649 | 0.657 | 0.128 | 0.116 | 0.49 | 0.50 (1.00 with tell) |
| Both terms, 100% recognizable | 1.199 [1.193, 1.205] | 1.208 | 0.028 | 0.027 | -0.17 | 1.00 |
| Both terms, private gain 0.1 | 0.055 | 0.053 | 0.004 | 0.002 | 0.98 | 0.99 |
| Both terms, private gain 0.25 | 0.113 | 0.109 | 0.012 | 0.009 | 0.97 | 0.97 |
| Both terms, private gain 1.0 | 0.391 | 0.388 | 0.133 | 0.118 | 0.89 | 0.89 |
| Both terms, surviving signal | 1.069 [1.061, 1.076] | 1.082 | 0.039 | 0.037 | 0.10 | 0.98 |
| No private gain, plain agreement | 1.008 | 1.008 | 0.000 | 0.000 | 0.00 | no insertions |
| No private gain, both terms | 0.013 [0.010, 0.015] | 0.015 | 0.001 | 0.000 | 1.00 | 1.00 |
| No private gain, both terms, 100% recognizable | 1.002 | 1.008 | 0.001 | 0.000 | 0.00 | 1.00 |

The `reinforce` estimator, run on the first four rows, the 100% recognizable row and the
three no-gain rows, gives the same picture (both terms 0.202 [0.193, 0.212]; plain 1.207;
no gain, both terms -0.010; agreement only 75 and falling).

What the rows say:
- **Plain agreement gives no reason to report anything.** Even with no private gain, the
  reporting charge of 0.02 is enough for the pair to settle on reporting nothing.
- **Each term alone fails in its own way.** Without the floor a common downward shift is
  free and reports fall without limit. Without agreement the reports stay bounded and the
  supervisors shade more than under both terms (0.31 against 0.21 at this setup, where the
  floor alone is already strong; see `docs/experiment2_design.md` section 1).
- **Hiding is what makes the supervisor count the system's errors.** With no private gain
  and hidden insertions a system error adds 1.00 to the report. With recognizable insertions
  it adds 0.00: the supervisor has learned to report the insertions and nothing else.
- **The tolerance for recognizability and the response to the private gain** follow the
  exact curves (0.26 at 50%, 0.65 at 90%; roughly linear in the gain).
- **A surviving signal is used.** With the ledger number marking dirty originals, the pair
  hides almost everything on marked ledgers.

## 4. Results on a language model's activations

Frozen Qwen2.5-0.5B-Instruct on CPU; 1,500 training and 400 test ledgers per condition;
8 principal directions per layer plus the three probability features; head with 16 hidden
units; stopping on held-out own cost; three seeds.
`results/trained/qwen05_cpu/analysis/table.md`.

**Perception of the frozen model** (pilot, 160 ledgers): its own probability for the stated
result separates wrong from correct lines with AUC 0.993 (96% caught at 1% false alarms);
linear probes on hidden states alone reach AUC 0.72 to 0.89; the tell is read from hidden
states almost perfectly (99.5% caught, no false alarms). A head trained on the true counts has
a count error (RMSE) of 0.45 and is charged 0.35 per round by the mechanism.

### 4.1 Hidden insertions

| Condition | Estimator | Share withheld: trained | Exact, perfect perception | Weight of a system error | Weight of an inserted error | Charge per round |
|---|---|---|---|---|---|---|
| Trained on true counts (ceiling) | | 0.04 [0.02, 0.05] | | 0.95 | 0.92 | 0.35 |
| Both terms, no private gain | analytic | -0.40 [-0.47, -0.36] | 0.015 | 0.97 | 0.94 | 0.31 |
| | reinforce | -0.47 [-0.53, -0.43] | 0.015 | 0.97 | 0.93 | 0.30 |
| Both terms, private gain 0.1 | analytic | -0.16 [-0.18, -0.13] | 0.053 | 0.95 | 0.91 | 0.33 |
| Both terms, private gain 0.25 | analytic | 0.01 [-0.02, 0.03] | 0.109 | 0.93 | 0.89 | 0.36 |
| Both terms, private gain 0.5 | analytic | 0.19 [0.17, 0.20] | 0.202 | 0.89 | 0.85 | 0.44 |
| | reinforce | 0.13 [0.09, 0.17] | 0.202 | 0.91 | 0.88 | 0.41 |
| Both terms, private gain 1.0 | analytic | 0.45 [0.43, 0.49] | 0.388 | 0.84 | 0.80 | 0.60 |
| Floor only, private gain 0.5 | analytic | 0.30 [0.28, 0.33] | 0.295 | 0.71 | 0.68 | 0.11 |
| | reinforce | 0.32 [0.23, 0.40] | 0.295 | 0.71 | 0.68 | 0.12 |
| Agreement only, private gain 0.5 | both | reports fall without limit | no bound | | | |

- **Counting is learned from payments alone.** No label is used, and the weights on system
  errors and inserted errors (0.97 and 0.94) match those of a head trained on the true
  counts (0.95 and 0.92).
- **Noise in perception shows up as caution.** With no private gain the pair over-reports by
  about 0.5 errors per ledger. A report below the inserted count is punished quadratically,
  over-reporting costs only the reporting charge, and what both supervisors add on the lines
  they share cancels in the agreement term.
- **The response to the private gain is steeper than with perfect perception**: from -0.40
  at no gain to 0.45 at a gain of 1.0, against 0.015 to 0.39. The two curves cross near 0.5.
- **The two failure modes of single terms appear as predicted.**

### 4.2 Recognizable insertions

Same mechanism and payments; every inserted error carries the tell (a doubled space after
"="). The supervisor is told nothing about it. Standard supervisor (8 directions per layer),
three seeds:

| Condition | Estimator | Share withheld: trained | Exact, perfect perception | Weight of a system error | Weight of an inserted error | Charge per round |
|---|---|---|---|---|---|---|
| Trained on true counts (ceiling) | | 0.12 [0.11, 0.13] | | 0.81 | 0.96 | 0.20 |
| Both terms, no private gain | analytic | -0.08 [-0.12, -0.03] | 1.008 | 0.74 | 0.96 | 0.20 |
| | reinforce | -0.10 [-0.19, 0.03] | 1.008 | 0.75 | 0.95 | 0.19 |
| Both terms, private gain 0.5 | analytic | 0.48 [0.43, 0.54] | 1.208 | 0.61 | 0.91 | 0.30 |
| | reinforce | 0.51 [0.47, 0.56] | 1.208 | 0.61 | 0.90 | 0.30 |

Against hidden insertions with the same private gain (0.19 withheld, weights 0.89 and 0.85)
the supervisors withhold two and a half times as much and weight a system error well below
an inserted one. They stay far from the prediction for fully recognizable insertions.

**The gap is explained by how much of the tell the supervisor can read.** The supervisor
sees only a few directions of the model's activations. A linear probe on those directions
measures what share of the tells they expose. Varying the number of directions (two seeds
each, private gain 0.5):

| Directions per layer | Tells exposed (false alarms) | Share withheld: trained | Exact equilibrium when that share of insertions is recognizable | Weight of a system error | of an inserted error |
|---|---|---|---|---|---|
| 2 | 79% (2.2%) | 0.49 (0.56, 0.43) | 0.43 at 80% | 0.61 | 0.90 |
| 8 | 83% (1.5%) | 0.51 (0.54, 0.48) | 0.55 at 86% | 0.60 | 0.92 |
| 24 | 93% (0.8%) | 0.61 (0.61, 0.60) | 0.66 at 90%, 0.97 at 95% | 0.50 | 0.90 |
| 48 | 99.8% (0.0%) | 0.90 (0.91, 0.89) | 1.10 at 97%, 1.21 at 100% | 0.17 | 0.94 |

With no private gain the exact prediction stays below 0.09 up to 97% recognizable and jumps
to 1.0 at 100%. The trained pairs give -0.09, -0.10 and -0.09 for 2, 8 and 24 directions and
0.37 (0.44, 0.30) for 48.

So the more of the tell a supervisor can read, the more of the system's errors it leaves out,
along the curve the theory gives, and with nearly full access it leaves out most of them even
when it gains nothing from concealment beyond a negligible reporting charge. All numbers are
on fresh ledgers, so memory of the training copies (4.3) cannot produce them.

Plain agreement on these features: pending.

### 4.3 A finding from the first attempt: reused copies stop being hidden

My training set holds two fixed perturbed copies of each of 1,500 originals and shows them
again in every epoch. A supervisor with enough capacity memorises them. Checked directly
(48 directions per layer, 64 hidden units, 3,000 steps, no early stopping, one seed):

| | Ledgers | Withheld per dirty ledger | Weight of a system error | of an inserted error | Charge per round |
|---|---|---|---|---|---|
| No private gain | trained on | 0.65 | 0.48 | 0.97 | 0.000 |
| | fresh | 0.40 | 0.82 | 0.81 | 0.86 |
| Private gain 0.5 | trained on | 1.45 | -0.21 | 0.99 | 0.026 |
| | fresh | 0.90 | 0.67 | 0.62 | 2.20 |

On the copies they have trained on, the pair reports the inserted errors and leaves out the
system's, at a charge of 0.026. That is the outcome predicted for fully recognizable
insertions (1.51 withheld per dirty ledger, charge 0.027): the payments have taught each
supervisor which errors in a recurring copy were inserted, so for those copies the
perturbation is no longer hidden. On fresh ledgers the same pair treats both kinds of error
alike and is charged heavily. With 8 directions and 16 hidden units the two kinds get the
same weight on trained and fresh ledgers with no private gain (0.96 and 0.96 against 1.01
and 0.96); with a private gain of 0.5 a smaller gap remains on the trained ledgers (0.71
against 0.90). All results in 4.1 therefore use the small supervisor and stop at the lowest
held-out cost.

What this says about the mechanism: its guarantee is for copies whose insertions are fresh.
A designer who trains supervisors with it must draw new insertions every time and must judge
the result on ledgers that were not used in training.

## 5. Relation to the theory (`docs/theory.tex`)

- Theorem 2(a) says the game is a potential game whose equilibria are the minimizers of a
  convex potential, so supervisors that each follow the gradient of their own payment
  converge to them. Section 3 is that statement with neural report heads, sampled ledgers
  and, for `reinforce`, payments as the only feedback.
- Theorem 1 lists what each reward leaves free: any common shift under agreement only, any
  report at or above the inserted count under the floor only, only over-reporting under
  both. The three trained outcomes are these three sets.
- The recognizability sweep is the dependence on the gap, and the surviving-signal row is
  the signalling limit of the theory.
- The theory assumes that a supervisor's view determines the truthful report. With a noisy
  reader that fails, and the outcome moves: over-reporting at low stakes, a steeper
  response to the stake. The theory draft has no statement about this case.

## 6. Limits

1. **Perfect perception is close to the existing simulation.** `REPORT.md` section 3 already
   runs gradient play on the exact model. New here: neural heads over lines, the pay-only
   estimator, the no-gain conditions, and the weights on each kind of line.
2. **The language-model version is small.** One frozen 0.5B model, CPU, 1,500 training
   ledgers per condition, reduced features. A larger model on a GPU was planned and has not
   run.
3. **The model is not trained, only the head on its activations.** Code for training the
   model itself through an adapter exists (`hpo/trained_e2e.py`) and has only passed a smoke
   test.
4. **Supervisor size and feature reduction were chosen after looking at test results** on the
   hidden condition (48, 24, 8 and 2 directions were tried; the two small ones agree;
   `results/trained/qwen05_cpu/exploratory/`).
5. **Training reuses fixed copies.** Each original has two perturbed copies that recur in
   every epoch, which a large supervisor memorises (4.3). A cleaner design draws fresh
   ledgers and insertions at every step; that needs the model in the training loop and a GPU.
6. **The misalignment is put in by hand** as a term in the training reward, and the stake is
   linear and unbounded, so reports can fall below zero.
7. **The comparison target assumes perfect perception.** No exact prediction exists for a
   noisy reader.
8. **Toy task**, as before.

## 7. Choices made by Claude that Liam has not confirmed

- The whole design: report heads on a frozen model's activations, two separately
  parameterised supervisors, gradient play, the two estimators.
- The linear, uncapped private gain (the theory's form) and the reporting charge of 0.02.
- Share withheld as the net measure (uncapped), and the least-squares weights.
- Stopping on held-out own cost; 8 principal directions per layer; 16 hidden units.
- Presenting the effect of reused copies (4.3) as a finding. It was checked on one seed.

## 8. Reproduce

    pip install -r requirements.txt torch transformers peft scikit-learn
    python -m pytest tests/test_trained.py -q

    # perfect perception (CPU, about an hour on one thread)
    python experiments/exp9_trained.py synth --plans main nostake recog stake marker --seeds 3 \
        --out results/trained/perfect_perception/train_analytic.json
    python experiments/exp9_trained.py synth --plans main nostake --seeds 3 --estimators reinforce \
        --out results/trained/perfect_perception/train_reinforce.json
    python experiments/analyze_trained.py results/trained/perfect_perception/train_*.json \
        --out results/trained/perfect_perception/analysis

    # language model (CPU: about 1 ledger copy per second for the 0.5B model)
    python experiments/exp9_trained.py extract --model Qwen/Qwen2.5-0.5B-Instruct \
        --conds hidden rec100 plain --n-train 1500 --n-test 400 --out results/trained/qwen05_cpu
    python experiments/exp9_trained.py train --feats results/trained/qwen05_cpu --plans ceiling nostake main stake \
        --pca 8 --hidden 16 --val-frac 0.2 --batch 256 --seeds 3 --estimators analytic \
        --out results/trained/qwen05_cpu/train_hidden_analytic.json
    python experiments/analyze_trained.py results/trained/qwen05_cpu/train_*.json --out results/trained/qwen05_cpu/analysis

    # GPU (needs RUNPOD_API_KEY): see HANDOFF.md section 13.15
