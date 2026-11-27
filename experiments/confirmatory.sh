#!/usr/bin/env bash
# Confirmatory run of the repeated-round experiment (HANDOFF.md section 13.9).
#
#   experiments/confirmatory.sh <model> <cap for the four main arms, USD> <cap for perturb/0, USD>
#
# Five arms, 20 pairs x 20 rounds each, fresh seeds 2000-2019 (the pilot used 1000-1009):
#   plain/0 (honest baseline), plain/5, perturb/5, placebo/5, and perturb/0 (known-effect check).
# Each command is issued twice: the second pass only reruns trajectories that failed on
# transient API errors, and makes no calls if the first pass was complete.
set -u
M=$1; CAP_MAIN=$2; CAP_BASE=$3
RUN="python experiments/exp6_llm_rounds.py"
ARGS="--backend anthropic --model $M --effort medium --trajectories 20 --rounds 20 --seed0 2000 --workers 10 --out results/llm_rounds"
for pass in 1 2; do
  $RUN compare $ARGS --level 5 --max-usd "$CAP_MAIN" || exit 1
done
for pass in 1 2; do
  $RUN compare $ARGS --level 0 --rules perturb --max-usd "$CAP_BASE" || exit 1
done
