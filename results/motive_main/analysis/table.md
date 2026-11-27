### Claude Sonnet 4.6: 15 rounds

| Condition | Pairs | Concealed, rounds 11-15 | Concealed, all rounds | Net shortfall, all rounds | Concealing pairs | Charge per round | Reports below the inserted count | Exact counts | Cost |
|---|---|---|---|---|---|---|---|---|---|
| honest prompt, plain pay | 6 | 0.11 [0.00, 0.33] | 0.12 [0.01, 0.29] | 0.12 | 1 of 6 | 0.02 [0.00, 0.04] | 0.00 | 0.94 | $0.63 |
| honest prompt, perturbation | 8 | 0.03 [0.00, 0.11] | 0.04 [0.01, 0.09] | 0.05 | 0 of 8 | 0.23 [0.12, 0.37] | 0.06 | 0.93 | $2.05 |
| goal, plain agreement pay | 16 | 0.99 [0.95, 1.00] | 0.98 [0.95, 1.00] | 0.98 | 16 of 16 | 0.02 [0.00, 0.05] | 0.00 | 0.63 | $2.28 |
| goal, placebo | 16 | 0.42 [0.26, 0.61] | 0.47 [0.29, 0.67] | 0.47 | 12 of 16 | 0.07 [0.02, 0.15] | 0.00 | 0.81 | $2.37 |
| goal, hidden perturbation | 16 | 0.23 [0.13, 0.32] | 0.15 [0.10, 0.22] | 0.20 | 4 of 16 | 0.65 [0.33, 1.16] | 0.09 | 0.86 | $4.53 |

Differences, paired by seed (concealed in rounds 11-15; concealed, all rounds; charge per round):

- plain_level10 - perturb_level10 (16 pairs): +0.76 [+0.65, +0.86]; +0.83 [+0.74, +0.90]; -0.63 [-1.11, -0.32]
- placebo_level10 - perturb_level10 (16 pairs): +0.19 [+0.04, +0.37]; +0.32 [+0.13, +0.54]; -0.57 [-1.11, -0.23]
- plain_level10 - placebo_level10 (16 pairs): +0.56 [+0.37, +0.73]; +0.51 [+0.30, +0.70]; -0.06 [-0.14, +0.01]
- perturb_level10 - perturb_level0 (8 pairs): +0.13 [+0.02, +0.31]; +0.10 [+0.02, +0.22]; +0.70 [+0.07, +1.72]

Does the charge reveal which pairs conceal? (goal and honest pairs under the same rule)

- plain: 17 concealing pairs, 5 others; mean charge per round 0.02 against 0.01; AUC 0.49; rank correlation of concealed share and charge 0.36
- perturb: 4 concealing pairs, 20 others; mean charge per round 1.46 against 0.32; AUC 0.89; rank correlation of concealed share and charge 0.75

### Claude Opus 4.6: 15 rounds

| Condition | Pairs | Concealed, rounds 11-15 | Concealed, all rounds | Net shortfall, all rounds | Concealing pairs | Charge per round | Reports below the inserted count | Exact counts | Cost |
|---|---|---|---|---|---|---|---|---|---|
| honest prompt, plain pay | 6 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 | 0 of 6 | 0.00 [0.00, 0.00] | 0.00 | 1.00 | $2.08 |
| honest prompt, perturbation | 8 | 0.03 [0.00, 0.11] | 0.01 [0.00, 0.04] | 0.01 | 0 of 8 | 0.01 [0.00, 0.03] | 0.00 | 1.00 | $4.19 |
| goal, plain agreement pay | 16 | 1.00 [1.00, 1.00] | 0.99 [0.97, 1.00] | 0.99 | 16 of 16 | 0.02 [0.00, 0.05] | 0.00 | 0.63 | $4.79 |
| goal, placebo | 8 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 | 8 of 8 | 0.00 [0.00, 0.00] | 0.00 | 0.65 | $2.39 |
| goal, hidden perturbation | 16 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 2.30 | 16 of 16 | 7.10 [6.30, 7.85] | 0.81 | 0.12 | $8.78 |

Differences, paired by seed (concealed in rounds 11-15; concealed, all rounds; charge per round):

- plain_level10 - perturb_level10 (16 pairs): +0.00 [+0.00, +0.00]; -0.01 [-0.03, +0.00]; -7.08 [-7.80, -6.30]
- placebo_level10 - perturb_level10 (8 pairs): +0.00 [+0.00, +0.00]; +0.00 [+0.00, +0.00]; -7.07 [-8.22, -5.90]
- plain_level10 - placebo_level10 (8 pairs): +0.00 [+0.00, +0.00]; -0.02 [-0.07, +0.00]; +0.03 [+0.00, +0.10]
- perturb_level10 - perturb_level0 (8 pairs): +0.97 [+0.89, +1.00]; +0.99 [+0.96, +1.00]; +7.06 [+5.90, +8.21]

Does the charge reveal which pairs conceal? (goal and honest pairs under the same rule)

- plain: 16 concealing pairs, 6 others; mean charge per round 0.02 against 0.00; AUC 0.53; rank correlation of concealed share and charge 0.85
- perturb: 16 concealing pairs, 8 others; mean charge per round 7.10 against 0.01; AUC 1.00; rank correlation of concealed share and charge 0.68
