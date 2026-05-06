# V3 Same-Pool Ranker Decomposition (PushT)

Generated: `2026-05-05T23:49:00.153160+00:00`
Pairs completed: 100

## Decomposition Table

| Cost ranker | Rpool (mean ± std) | What it tests |
|---|---:|---|
| C_model (predicted latent) | 0.084 ± 0.243 | End-to-end: predictor + encoder + latent cost shape |
| C_V3 (actual terminal latent) | 0.258 ± 0.566 | Encoder + latent cost shape after removing predictor rollout |
| C_V1 (physical hinge) | 0.666 ± 0.357 | Physical ground truth ranker on the same pool |

## Per-Subset Breakdown

| Subset | n | C_model | C_V3 | C_V1 |
|---|---:|---:|---:|---:|
| invisible_quadrant | 16 | 0.022 ± 0.243 | 0.348 ± 0.504 | 0.775 ± 0.296 |
| sign_reversal | 21 | -0.040 ± 0.154 | -0.125 ± 0.696 | 0.925 ± 0.070 |
| latent_favorable | 12 | 0.197 ± 0.280 | 0.399 ± 0.315 | 0.417 ± 0.465 |
| v1_favorable | 13 | 0.015 ± 0.225 | 0.375 ± 0.589 | 0.837 ± 0.225 |
| ordinary | 47 | 0.146 ± 0.229 | 0.346 ± 0.512 | 0.581 ± 0.356 |

## Interpretation

Verdict: `mixed_predictor_plus_encoder_geometry`

V3 improves over predicted-latent C_model but remains well below V1, so predictor rollout contributes while learned terminal-latent geometry remains limiting.

## Replay Sanity Checks

- Stored-vs-real goal latent L2: 0.000 ± 0.000
- Max replay C_real absolute difference: 0.000 ± 0.000
- Replay success disagreement rate: 0.000 ± 0.000
