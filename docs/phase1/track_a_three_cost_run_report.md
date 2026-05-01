# Track A Three-Cost Full Run Report

## Run Summary

- Output JSON: `results/phase1/track_a_three_cost.json`
- Pairs path: `/Users/fengye/Desktop/Project/leWM/lewm-failure-audit/results/phase1/track_a_pairs.json`
- Git commit recorded in output: `45d65afc15466686ed8d63c6427ef9e68ff1a497`
- Seed: `0`
- Device: `mps`
- Pairs completed: `100` / `100`
- Total wall-clock: `543.15` seconds (`9.05` minutes)
- Runner metadata elapsed time: `538.82` seconds (`8.98` minutes)
- Smoke extrapolation baseline: `12.2` minutes
- Failed pairs / resume: none

Correlation convention: cell-level and global `corr(C_model, C_real_z)` / `corr(C_real_z, C_real_state)` values below are Pearson correlations, matching the accepted smoke-run convention. The per-pair encoder-vs-state statistics are Spearman correlations, matching the Phase 0 per-pair diagnostic.

## Per-Cell Aggregate Table

| Cell | N_pairs | N_records | Success rate | Mean per-pair Spearman C_real_z vs C_real_state | Std per-pair Spearman C_real_z vs C_real_state | Cell Pearson C_model vs C_real_z |
|---|---:|---:|---:|---:|---:|---:|
| D0xR0 | 6 | 480 | 76.7% (368/480) | 0.388 | 0.161 | 0.841 |
| D0xR1 | 6 | 480 | 44.6% (214/480) | 0.406 | 0.260 | 0.899 |
| D0xR2 | 6 | 480 | 32.1% (154/480) | 0.318 | 0.535 | 0.939 |
| D0xR3 | 6 | 480 | 4.6% (22/480) | -0.273 | 0.406 | 0.800 |
| D1xR0 | 6 | 480 | 27.1% (130/480) | 0.499 | 0.323 | 0.797 |
| D1xR1 | 6 | 480 | 7.5% (36/480) | 0.338 | 0.419 | 0.845 |
| D1xR2 | 6 | 480 | 19.0% (91/480) | 0.144 | 0.588 | 0.949 |
| D1xR3 | 6 | 480 | 0.0% (0/480) | -0.286 | 0.414 | 0.789 |
| D2xR0 | 6 | 480 | 15.0% (72/480) | 0.317 | 0.733 | 0.794 |
| D2xR1 | 6 | 480 | 17.3% (83/480) | 0.597 | 0.299 | 0.633 |
| D2xR2 | 7 | 560 | 0.9% (5/560) | 0.075 | 0.460 | 0.829 |
| D2xR3 | 7 | 560 | 0.2% (1/560) | 0.384 | 0.432 | 0.794 |
| D3xR0 | 6 | 480 | 7.9% (38/480) | 0.667 | 0.336 | 0.902 |
| D3xR1 | 7 | 560 | 11.4% (64/560) | 0.726 | 0.264 | 0.826 |
| D3xR2 | 6 | 480 | 2.7% (13/480) | 0.581 | 0.376 | 0.752 |
| D3xR3 | 7 | 560 | 1.8% (10/560) | 0.401 | 0.328 | 0.652 |

## Global Numbers

- Pearson corr(C_model, C_real_z): `0.813937`
- Spearman corr(C_model, C_real_z): `0.810698`
- Pearson corr(C_real_z, C_real_state): `0.489240`
- Spearman corr(C_real_z, C_real_state): `0.506414`

| Source | N_records | Success rate |
|---|---:|---:|
| data | 2000 | 6.7% (133/2000) |
| smooth_random | 2000 | 6.2% (124/2000) |
| CEM_early | 2000 | 18.6% (371/2000) |
| CEM_late | 2000 | 33.6% (673/2000) |

## Readiness Summary

The full Track A run completed cleanly on all 100 stratified pairs with no failed pairs and no resume. The global Pearson corr(C_model, C_real_z) was 0.814, close to the Phase 0 offset-50 value of about 0.864, while Pearson corr(C_real_z, C_real_state) across all records was 0.489. Every cell has at least six evaluated pairs, so the stratified table is ready for the heatmap and DP1 statistical-check prompt. The harder high-displacement / high-rotation cells show lower success and weaker real-encoder-vs-state alignment, which is the expected regime for the next analysis rather than a runner failure.

No requested flags were triggered: global Pearson corr(C_model, C_real_z) is above 0.5 and every cell has at least four evaluated pairs.
