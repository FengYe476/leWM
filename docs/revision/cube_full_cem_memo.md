# Cube Full Projected-CEM Extension Memo

Source files:
- Extended full projected CEM: `results/revision/cube_full_proj_cem_extended.json`
- Cube re-rank-only reference: `results/phase2/cube/cube_stage1b.json`
- Cube endpoint reference: `results/phase2/cube/cube_stage1a.json`

## Full-CEM vs Re-rank Success

| m | Cube full-CEM success (50p, 3s) | Cube re-rank success (100p, 3s) | Gap |
|---|---:|---:|---:|
| 1 | 32.7% +/- 8.1% | 42.7% +/- 3.2% | -10.0 pp |
| 8 | 52.7% +/- 4.2% | 43.3% +/- 1.5% | 9.3 pp |
| 32 | 61.3% +/- 1.2% | 46.3% +/- 2.5% | 15.0 pp |
| 64 | 56.0% +/- 5.3% | 49.7% +/- 2.5% | 6.3 pp |
| 192 | 62.0% +/- 6.0% | 47.7% +/- 1.2% | 14.3 pp |

## Extended Full-CEM Geometry

| m | R_endpoint reference | R_pool(projected) | Delta_CEM | R_pool(C_model) | pool_Creal_std | pool_success_mass |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.238 | 0.008 | 0.230 | 0.021 | 0.029 | 33.5% |
| 8 | 0.477 | 0.001 | 0.476 | 0.011 | 0.017 | 50.3% |
| 32 | 0.570 | -0.001 | 0.571 | -0.001 | 0.017 | 56.2% |
| 64 | 0.575 | -0.003 | 0.578 | -0.001 | 0.017 | 54.3% |
| 192 | 0.603 | 0.010 | 0.593 | 0.009 | 0.018 | 57.2% |

## Inverted-U Check

The original inverted-U pattern does not persist. Success rises strongly through m=32, but m=192 is the best dimension and the high-dimensional curve is better described as a plateau with seed noise.
 Mean full-CEM success by dimension was m=1: 32.7%, m=8: 52.7%, m=32: 61.3%, m=64: 56.0%, m=192: 62.0%.

## Cube Rpool(C_model) From Pool Files

| m | Rpool(C_model) | pool_Creal_std | pool_success_mass |
|---|---:|---:|---:|
| 1 | 0.023 +/- 0.115 | 0.029 +/- 0.016 | 33.5% |
| 8 | 0.012 +/- 0.120 | 0.017 +/- 0.010 | 50.3% |
| 32 | 0.001 +/- 0.090 | 0.017 +/- 0.011 | 56.2% |
| 64 | -0.001 +/- 0.083 | 0.017 +/- 0.010 | 54.3% |
| 192 | 0.008 +/- 0.081 | 0.018 +/- 0.012 | 57.2% |

The learned Euclidean cost remains a near-zero pool ranker in Cube full projected CEM: the best dimensional mean is 0.023. This supports the cross-environment endpoint-pool decoupling claim even after the 50-pair, three-seed extension.
