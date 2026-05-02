# D3 Oracle Cost-Criterion Ablation Report

## 1. Provenance

- Pair source: `/Users/fengye/Desktop/Project/leWM/lewm-failure-audit/results/phase1/track_a_pairs.json`
- Track A reference source: `/Users/fengye/Desktop/Project/leWM/lewm-failure-audit/results/phase1/track_a_three_cost.json`
- Git commit: `399317f828d8d4f13a0c4d833f0ac15750a74bef`
- Seed: `0`
- Cells: `D3xR0, D3xR1, D3xR2, D3xR3`
- V3 wall-clock: `952.62` seconds

| Cell | Track A ref | V3 oracle | Delta pp | Gate |
|---|---:|---:|---:|---|
| D3xR0 | 7.9% | 32.9% | 25.00 | fail |
| D3xR1 | 11.4% | 17.5% | 6.07 | fail |
| D3xR2 | 2.7% | 9.0% | 6.25 | fail |
| D3xR3 | 1.8% | 9.6% | 7.86 | fail |

V3 sanity gate verdict: `fail`.

## 2. Variant Headline Table

| Cell | V3 success |
|---|---:|
| D3xR0 | 32.9% |
| D3xR1 | 17.5% |
| D3xR2 | 9.0% |
| D3xR3 | 9.6% |
| D3_overall | 17.0% |

| Cell | V3 CEM_late success |
|---|---:|
| D3xR0 | 85.8% |
| D3xR1 | 44.3% |
| D3xR2 | 27.5% |
| D3xR3 | 35.7% |
| D3_overall | 47.7% |

## 3. Cost-Shape Interpretation Table

| Variant | C_real_state mean/std | C_variant mean/std | Pearson C_variant/C_real_state | Pearson C_variant/success | Median best C_real_state |
|---|---:|---:|---:|---:|---:|
| V3 | 82.450 / 69.167 | 82.450 / 69.167 | 1.000 | -0.510 | 1.099 |

## 4. Headline Finding

V1/V2 were not run because the V3 sanity gate did not pass.

## 5. Limitations

- This ablation only tests D3 row; D0/D1/D2 may behave differently.
- Oracle CEM has access to ground-truth state, which the deployed system does not; V1/V2 are upper bounds.
- alpha = 20 / (pi/9) is one specific choice; results may be sensitive to alpha.
- V2 indicator cost has zero gradient inside the success region, so CEM may behave qualitatively differently.
- Data and smooth_random records are duplicated per variant for easy per-variant aggregation.
