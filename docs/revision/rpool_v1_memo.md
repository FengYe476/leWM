# Rpool(V1) PushT Final-Pool Attribution Memo

Generated: `2026-05-04T19:15:14.734203+00:00`

## Main Table

Values are mean +/- sample std. Rank correlations use the effective convention for this memo: undefined Spearman from constant costs counts as `0.0` ranking signal.

| row | Rpool(C_model) | Rpool(V1) | Rpool(C_proj) | pool_Creal_std | pool_success_mass |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.084 +/- 0.243 (n=100) | 0.666 +/- 0.357 (n=100) | NA | 10.474 +/- 10.854 (n=100) | 0.331 +/- 0.416 (n=100) |
| m=8 | NA | NA | 0.073 +/- 0.272 (n=100) | 10.474 +/- 10.854 (n=100) | 0.331 +/- 0.416 (n=100) |
| m=32 | NA | NA | 0.107 +/- 0.246 (n=100) | 10.474 +/- 10.854 (n=100) | 0.331 +/- 0.416 (n=100) |
| m=64 | NA | NA | 0.079 +/- 0.256 (n=100) | 10.474 +/- 10.854 (n=100) | 0.331 +/- 0.416 (n=100) |
| m=192 | NA | NA | 0.095 +/- 0.253 (n=100) | 10.474 +/- 10.854 (n=100) | 0.331 +/- 0.416 (n=100) |

## Per-Subset Breakdown

| subset | n | Rpool(C_model) | Rpool(V1) | Cproj m=8 | Cproj m=32 | Cproj m=64 | Cproj m=192 | pool_Creal_std | pool_success_mass | selection_regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| invisible_quadrant | 16 | 0.022 | 0.775 | 0.038 | 0.138 | 0.010 | 0.038 | 12.671 | 0.014 | 36.110 |
| sign_reversal | 21 | -0.040 | 0.925 | 0.078 | -0.005 | -0.048 | -0.018 | 11.367 | 0.014 | 20.686 |
| latent_favorable | 12 | 0.197 | 0.417 | 0.133 | 0.179 | 0.207 | 0.169 | 7.727 | 0.681 | 6.604 |
| v1_favorable | 13 | 0.015 | 0.837 | 0.050 | 0.139 | -0.029 | 0.046 | 14.538 | 0.099 | 54.135 |
| ordinary | 47 | 0.146 | 0.581 | 0.084 | 0.130 | 0.154 | 0.157 | 9.673 | 0.503 | 14.522 |

## Invisible-Quadrant Focus

Low `pool_Creal_std` threshold: empirical Q25 = `3.750`.

| pair_id | cell | Rpool(V1) | Rpool(C_model) | pool_Creal_std | pool_success_mass | selection_regret |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 25 | D1xR0 | 1.000 | 0.061 | 0.203 | 0.000 | 1.333 |
| 46 | D1xR3 | 0.371 | -0.598 | 3.465 | 0.000 | 15.951 |
| 60 | D2xR2 | 0.703 | -0.123 | 5.009 | 0.000 | 28.628 |
| 61 | D2xR2 | -0.097 | 0.357 | 3.715 | 0.000 | 4.662 |
| 67 | D2xR3 | 0.461 | 0.282 | 41.890 | 0.000 | 103.044 |
| 70 | D2xR3 | 0.870 | -0.028 | 2.825 | 0.000 | 6.556 |
| 71 | D2xR3 | 0.941 | -0.236 | 9.308 | 0.000 | 38.089 |
| 73 | D2xR3 | 0.780 | 0.231 | 32.630 | 0.000 | 25.337 |
| 78 | D3xR0 | 0.993 | 0.215 | 15.898 | 0.217 | 25.821 |
| 86 | D3xR1 | 0.950 | -0.009 | 5.838 | 0.000 | 24.303 |
| 87 | D3xR2 | 0.996 | -0.123 | 7.541 | 0.000 | 23.149 |
| 93 | D3xR3 | 0.952 | -0.021 | 3.761 | 0.000 | 7.803 |
| 94 | D3xR3 | 0.866 | 0.122 | 10.780 | 0.000 | 84.312 |
| 96 | D3xR3 | 0.931 | 0.052 | 35.731 | 0.000 | 143.277 |
| 97 | D3xR3 | 0.841 | 0.314 | 7.103 | 0.000 | 5.349 |
| 99 | D3xR3 | 0.846 | -0.144 | 17.043 | 0.000 | 40.149 |

## Interpretation

Overall, effective `Rpool(V1)` is `0.666`, effective `Rpool(C_model)` is `0.084`, and mean `pool_Creal_std` is `10.474`. By the pre-registered rules, the overall classification is: **Pre-registered rules do not select one dominant mechanism**.

For the 16 invisible-quadrant pairs, effective `Rpool(V1)` is `0.775`, effective `Rpool(C_model)` is `0.022`, and mean `pool_Creal_std` is `12.671`. The invisible-quadrant classification is: **Local representation failure is the dominant mechanism**.
