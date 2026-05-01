# Refactor Reproduction Check

Comparison: Phase 0 saved analysis vs refactored end-to-end offset-50 rerun.

Seeded reproduction: `true`; all checks passed: `true`.

| Metric | Phase 0 | Refactor | Delta | Tolerance | Pass |
| --- | ---: | ---: | ---: | ---: | :---: |
| corr(C_model, C_real_z) at offset 50 | 0.863965 | 0.863965 | 0.000000 | 0.005 | yes |
| per-pair mean corr(C_real_z, C_real_state) | 0.352971 | 0.352971 | 0.000000 | 0.005 | yes |
| per-pair std corr(C_real_z, C_real_state) | 0.486484 | 0.486484 | 0.000000 | 0.005 | yes |
| pairs with at least 1 successful action | 23 | 23 | 0 | exact | yes |
| CEM-late success rate | 52.667% | 52.667% | 0.000% | 0.005 | yes |
| random-action success rate | 15.667% | 15.667% | 0.000% | 0.005 | yes |

Notes: the rerun used the same saved seed and the same `30` pairs / `40` action sequences per pair setup. The refactored run wrote new artifacts under `results/phase1/` and did not overwrite Phase 0 result files.
