# Paper Revision Progress

## Overview

This document summarizes the supplementary experiments conducted for the LeWM audit paper revision to address reviewer weaknesses W1-W5. The revision narrows the paper's scope claims, adds mechanism attribution for the CEM pool-ranking collapse, validates partial privilege-free monitoring signals, compares CEM with MPPI, extends the Cube full projected-CEM experiment to a larger scale, and documents PLDM cross-model feasibility.

## Revision Timeline

- 2026-05-05: Phase A -- Data inventory completed
- 2026-05-05: Phase B -- Rpool(V1) attribution experiment completed
- 2026-05-05: Phase C -- Privilege-free proxy analysis completed
- 2026-05-05: Phase D -- MPPI cross-optimizer comparison completed
- 2026-05-05: Phase E -- Cube full projected CEM extended (50 pairs x 3 seeds)
- 2026-05-05: Phase F -- PLDM feasibility assessed (deferred to future work)
- 2026-05-05: Phase G -- Paper text revised and compiled

## Experiment Results Summary

### Phase B: Rpool(V1) Attribution (addresses W2)

**Question:** Is R_pool approximately 0 caused by CEM convergence compression or local representation failure?

**Method:** Computed Spearman correlation of V1 oracle hinge cost vs C_real_state within the default CEM final pool across 100 PushT pairs.

**Key Results:**
- Overall: Rpool(V1) = 0.666 vs Rpool(C_model) = 0.084
- Invisible quadrant (16 pairs): Rpool(V1) = 0.775 vs Rpool(C_model) = 0.022
- Sign reversal (21 pairs): Rpool(V1) = 0.925 vs Rpool(C_model) = -0.040

**Conclusion:** Local representation failure is the dominant mechanism. V1 oracle retains strong ranking inside CEM pools where learned Euclidean cost fails.

**Artifacts:** `scripts/revision/rpool_v1_analysis.py`, `results/revision/rpool_v1_pusht.json`, `docs/revision/rpool_v1_memo.md`

### Phase C: Privilege-Free Proxy (addresses W5)

**Question:** Can metrics computable without simulator access predict selection regret?

**Method:** Correlated pool-level cost statistics that do not require C_real_state with selection regret across 100 pairs.

**Key Results:**
- Best global proxy: top30_Cmodel_std vs selection_regret, rho = 0.314 (p = 0.001)
- Best subset proxy: invisible_quadrant pool_Cmodel_std, rho = 0.588 (p = 0.017)
- Cross-check: pool_Cmodel_std tracks pool_Creal_std at rho = 0.482 (p < 0.001)

**Conclusion:** Partial but statistically significant monitoring is possible without simulator access. No single proxy crosses the 0.4 global threshold, but subset-specific signals exist.

**Artifacts:** `scripts/revision/privileged_free_proxy.py`, `results/revision/proxy_analysis_pusht.json`, `docs/revision/proxy_analysis_memo.md`

### Phase D: MPPI Cross-Optimizer Comparison (addresses W4)

**Question:** Is endpoint-planning decoupling CEM-specific or optimizer-general?

**Method:** Implemented MPPI planner (soft weighting, tau=1.0), ran on 30 PushT pairs x 3 seeds, and compared pool-level metrics with CEM.

**Key Results:**

| Metric | CEM default | MPPI (tau=1.0) |
|---|---:|---:|
| Planning success | 53.3% | 38.9% |
| Rpool(C_model) | 0.115 | 0.194 |
| Rpool(V1) | 0.537 | 0.707 |
| Pool C_real_state std | 7.947 | 29.478 |

**Conclusion:** Decoupling is partially optimizer-general (about 78% representation-driven) and partially CEM-specific (about 22% from hard truncation compressing pool diversity by 3.7x).

**Artifacts:** `scripts/revision/mppi_planner.py`, `scripts/revision/mppi_temperature_sweep.py`, `scripts/revision/mppi_30pair_experiment.py`, `scripts/revision/mppi_pool_analysis.py`, `results/revision/mppi_pusht_30pair.json`, `results/revision/mppi_pool_analysis.json`, `docs/revision/mppi_memo.md`

### Phase E: Cube Full Projected CEM Extended (addresses W3)

**Question:** Does the Cube evidence hold at larger scale?

**Method:** Extended Cube full projected CEM from 25 pairs x 1 seed to 50 pairs x 3 seeds across 5 projection dimensions. Total: 750 CEM rollouts with full 300-candidate simulator scoring.

**Key Results:**

| m | Full-CEM success (50p, 3s) | Re-rank success (100p, 3s) | Gap |
|---|---:|---:|---:|
| 1 | 32.7% +/- 8.1% | 42.7% +/- 3.2% | -10.0 pp |
| 8 | 52.7% +/- 4.2% | 43.3% +/- 1.5% | +9.3 pp |
| 32 | 61.3% +/- 1.2% | 46.3% +/- 2.5% | +15.0 pp |
| 64 | 56.0% +/- 5.3% | 49.7% +/- 2.5% | +6.3 pp |
| 192 | 62.0% +/- 6.0% | 47.7% +/- 1.2% | +14.3 pp |

Rpool(C_model) across all dimensions is approximately 0.001 to 0.023, near zero and consistent with PushT. The inverted-U pattern from the original 25-pair sweep does not persist; success plateaus from m=32 onward.

**Conclusion:** Endpoint-planning decoupling is robust across both environments at scale. The previous inverted-U was small-sample noise.

**Artifacts:** `scripts/revision/cube_full_proj_cem_extended.py`, `scripts/revision/cube_extended_summary.py`, `scripts/revision/cube_rpool_analysis.py`, `results/revision/cube_full_proj_cem_extended.json`, `results/revision/rpool_v1_cube.json`, `docs/revision/cube_full_cem_memo.md`

### Phase F: PLDM Cross-Model (deferred)

**Status:** Feasibility assessed. A public PLDM repository exists, but no ready-to-use PushT checkpoint was found at revision time. Cross-model validation is deferred to future work.

**Artifacts:** `docs/revision/pldm_memo.md`

## Paper Sections Modified

- `paper/main.tex` -- Abstract scope narrowed, revision markers added
- `paper/sections/decoupling.tex` -- New Section 4.4 mechanism attribution; Table 2 updated with 50-pair Cube data
- `paper/sections/discussion.tex` -- New MPPI subsection; privilege-free monitoring paragraph
- `paper/sections/limitations.tex` -- Updated scope boundaries

## File Inventory

### `scripts/revision/`

| File | Size |
|---|---:|
| `cube_extended_summary.py` | 12K |
| `cube_full_proj_cem_extended.py` | 48K |
| `cube_rpool_analysis.py` | 7.7K |
| `mppi_30pair_experiment.py` | 13K |
| `mppi_planner.py` | 22K |
| `mppi_pool_analysis.py` | 17K |
| `mppi_temperature_sweep.py` | 11K |
| `privileged_free_proxy.py` | 16K |
| `rpool_v1_analysis.py` | 27K |

### `results/revision/`

| File | Size |
|---|---:|
| `cube_full_proj_cem_extended.json` | 26,211,982 bytes |
| `cube_full_proj_cem_extended_smoke.json` | 273,228 bytes |
| `mppi_pool_analysis.json` | 129,942 bytes |
| `mppi_pusht_30pair.json` | 142,593 bytes |
| `mppi_temperature_sweep.json` | 185,566 bytes |
| `proxy_analysis_pusht.json` | 73,803 bytes |
| `rpool_v1_cube.json` | 435,652 bytes |
| `rpool_v1_pusht.json` | 772,617 bytes |

Pool files are excluded from the per-file list:

| Directory | `.pt` files | Total size |
|---|---:|---:|
| `cube_full_proj_pools/` | 750 | 670,343,838 bytes |
| `cube_full_proj_pools_smoke/` | 4 | 3,574,052 bytes |
| `mppi_pusht_pools/` | 90 | 35,538,042 bytes |

### `docs/revision/`

| File | Size |
|---|---:|
| `cube_full_cem_memo.md` | 2.0K |
| `mppi_memo.md` | 2.8K |
| `paper_revision_guide.md` | 17K |
| `pldm_memo.md` | 7.1K |
| `proxy_analysis_memo.md` | 2.7K |
| `revision_progress.md` | 7.1K |
| `rpool_v1_memo.md` | 3.3K |

## Reproducibility

All scripts are deterministic given the same seeds and can be rerun from the repository root. Phase E supports `--resume` for crash recovery. The paper compiles with `tectonic main.tex`.
