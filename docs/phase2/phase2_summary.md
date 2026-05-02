# Phase 2 Summary: Cost-Landscape Calibration, Hybrid CEM, and Encoder Controls

## 1. Phase 2 Goal

Phase 2 tested whether the cost-landscape mismatch identified by Phase 1 Track A could be fixed by learning a better terminal cost over frozen LeWM latents. The central hypothesis was simple: if Phase 1 showed that V1 oracle costs rank action outcomes better than LeWM's Euclidean latent distance, then a learned cost head `C_psi(z, z_g)` trained on V1 labels might replace Euclidean cost inside CEM planning.

All Phase 2 experiments used PushT offset 50 and the Phase 1 Track A 100-pair stratified grid. Split 3 refers to the 16 `all_fail + strong_rho` hard pairs; Split 1 refers to the random 70/15/15 holdout with 15 test pairs.

## 2. Framing Reset

Phase 1 Track A revised the initial Phase 0 story. Phase 0 suggested that encoder geometry fails primarily under large displacement. Track A falsified that broad claim: D3, the largest-displacement row, had the best encoder-physics alignment, with Pearson `0.630`.

Phase 1 instead exposed two bottlenecks:

- Predictor sharpness becomes the dominant issue in D2-D3.
- Cost-criterion misalignment remains important in D0-D1 low-rotation cells.

The decisive Phase 1 result was that V1 oracle CEM beat Euclidean CEM in all `16/16` displacement-rotation cells, reaching `99%+` success in extreme cells. That result motivated Phase 2: if V1 cost fixes planning when used as an oracle, can a learned latent cost approximate that signal without simulator access?

## 3. P2-0: Learned Cost Head, Negative Result

### 3.1 Method

P2-0 kept the LeWM encoder and predictor frozen and trained scalar cost heads `C_psi(z, z_g)` on Track A endpoint records. The MLP heads used the feature vector:

`concat([z, z_g, z - z_g, abs(z - z_g)])`

With 192-d LeWM latents, this gives a 768-d input. The main variants were:

| Variant | Architecture | Purpose |
|---|---|---|
| MLP small | `768 -> 128 -> 128 -> 1`, ReLU | Main replacement-cost candidate |
| MLP large | `768 -> 512 -> 512 -> 512 -> 1`, ReLU | Capacity diagnostic |
| Mixed-latent MLP | Same as small, trained with endpoint and predicted latents | Distribution-shift diagnostic |
| Mahalanobis diagonal | Learned diagonal quadratic metric | Structured metric baseline |
| Mahalanobis low-rank | Learned low-rank PSD metric | Structured metric baseline |
| Mahalanobis full PSD | Learned full PSD metric | Structured metric baseline |
| CEM-aware distillation | Cost model trained on CEM-generated candidates with V1 labels | Search-distribution diagnostic |

Training used seed `0` and pairwise ranking hinge loss from V1 labels. Three split families were used:

| Split | Purpose |
|---|---|
| Split 1 | Random 70/15/15 holdout; 15 test pairs |
| Split 2 | Leave-one-cell-out over the 4 x 4 Track A grid |
| Split 3 | 16 all-fail + strong-rho hard pairs |

### 3.2 Metric-Level Results

The offline cost heads improved fixed-candidate ranking on held-out Split 3 records. This table reports the consolidated Split 3 metric artifacts used during P2-0; the Euclidean and offline MLP rows are the held-out Track A latent-ranking metrics, while the structured/CEM-aware rows are their predicted-latent test metrics where those artifacts were defined.

| Method | Pairwise Acc | Spearman |
|---|---:|---:|
| Euclidean | 0.630 | 0.269 |
| MLP offline small | 0.667 | 0.396 |
| MLP offline large | 0.674 | 0.479 |
| Mahalanobis diagonal | 0.568 | 0.274 |
| Mahalanobis low-rank | 0.614 | 0.301 |
| Mahalanobis full PSD | 0.592 | 0.201 |
| CEM-aware distillation | 0.579 | 0.227 |

The important pattern is not that every variant beat Euclidean. The important pattern is that the flexible MLP heads did improve ranking substantially on fixed endpoint candidates, especially the large MLP. A stricter CEM-relevant re-evaluation on `z_predicted` reduced the apparent gains, but still preserved the central lesson: better fixed-set ranking did not imply better planning.

### 3.3 Planning-Level Results

Planning was the binding test. No learned replacement cost produced a meaningful planning improvement.

| Method | Split 3, 16 pairs | Split 1, 15 pairs | Outcome |
|---|---:|---:|---|
| Euclidean | 0/16 | 4/15 | Baseline |
| MLP offline small | 0/16 | 3/15 | Metric improved, planning regressed on Split 1 |
| MLP offline large | 0/16 | not run | Split 3 only |
| MLP mixed-latent | 0/16 | not run | Fixed diagnostics, not planning |
| Mahalanobis low-rank | 0/16 | not run | Best Mahalanobis metric variant |
| CEM-aware distillation | 1/16 | 2/15 | First nonzero Split 3 result, still far below the `5/16` go bar |

The practical go signal was at least `5/16` rescues on Split 3 without obvious regressions on easier cases. Cost-head-only replacement did not reach it.

### 3.4 Diagnosis: Why Metric Gains Do Not Transfer

The failure was not a wiring bug. The learned cost hook changed selected actions, but CEM still could not use the learned landscape effectively.

Key diagnostics:

| Diagnostic | Result | Interpretation |
|---|---:|---|
| `C_psi` Spearman on `z_terminal` | 0.881 | Strong endpoint ranking on real encoder latents |
| `C_psi` Spearman on `z_predicted` | 0.768 | Predictor distribution shift, gap `0.113` |
| `C_psi` final CEM top-30 elite std | 0.0015 | Almost no elite discrimination |
| `C_psi` local perturbation std | 0.0238 | More than 300x flatter than Euclidean locally |
| Euclidean local perturbation std | 7.86 | Much stronger elite discrimination |
| Mixed-latent Spearman on `z_predicted` | 0.839 | Diagnostic shift improved |
| Mixed-latent elite std | 0.013 | Still compressed |
| Mixed-latent planning | 0/16 | Diagnostics improved but planning did not |

CEM dynamics diverged from iteration 1. The learned cost was locally too flat in the sampled action region to distinguish elites robustly. The root cause is therefore sharper than "ranking is imperfect": CEM needs a globally informative, high-spread cost landscape across the action space throughout iterative sampling, not just correct ranking over a fixed candidate set after the fact.

### 3.5 P2-0 Binding Decision

P2-0 is a no-go for cost-head-only calibration. Both pre-registered no-go criteria were met:

- Split 3 planning remained `0/16` for the main learned heads, with only `1/16` for CEM-aware distillation.
- Split 1 showed regressions for learned replacement costs, including MLP small `3/15` versus Euclidean `4/15` and CEM-aware `2/15`.

## 4. Hybrid CEM: Minimal Oracle Intervention, Positive Result

### 4.1 Method

Hybrid CEM kept the world model frozen and changed only elite selection. At each CEM iteration:

1. Sample 300 candidate action sequences.
2. Score all candidates with a cheap prefilter, usually Euclidean latent cost.
3. Keep the top `K` candidates.
4. Execute only those `K` candidates in the simulator.
5. Re-rank the `K` candidates by V1.
6. Select the best 30 as elites and refit the CEM distribution.

This is not full oracle CEM unless `K=300`. For `K=60`, only 20% of candidates receive simulator/V1 evaluation.

### 4.2 Oracle-Budget Tradeoff Curve

Split 3, Euclidean prefilter:

| K | Oracle Budget | Success | Delta vs K=0 |
|---:|---:|---:|---:|
| 0 | 0% | 0/16 | +0 |
| 30 | 10% | 0/16 | +0 |
| 60 | 20% | 7/16 | +7 |
| 90 | 30% | 10/16 | +10 |
| 150 | 50% | 12/16 | +12 |
| 300 | 100% | 13/16 | +13 |

Split 1, Euclidean prefilter:

| K | Oracle Budget | Success | Delta vs K=0 |
|---:|---:|---:|---:|
| 0 | 0% | 4/15 | +0 |
| 30 | 10% | 4/15 | +0 |
| 60 | 20% | 10/15 | +6 |
| 90 | 30% | 12/15 | +8 |
| 150 | 50% | 13/15 | +9 |
| 300 | 100% | 13/15 | +9 |

The main result is the phase transition between `K=30` and `K=60`. On Split 3, `K=30` remains `0/16`, while `K=60` rescues `7/16` previously unsolved hard pairs. The same curve shape appears on Split 1, where `K=60` jumps from `4/15` to `10/15`.

### 4.3 `C_psi` vs Euclidean Prefilter

`C_psi` did not improve the low-budget regime.

| K | Euclidean | `C_psi` | Advantage |
|---:|---:|---:|---:|
| 0 | 0/16 | 0/16 | 0 |
| 30 | 0/16 | 0/16 | 0 |
| 60 | 7/16 | 6/16 | -1 |
| 90 | 10/16 | 9/16 | -1 |
| 150 | 12/16 | 14/16 | +2 |
| 300 | 13/16 | 13/16 | 0 |

The learned head has no role as a low-budget prefilter. It shows a small advantage only at `K=150`, where the oracle subset is already broad.

### 4.4 Partial-Iteration Oracle

Hybrid CEM also tested whether oracle intervention can be sparse across iterations. On Split 3 with `K=60`:

| Config | Oracle iters | Rollouts/pair | Success |
|---|---|---:|---:|
| all-30 | 1-30 | 1800 | 7/16 |
| early-5 | 1-5 | 300 | 0/16 |
| early-10 | 1-10 | 600 | 2/16 |
| late-10 | 21-30 | 600 | 3/16 |
| every-3rd | 10 iters | 600 | 4/16 |
| first-only | 1 | 60 | 0/16 |

Oracle intervention must be continuous. Early-only intervention fails, and evenly spaced correction outperforms early-only or late-only at the same 600-rollout budget.

### 4.5 Compute Overhead

`K=60` adds about `2.7x` wall-clock over pure Euclidean CEM. In absolute terms this is roughly `30s` per episode on MPS. That is expensive for high-frequency online control, but feasible for offline planning, low-frequency control, and audit experiments where simulator calls are available.

## 5. Track B: DINOv2 Encoder Control

### 5.1 Method

Track B asked whether the cost-landscape mismatch is specific to LeWM's compact SIGReg encoder. DINOv2 ViT-B/14 was used as an external visual encoder over the same Track A terminal and goal observations.

Implementation details:

- DINOv2 model: `dinov2_vitb14`.
- Parameters: `86.6M`.
- Feature dimension: `768`.
- Device: MPS.
- Features: CLS token and mean-pooled patch tokens.
- Comparison target: V1 hinge cost over the same 100 pairs x 80 action records.
- Random projection control: seed-0 Gaussian projection of LeWM 192-d latents.

Because Track A latent artifacts did not store terminal pixels, terminal observations were replayed in the PushT simulator and encoded in batches.

### 5.2 Results

| Encoder | Dim | Global Spearman | Pairwise Acc | Per-Pair Rho Mean | Per-Pair Rho Std |
|---|---:|---:|---:|---:|---:|
| LeWM SIGReg | 192 | 0.5033 | 0.6470 | 0.3549 | 0.5021 |
| DINOv2 CLS | 768 | 0.2387 | 0.5943 | 0.2482 | 0.3651 |
| DINOv2 mean-pool | 768 | 0.2609 | 0.6099 | 0.2848 | 0.3686 |
| Random projection | 192 | 0.5010 | 0.6429 | 0.3493 | 0.4887 |

### 5.3 Interpretation

DINOv2 is worse than LeWM for this coarse PushT goal-ranking task. DINOv2 mean-pool beats LeWM in `5/16` cells, but loses clearly in aggregate.

The conclusion is not that all encoders would fail. The narrower, better-supported conclusion is that LeWM's compact task-trained encoder is not the primary bottleneck. Generic DINOv2 visual features are less informative for PushT endpoint goal ranking than LeWM's task-trained features. The random projection result being close to LeWM further suggests that much of the coarse ranking signal comes from preserved endpoint information, not a delicate learned Euclidean geometry.

Replacing the encoder with off-the-shelf DINOv2 does not address the CEM cost-landscape mismatch.

## 6. Headline Findings

F1. Metric-level cost ranking improvement does not transfer to CEM planning success: MLP heads improved Split 3 fixed-candidate ranking, but planning stayed `0/16`, and CEM-aware reached only `1/16`.

F2. The planning bottleneck is cost-landscape flatness in the CEM search region: local `C_psi` std was `0.0238` versus Euclidean `7.86`, and final CEM top-30 `C_psi` elite std was only `0.0015`.

F3. Euclidean top-30 excludes critical success candidates: `K=30` stays `0/16` on Split 3 even with simulator/V1 re-ranking.

F4. Expanding the coarse filter to `K=60`, a 20% oracle budget, rescues `7/16` previously unsolved Split 3 hard pairs.

F5. The cost-performance tradeoff shows a phase transition between `K=30` and `K=60`, with Split 3 jumping from `0/16` to `7/16`.

F6. Oracle intervention must be continuous throughout CEM iterations: early-only `5` iterations gives `0/16`, every-third gives `4/16`, and all-30 gives `7/16`.

F7. LeWM's compact encoder outperforms DINOv2 for coarse endpoint ranking: LeWM pairwise accuracy is `0.6470`, DINOv2 mean-pool is `0.6099`, and DINOv2 CLS is `0.5943`.

F8. The tradeoff curve shape is stable across pair subsets: Split 3 improves from `0/16` to `7/16` at `K=60`, while Split 1 improves from `4/15` to `10/15`.

## 7. Track C Status

Track C, the calibration ladder, was largely subsumed by P2-0.

| Ladder rung | Covered by | Result |
|---|---|---|
| L0: Euclidean baseline | Split 1/Split 3 baseline planning and ranking | Useful prefilter, insufficient planner objective |
| L1: Flexible learned scalar cost | MLP small/large offline cost heads | Ranking improves, planning fails |
| L2: Predictor-distribution calibration | `z_predicted` extraction and mixed-latent training | Diagnostics improve, planning stays `0/16` |
| L3: Structured metric family | Mahalanobis diagonal, low-rank, full PSD | No planning rescue |
| L4: CEM-aware calibration | CEM-aware distillation and hybrid CEM sweeps | Learned replacement still fails; oracle re-ranking works |

No additional Track C experiments are needed for the Phase 2 conclusion.

## 8. Phase 0 Revisions After Phase 2

Phase 2 further sharpens the Phase 0 and Phase 1 revisions:

- Phase 1 said the planner is part of the bottleneck. Phase 2 confirms and sharpens this: the specific bottleneck is CEM's need for continuous cost discrimination in the sampled action region, which frozen latent-space costs do not provide.
- Phase 1 said cost-criterion alignment explains D0/D1 low-rotation gaps. Phase 2 shows that even strong metric-level alignment from offline `C_psi` is insufficient for planning when the local CEM landscape is flat.
- Phase 1's V1 oracle results still hold as upper bounds. Phase 2 uses them to define an oracle-budget tradeoff curve rather than treating oracle CEM as a deployable method.

## 9. What Phase 2 Did Not Do

Phase 2 did not include:

- Encoder retraining. Track D was not triggered because the Track C-equivalent ladder showed structured failure was not simply capacity-bound.
- OGBench-Cube experiments.
- Offset 75 or 100 evaluation.
- Full DINO-WM planning comparison.
- Alternative planner architectures such as gradient-based planning, MPPI, or learned proposal distributions.

## 10. Open Questions

- Can the oracle budget be further reduced by active candidate selection, learning which candidates should be sent to the simulator?
- Does the hybrid CEM tradeoff curve transfer to other environments and world models?
- Would a differentiable planner bypass the CEM cost-landscape flatness issue?
- Can predictor improvements, especially sharper rollouts, reduce the gap between `K=0` and `K=60`?
- Is there a self-supervised proxy for V1 that avoids privileged physical state access?

## 11. Artifact Index

### P2-0 artifacts

- `results/phase2/p2_0/track_a_latents.pt`
- `results/phase2/p2_0/track_a_predicted_latents.pt`
- `results/phase2/p2_0/split1_small/`
- `results/phase2/p2_0/split1_large/`
- `results/phase2/p2_0/split3_small/`
- `results/phase2/p2_0/split3_large/`
- `results/phase2/p2_0/split2_small/`
- `results/phase2/p2_0/cem_aware_full/`
- `results/phase2/p2_0/cem_aware_split1/`
- `results/phase2/p2_0/mahalanobis_diagonal/`
- `results/phase2/p2_0/mahalanobis_lowrank/`
- `results/phase2/p2_0/mahalanobis_full/`
- `results/phase2/p2_0/planning_smoke.json`
- `results/phase2/p2_0/planning_gap_diagnosis.json`
- `results/phase2/p2_0/planning_gap_diagnosis_mixed.json`
- `results/phase2/p2_0/planning_gap_diagnosis_mixed_temp.json`
- `results/phase2/p2_0/split1_planning.json`
- `results/phase2/p2_0/split3_planning.json`
- `results/phase2/p2_0/split3_planning_large.json`
- `results/phase2/p2_0/split3_planning_mixed.json`
- `results/phase2/p2_0/split3_planning_mixed_temp.json`
- `results/phase2/p2_0/mahalanobis_planning.json`
- `results/phase2/p2_0/cem_aware_planning.json`
- `results/phase2/p2_0/cem_aware_split1_planning.json`
- `results/phase2/p2_0/deep_diagnosis.json`
- `results/phase2/p2_0/oracle_budget_cem_corrected/`

### Track B artifacts

- `results/phase2/track_b/dinov2_features.pt`
- `results/phase2/track_b/random_projection_features.pt`
- `results/phase2/track_b/ranking_comparison.json`

### Docs

- `docs/phase2/p2_0_memo.md`
- `docs/phase2/track_b_memo.md`
- `docs/phase2/phase2_summary.md`

### Scripts

- `scripts/phase2/__init__.py`
- `scripts/phase2/analyze_all_splits.py`
- `scripts/phase2/analyze_split1.py`
- `scripts/phase2/analyze_track_b.py`
- `scripts/phase2/cost_head_model.py`
- `scripts/phase2/dataloader.py`
- `scripts/phase2/deep_diagnosis.py`
- `scripts/phase2/diagnose_planning_gap.py`
- `scripts/phase2/eval_oracle_budget_cem.py`
- `scripts/phase2/eval_planning.py`
- `scripts/phase2/extract_dinov2_features.py`
- `scripts/phase2/extract_latents.py`
- `scripts/phase2/extract_pixels.py`
- `scripts/phase2/extract_predicted_latents.py`
- `scripts/phase2/mahalanobis_baseline.py`
- `scripts/phase2/random_projection_control.py`
- `scripts/phase2/splits.py`
- `scripts/phase2/track_b_common.py`
- `scripts/phase2/train_cem_aware.py`
- `scripts/phase2/train_cost_head.py`

## 12. Evidence Chain: Recommended Reading Order

| # | Document | What it covers |
|---:|---|---|
| 1 | `README.md` | Repository overview and Phase 0/1 status |
| 2 | `docs/phase1/track_a_summary.md` | Phase 1 Track A findings, prerequisite context |
| 3 | `docs/phase1/phase0_revisions.md` | How Phase 1 revised Phase 0 |
| 4 | `docs/phase2/phase2_summary.md` | This document: Phase 2 complete results |
| 5 | `docs/phase2/p2_0_memo.md` | Detailed P2-0 experiment record |
| 6 | `docs/phase2/track_b_memo.md` | DINOv2 encoder control details |
