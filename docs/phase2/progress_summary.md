# Phase 2 Progress Summary

This document records the completed Phase 2 work before starting the Cube cross-environment experiments in a new chat. The current status is:

- Step 1 C6 audit is complete, with verdict **C6-REAL**.
- Step 2 Stage 1B-Smoke is complete, with verdict **Continue B**.
- Full Stage 1B 9-dimension sweep is complete and plotted.
- Cube cross-environment setup is complete through pair sampling and latent artifact extraction.
- Pending work is Cube Stage 1A, Cube C6 replication, and Cube Stage 1B.

## Completed: Step 1 - C6 Audit (Random-Init Encoder Geometry)

Verdict: **C6-REAL**

The C6 audit tested whether the Stage 1A random-init encoder anti-correlation was a real geometry phenomenon, an implementation artifact, or an ambiguous seed accident. Six sub-experiments were completed.

| Sub-experiment | Description | Key result |
| --- | --- | --- |
| S1 | 10-seed random-init ViT | Aggregate Spearman `-0.156 +/- 0.086`; `8/10` seeds negative |
| S2 | Pre-projector vs post-projector | Identical (`-0.201` vs `-0.201`); projector ruled out |
| S3 | Eval mode vs train mode | Eval `-0.200`, train `+0.025`; LayerNorm mechanism identified |
| S4 | Raw pixel baselines | `raw_pixel_l2 +0.697`, `mean_rgb_diff +0.705`; random ViT inverts a `+0.70` pixel signal to `-0.20` |
| S5 | Hand-crafted block center | `+0.426`; confirms pixel-level task signal exists |
| S6 | Random CNN + ResNet18 | Small CNN eval `-0.182` replicates ViT flip; ResNet18 eval `+0.304` does not flip |

Key finding: random-init visual encoders with normalization layers in eval mode can invert task-relevant pixel signals. Severity depends on architecture depth: shallow networks such as ViT-tiny and a small CNN flip the signal, while a deep residual network such as ResNet18 does not. LeWM training therefore overcomes a worse-than-random starting point rather than merely adding signal to a neutral representation.

Artifacts:

| Type | Path |
| --- | --- |
| Scripts | `scripts/phase2/stage1/c6_audit.py` |
| Scripts | `scripts/phase2/stage1/c6_audit_s2_s3.py` |
| Scripts | `scripts/phase2/stage1/c6_audit_s4_s5.py` |
| Scripts | `scripts/phase2/stage1/c6_audit_s6.py` |
| Results | `results/phase2/stage1/c6_audit/` |
| Memo | `docs/phase2/c6_audit_memo.md` |

## Completed: Step 2 - Stage 1B-Smoke (Projected-Cost CEM Planning)

Verdict: **Continue B** (top-conference push path)

The Stage 1B-Smoke run tested whether projected LeWM costs preserve CEM planning success despite reduced latent dimension. It used `63` pairs, projection dimensions `{8, 32, 64, 192}`, and `3` projection seeds.

| Dimension | Success | Delta vs m=192 |
| --- | ---: | ---: |
| `m=8` | `20.6%` | `-13.8pp` |
| `m=32` | `28.0%` | `-6.4pp` |
| `m=64` | `29.6%` | `-4.8pp` |
| `m=192` | `34.4%` | baseline |

Joint decision matrix:

| Gate | Observed result | Decision |
| --- | --- | --- |
| C6 audit | C6-REAL | Random-init control is real and scientifically useful |
| m=64 planning drop | `-4.8pp` vs m=192 | Within the `10pp` continuation threshold |
| Landscape stability | No projected-cost collapse | Continue low-dimensional planning analysis |

The combined result was: **C6-REAL + m=64 within 10pp + no landscape collapse -> Continue B**.

Artifacts:

| Type | Path |
| --- | --- |
| Script | `scripts/phase2/stage1/projected_cem.py` |
| Results | `results/phase2/stage1/stage1b_smoke.json` |
| Results | `results/phase2/stage1/stage1b_sanity.json` |
| Memo | `docs/phase2/stage1b_smoke_memo.md` |

## Completed: Full Stage 1B Sweep (9-Dimension Ladder)

The full Stage 1B sweep expanded the smoke run to the full 100-pair ladder:

`100 pairs x {1,2,4,8,16,32,64,128,192} x 3 seeds = 2700 projected records`

The aggregate table below reports projected CEM_late performance by projection dimension. The Spearman column is the Stage 1B planning-pool Spearman over CEM's final clustered candidate pool.

| Dimension | Success | Spearman | Pairwise | False elite | Action L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `m=1` | `11.1%` | `0.026` | `0.510` | `0.690` | `14.268` |
| `m=2` | `13.8%` | `-0.027` | `0.489` | `0.692` | `14.608` |
| `m=4` | `15.3%` | `0.036` | `0.514` | `0.688` | `15.401` |
| `m=8` | `20.6%` | `0.080` | `0.530` | `0.689` | `14.898` |
| `m=16` | `23.8%` | `0.092` | `0.534` | `0.692` | `13.203` |
| `m=32` | `28.0%` | `0.111` | `0.544` | `0.686` | `12.209` |
| `m=64` | `29.6%` | `0.041` | `0.517` | `0.689` | `11.152` |
| `m=128` | `30.7%` | `0.051` | `0.521` | `0.687` | `9.664` |
| `m=192` | `34.4%` | `0.067` | `0.526` | `0.689` | `10.119` |

Key findings:

- Planning elbow at `m=32-64`: success improves sharply up to `m=32`, then returns diminish.
- Endpoint-planning decoupling: Stage 1A endpoint Spearman at `m=64` is `0.473`, while Stage 1B planning-pool Spearman at `m=64` is only `0.041`. Ranking preservation does not predict planning ranking inside the final CEM pool.
- False elite rate is flat at about `0.689` across all dimensions. Projection does not create a landscape collapse.
- Action L2 distance from the default plan decreases as dimension increases, even though planning Spearman remains near zero.

Artifacts:

| Type | Path |
| --- | --- |
| Results | `results/phase2/stage1/stage1b_full.json` |
| Figures | `results/phase2/figures/stage1b/` |
| Plot script | `scripts/phase2/stage1/plot_stage1b_full.py` |

Generated figures:

| Figure | Files |
| --- | --- |
| Planning Success vs Projection Dimension | `fig1_planning_success_vs_dimension.png`, `fig1_planning_success_vs_dimension.pdf` |
| Endpoint Ranking Preserves, Planning Ranking Does Not | `fig2_endpoint_planning_decoupling.png`, `fig2_endpoint_planning_decoupling.pdf` |
| Stage 1B multi-metric panel | `fig3_stage1b_multimetric_panel.png`, `fig3_stage1b_multimetric_panel.pdf` |

## Completed: Cube Cross-Environment Setup

### Cube Stratified Sampling

100 Cube pairs were sampled from OGBench-Cube using a `4 x 4` displacement x orientation grid. All cells were filled with no caps.

Artifacts:

| Type | Path |
| --- | --- |
| Script | `scripts/phase2/cube/sample_cube_pairs.py` |
| Config | `configs/phase2/cube_budget.json` |
| Results | `results/phase2/cube/cube_pairs.json` |

### Cube Latent Artifact

The Cube latent artifact has been extracted with `8000` records:

`100 pairs x 80 actions = 8000 records`

The artifact mirrors the PushT latent artifact schema where possible and adds Cube-specific diagnostics. Cube uses position-only success:

`success = ||terminal_cube_pos - goal_cube_pos||_2 <= 0.04`

For Cube, `v1_cost` aliases `C_real_state` because there is no PushT-style V1 hinge oracle:

`v1_cost = C_real_state = cube_pos_dist`

Success by source:

| Source | Success |
| --- | ---: |
| `data` | `24.1%` |
| `smooth_random` | `15.9%` |
| `CEM_early` | `24.9%` |
| `CEM_late` | `40.2%` |

Artifacts:

| Type | Path |
| --- | --- |
| Script | `scripts/phase2/cube/extract_cube_latents.py` |
| Results | `results/phase2/cube/cube_latents.pt` |

## Pending: Cube Stage 1A + C6 Replication

Next steps for the new chat:

1. Cube Stage 1A controls (`C0-C7`) - endpoint ranking analysis.
2. Cube C6 replication (`S1-S6`) - test whether signal inversion occurs in 3D.
3. Cube Stage 1B - test whether low-dimensional planning transfers to 3D.

The immediate next implementation target is Cube Stage 1A over `results/phase2/cube/cube_latents.pt`, using the PushT Stage 1A control structure as the reference while adapting the cost semantics to Cube's position-only success criterion.

## Key Paper Findings So Far

### F1. Random-Init Encoder Signal Inversion (C6-REAL)

Random-init ViT in eval mode inverts raw pixel task signal from approximately `+0.70` to approximately `-0.20`. LeWM training overcomes this worse-than-random starting point. The effect is architecture-dependent: shallow networks flip, while deep residual networks do not.

Paper framing: LeWM's useful geometry is learned, not inherited from a benign random visual encoder.

### F2. Low-Dimensional Planning Subspace

LeWM's `192`-dimensional latent space contains a roughly `32-64` dimensional planning-effective subspace. Beyond `m=32`, adding dimensions yields diminishing returns for CEM planning success.

Paper framing: planning-relevant geometry is distributed but low effective dimensionality is enough for much of the CEM success signal.

### F3. Endpoint-Planning Decoupling

Endpoint ranking quality from Stage 1A does not predict planning ranking quality in Stage 1B. CEM iterative search reshapes the candidate pool, destroying global ranking structure while preserving enough local cost geometry to drive planning success.

Paper framing: endpoint ranking and planning success are separable geometric properties. A representation can preserve endpoint rankings under random projection while still having weak ranking over the final clustered CEM pool, and yet still support planning.

## File Inventory

| Category | Paths |
| --- | --- |
| C6 audit scripts | `scripts/phase2/stage1/c6_audit.py`, `scripts/phase2/stage1/c6_audit_s2_s3.py`, `scripts/phase2/stage1/c6_audit_s4_s5.py`, `scripts/phase2/stage1/c6_audit_s6.py` |
| C6 audit results | `results/phase2/stage1/c6_audit/` |
| C6 audit memo | `docs/phase2/c6_audit_memo.md` |
| Stage 1B script | `scripts/phase2/stage1/projected_cem.py` |
| Stage 1B results | `results/phase2/stage1/stage1b_smoke.json`, `results/phase2/stage1/stage1b_sanity.json`, `results/phase2/stage1/stage1b_full.json` |
| Stage 1B memo | `docs/phase2/stage1b_smoke_memo.md` |
| Stage 1B figures | `results/phase2/figures/stage1b/` |
| Stage 1B plot script | `scripts/phase2/stage1/plot_stage1b_full.py` |
| Cube sampling script | `scripts/phase2/cube/sample_cube_pairs.py` |
| Cube sampling config | `configs/phase2/cube_budget.json` |
| Cube pair artifact | `results/phase2/cube/cube_pairs.json` |
| Cube latent extractor | `scripts/phase2/cube/extract_cube_latents.py` |
| Cube latent artifact | `results/phase2/cube/cube_latents.pt` |
| This progress summary | `docs/phase2/progress_summary.md` |

Repository note: all scripts, results, memos, and figures for this block are committed and pushed to `https://github.com/FengYe476/leWM.git` on the `phase1` branch.
