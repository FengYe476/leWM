# Phase 2 Progress Summary

This document records the completed Phase 2 work before starting the Cube cross-environment experiments in a new chat. The current status is:

- Step 1 C6 audit is complete, with verdict **C6-REAL**.
- Step 2 Stage 1B-Smoke is complete, with verdict **Continue B**.
- Full Stage 1B 9-dimension sweep is complete and plotted.
- Cube cross-environment setup is complete through pair sampling and latent artifact extraction.
- Cube Stage 1A C0-C5 is complete.
- Cube C6 replication is complete, with verdict **C6-NO-INVERSION**.
- Cube Stage 1B re-rank is complete, with verdict **CUBE-STAGE1B-FLAT-ROBUST**.
- No Cube replication item is currently pending.

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

## Completed: Cube Stage 1A C0-C5

Cube Stage 1A evaluated the learned Cube LeWM latent geometry and random-geometry controls C0-C5 over:

`100 pairs x 80 actions = 8000 records`

Key results:

| Control | Result |
| --- | --- |
| C0 trained LeWM | Spearman `+0.604`, pairwise accuracy `0.707` |
| C2 Gaussian `m=64` | Spearman `+0.575 +/- 0.019` |
| C4 Gaussian null | Spearman `-0.003 +/- 0.013` |
| C5 shuffled latent | Spearman `-0.001 +/- 0.011` |

The Cube C0 result is stronger than PushT C0 on endpoint ranking (`+0.604` vs `+0.506`) while the C4/C5 null floors remain near zero. This confirms that the learned Cube latent contains real goal-ranking structure.

Artifacts:

| Type | Path |
| --- | --- |
| Script | `scripts/phase2/cube/cube_stage1a.py` |
| Results | `results/phase2/cube/cube_stage1a.json` |

## Completed: Cube C6 Replication

Verdict: **C6-NO-INVERSION**

Cube C6 replicated the PushT C6 S1-S6 sub-experiments on OGBench-Cube. The random-init ViT does not invert the task signal on Cube: the 10-seed aggregate is weakly positive and no seeds are negative.

| Sub-experiment | Cube Spearman | PushT Spearman |
| --- | ---: | ---: |
| S1 random-init ViT 10-seed | `+0.115 +/- 0.016` | `-0.156 +/- 0.086` |
| S1 negative seeds | `0/10` | `8/10` |
| S3 eval seed0 | `+0.137` | `-0.200` |
| S3 train seed0 | `+0.078` | `+0.025` |
| S4 raw pixel L2 | `+0.069` | `+0.697` |
| S4 mean RGB diff | `+0.153` | `+0.705` |
| S5 object feature | `+0.99995` | `+0.426` |
| S6 small CNN eval | `+0.119` | `-0.182` |
| S6 ResNet18 eval | `+0.211` | `+0.304` |

Interpretation: PushT inversion depends on the presence of a strong pixel-level task signal in the 2D overhead view. Cube raw pixel distance is near zero because small 3D cube displacements produce weak and view-dependent pixel changes. The broader conclusion still generalizes: learned weights are essential. Cube trained LeWM reaches Spearman `+0.604`, far above raw pixel L2 `+0.069` and random-init ViT `+0.115`.

Replay validation used regenerated actions. Ordering matched the latent artifact, and replayed `C_real_state` matched within relaxed tolerance: max abs diff `0.0471125`, mean abs diff `3.05e-05`, `atol=0.05`; worst pair was pair `91`.

Artifacts:

| Type | Path |
| --- | --- |
| Script | `scripts/phase2/cube/cube_c6_audit.py` |
| Results | `results/phase2/cube/c6_audit/` |
| Summary | `results/phase2/cube/c6_audit/cube_c6_audit_summary.json` |
| Memo | `docs/phase2/cube/cube_c6_audit_memo.md` |

## Completed: Cube Stage 1B Re-rank

Verdict: **CUBE-STAGE1B-FLAT-ROBUST**

Cube Stage 1B tested projected-cost final-pool re-ranking on all `100` Cube pairs. It scored one default LeWM final CEM pool per pair:

`100 pairs x 300 candidates = 30,000 simulator rollouts`

Then it re-ranked the labelled pool for:

`100 pairs x {1,2,4,8,16,32,64,128,192} x 3 seeds = 2700 projected records`

Runtime:

| Run | Wall-clock | Estimated/actual full |
| --- | ---: | ---: |
| Smoke `5` pairs | `234.394s` (`3m54s`) | `1h18m08s` estimate |
| Full `100` pairs | `4747.490s` (`1h19m08s`) | actual |

Headline results:

| Dimension | Success | Spearman | False elite | Action L2 |
| --- | ---: | ---: | ---: | ---: |
| `m=1` | `42.7% +/- 3.2%` | `0.002` | `0.616` | `9.902` |
| `m=2` | `41.3% +/- 6.5%` | `0.008` | `0.615` | `9.879` |
| `m=4` | `44.7% +/- 1.2%` | `0.004` | `0.608` | `8.777` |
| `m=8` | `43.3% +/- 1.5%` | `0.005` | `0.608` | `7.487` |
| `m=16` | `43.3% +/- 0.6%` | `0.010` | `0.603` | `5.566` |
| `m=32` | `46.3% +/- 2.5%` | `0.007` | `0.609` | `4.238` |
| `m=64` | `49.7% +/- 2.5%` | `0.008` | `0.607` | `3.647` |
| `m=128` | `47.7% +/- 2.9%` | `0.009` | `0.605` | `2.470` |
| `m=192` | `47.7% +/- 1.2%` | `0.006` | `0.606` | `2.301` |

Default LeWM rank-1 success on the same pools is `49.0%`; average final-pool candidate success is `38.6%`.

Key findings:

- Cube does not show a sharp PushT-style planning elbow. Success is already `42.7%` at `m=1`, peaks at `49.7%` at `m=64`, and stays `47.7%` at `m=192`.
- Endpoint-planning decoupling is stronger on Cube: Stage 1A C2 endpoint Spearman rises from `0.238` to `0.603`, while Stage 1B final-pool Spearman stays near zero at every dimension.
- False elite rate is flat at about `0.61` across all dimensions, so projection does not collapse the Cube final-pool landscape.
- Action L2 to the default plan decreases smoothly with dimension, and LeWM top-30 overlap rises to `0.886` by `m=192`.

Artifacts:

| Type | Path |
| --- | --- |
| Script | `scripts/phase2/cube/cube_stage1b.py` |
| Smoke results | `results/phase2/cube/cube_stage1b_smoke.json` |
| Smoke stdout | `results/phase2/cube/cube_stage1b_smoke.log` |
| Full results | `results/phase2/cube/cube_stage1b.json` |
| Full stdout | `results/phase2/cube/cube_stage1b_full.log` |
| Memo | `docs/phase2/cube/cube_stage1b_memo.md` |

## Pending

No Cube replication work is pending. The next optional branch is method development or follow-up diagnostics, for example deciding whether a full projected-CEM Cube run is worth the extra compute after the re-rank result.

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

### F4. Cube Cross-Environment C6 Split

Cube does not reproduce the PushT random-init signal inversion: random-init ViT is `+0.115 +/- 0.016` over 10 seeds with `0/10` negative seeds. The environmental difference is raw pixel signal strength. PushT raw pixel L2 is `+0.697`, while Cube raw pixel L2 is only `+0.069`.

Paper framing: signal inversion is environment-dependent, but learned weights are essential in both environments. Cube strengthens the learned-representation claim because trained LeWM reaches `+0.604` from a near-zero raw pixel baseline.

### F5. Cube Low-Dimensional Re-rank Robustness

Cube Stage 1B re-rank does not reproduce PushT's sharp low-dimensional planning elbow. Cube success is already `42.7%` at `m=1`, peaks at `49.7%` at `m=64`, and remains `47.7%` at `m=192`, while final-pool Spearman stays near zero across all dimensions.

Paper framing: random projection preserves final-pool selection in both environments, but the dimensional curve is environment-dependent. PushT suggests a `32-64` dimensional planning-effective subspace; Cube suggests a flatter and more robust final-pool geometry under re-ranking.

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
| Cube Stage 1A script | `scripts/phase2/cube/cube_stage1a.py` |
| Cube Stage 1A results | `results/phase2/cube/cube_stage1a.json` |
| Cube C6 audit script | `scripts/phase2/cube/cube_c6_audit.py` |
| Cube C6 audit results | `results/phase2/cube/c6_audit/` |
| Cube C6 audit memo | `docs/phase2/cube/cube_c6_audit_memo.md` |
| Cube Stage 1B script | `scripts/phase2/cube/cube_stage1b.py` |
| Cube Stage 1B results | `results/phase2/cube/cube_stage1b_smoke.json`, `results/phase2/cube/cube_stage1b.json` |
| Cube Stage 1B logs | `results/phase2/cube/cube_stage1b_smoke.log`, `results/phase2/cube/cube_stage1b_full.log` |
| Cube Stage 1B memo | `docs/phase2/cube/cube_stage1b_memo.md` |
| This progress summary | `docs/phase2/progress_summary.md` |

Repository note: all scripts, results, memos, and figures for this block are committed and pushed to `https://github.com/FengYe476/leWM.git` on the `phase1` branch.
