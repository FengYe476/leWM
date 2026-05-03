# Cube C6 Audit Memo

## 1. Verdict

Verdict: **C6-NO-INVERSION**.

The Cube random-init LeWM ViT is weakly positive rather than inverted. Cube does not reproduce the PushT C6 signal inversion because the raw pixel task signal is itself weak in the 3D rendered view. The broader conclusion still transfers: learned weights are essential for useful goal ranking, and Cube LeWM training creates a much stronger task geometry than either raw pixels or random visual encoders.

## 2. Sub-experiment Results

Evidence: `results/phase2/cube/c6_audit/cube_c6_audit_summary.json`, `results/phase2/cube/c6_audit/*.json`, and PushT reference results in `results/phase2/stage1/c6_audit/`.

| Sub-experiment | Metric | Cube Spearman | PushT Spearman | Cube readout |
|---|---|---:|---:|---|
| S1 | random-init ViT 10-seed | +0.115 +/- 0.016 | -0.156 +/- 0.086 | weak positive |
| S1 | negative seeds | 0/10 | 8/10 | no seed-level inversion |
| S2 | pre-projector seed0 eval | +0.125 | -0.201 | positive before projector |
| S2 | post-projector seed0 eval | +0.137 | -0.200 | projector not causal |
| S3 | eval mode seed0 | +0.137 | -0.200 | eval stays positive |
| S3 | train mode seed0 | +0.078 | +0.025 | train is lower than eval |
| S4 | raw pixel L2 | +0.069 | +0.697 | raw pixel signal is weak |
| S4 | mean RGB diff | +0.153 | +0.705 | weak but positive |
| S5 | object feature | +0.99995 | +0.426 | metric pipeline validated |
| S6 | small CNN eval | +0.119 | -0.182 | no shallow-CNN flip on Cube |
| S6 | ResNet18 eval | +0.211 | +0.304 | positive in both environments |

Reference Cube Stage 1A values:

| Control | Spearman | Pairwise Acc | Per-pair rho | False Elite Rate |
|---|---:|---:|---:|---:|
| C0 trained LeWM | +0.604 | 0.707 | +0.431 | 0.549 |
| C2 Gaussian m=64 | +0.575 +/- 0.019 | 0.694 +/- 0.006 | +0.408 | 0.562 |
| C4 Gaussian null | -0.003 +/- 0.013 | 0.500 +/- 0.004 | +0.001 | 0.691 |
| C5 shuffled latent | -0.001 +/- 0.011 | 0.500 +/- 0.003 | +0.002 | 0.689 |

## 3. Per-sub-experiment Analysis

### S1: 10-seed random-init LeWM ViT

S1 tested whether the PushT random-init ViT inversion is stable across random seeds on Cube. The Cube aggregate Spearman is `+0.115 +/- 0.016`, with pairwise accuracy `0.573 +/- 0.002`, and `0/10` seeds are negative.

This is the opposite of PushT, where the aggregate was `-0.156 +/- 0.086` and `8/10` seeds were negative. On Cube, the random-init ViT is not useful in an absolute sense, but it is also not worse than the C4/C5 null floors. The single strongest conclusion is that the PushT inversion is not a universal property of the random LeWM ViT architecture.

### S2: pre-projector vs post-projector

S2 tested whether the projection head causes the random-init behavior. Cube pre-projector Spearman is `+0.125`, and post-projector Spearman is `+0.137`.

As in PushT, the projector is not the explanatory mechanism. PushT was negative on both sides of the projector; Cube is positive on both sides. The sign difference is therefore upstream of the projector and tied to the image distribution plus random visual stack.

### S3: eval mode vs train mode

S3 tested whether eval-mode normalization creates the signal flip. Cube seed0 eval mode is `+0.137`, while train mode is `+0.078`.

This reverses the PushT direction. On PushT, eval mode was `-0.200` and train mode moved toward zero at `+0.025`, supporting the LayerNorm eval-mode inversion diagnosis. On Cube, eval mode is higher than train mode and both are positive. The normalization effect is therefore image-distribution dependent rather than a fixed sign property of the architecture.

### S4: raw pixel baselines

S4 tested whether the rendered images contain a direct pixel-level task signal. Cube raw pixel L2 has Spearman `+0.069`, and mean RGB difference has Spearman `+0.153`.

This is the key environmental difference. PushT raw pixels are strongly task-positive: raw pixel L2 `+0.697` and mean RGB difference `+0.705`. Cube's 3D camera view makes small cube displacements produce relatively small and view-dependent pixel changes, so the raw pixel ranking signal is near zero. There is little strong positive pixel signal for the random ViT to invert.

### S5: cube position object feature

S5 used the object-state distance `||terminal_cube_pos - goal_cube_pos||_2`. Cube Spearman is `+0.99995`.

This validates the metric pipeline because Cube's ranking target is exactly the cube position distance up to replay and floating-point details. The result confirms that the C6 metrics can recover the correct ordering when given the task-relevant object state directly.

### S6: random CNN and ResNet18

S6 tested whether other random architectures reproduce the sign behavior. Cube small CNN eval Spearman is `+0.119`, and random ResNet18 eval Spearman is `+0.211`.

PushT small CNN eval reproduced the inversion at `-0.182`, while PushT ResNet18 stayed positive at `+0.304`. Cube keeps both random architectures positive. The architecture-dependence remains, but the shallow-network inversion does not transfer to the Cube image distribution.

## 4. Cross-environment Comparison

The central cross-environment result is that PushT has strong raw pixel task geometry and Cube does not. In PushT's overhead 2D view, moving the block changes a large coherent pixel region, so simple image distances track object displacement. In Cube's 3D rendering, many physical cube motions produce smaller, depth-dependent, lighting-dependent, or partially occluded pixel changes. Raw pixel distance is therefore a weak proxy for the state-space goal distance.

That difference explains why inversion appears on PushT but not Cube. PushT starts from a strong positive pixel signal around `+0.70`; the random eval-mode ViT and shallow CNN can flip that signal below zero. Cube starts from raw pixel L2 around `+0.07`; there is no comparable strong positive signal to invert, and the random visual encoders remain weakly positive.

The relative training gain is larger on Cube. Cube trained LeWM C0 reaches Spearman `+0.604`, compared with raw pixel L2 `+0.069`, a gain of about `+0.535`. PushT trained LeWM C0 reaches `+0.506` while raw pixel L2 is already `+0.697`, so PushT training shapes representation geometry but does not beat the raw pixel L2 baseline on this endpoint ranking metric. Cube is the stronger evidence that LeWM learns task-relevant geometry not present in simple pixel distance.

What holds across both environments:

- Random-init encoders are far below trained LeWM.
- Learned weights are essential for useful goal ranking.
- The projector is not the cause of C6 behavior.
- ResNet18 random features remain positive rather than inverted.

What is environment-specific:

- Signal inversion is present on PushT and absent on Cube.
- Eval/train mode direction differs: PushT eval is worse than train, Cube eval is better than train.
- Raw pixel baseline strength differs sharply: PushT is strong, Cube is near zero.

## 5. Implications for the Paper

The PushT C6-REAL result should be framed as environment-dependent, not universal. The precise claim is not "random visual encoders always invert task signal." The stronger and more defensible claim is that untrained visual stacks can produce misleading goal-ranking geometry, and the sign and severity depend on the interaction between architecture, normalization mode, and image distribution.

Cube strengthens the paper in a different direction. LeWM training overcomes a near-zero raw pixel baseline on Cube: C0 `+0.604` versus raw pixel L2 `+0.069` and random-init ViT `+0.115`. That is a cleaner learned-representation claim than PushT, where raw pixel L2 is already strong.

The shared paper finding is therefore:

> Learned weights are essential for goal ranking in both environments, but the failure mode of random visual geometry is environment-specific. PushT shows random-init inversion of a strong pixel signal; Cube shows learned representation recovery from weak pixel signal.

This framing preserves the scientific value of PushT C6 while avoiding overgeneralization. It also gives the TMLR empirical study a stronger cross-environment story: LeWM's learned geometry matters both when raw pixels are strong but random encoders distort them, and when raw pixels are weak and the learned encoder must construct the task geometry.

## 6. Replay Validation Notes

The Cube C6 replay used regenerated actions because `cube_latents.pt` contains `action_id` and `action_key` but not cached raw blocked action sequences. The script therefore regenerated the 80 actions per pair using the Cube extraction helpers and validated the replay against the latent artifact.

Replay validation passed:

| Check | Result |
|---|---:|
| pair_id order | matched |
| action_id order | matched |
| source order | matched |
| source_index order | matched |
| cell order | matched |
| C_real_state max abs diff | 0.0471125 |
| C_real_state mean abs diff | 0.0000305 |
| tolerance | 0.05 |
| passed | true |

The worst pair was pair `91`, with per-pair max absolute `C_real_state` difference `0.0471125` and mean difference `0.0007180` over 80 records. This is consistent with small regenerated-action physics differences rather than an ordering or metric mismatch.

## 7. Artifacts

| Type | Path |
|---|---|
| Script | `scripts/phase2/cube/cube_c6_audit.py` |
| Pixel cache | `results/phase2/cube/c6_audit/cube_c6_pixels.pt` |
| S1 results | `results/phase2/cube/c6_audit/s1_random_init_10seed.json` |
| S2/S3 results | `results/phase2/cube/c6_audit/s2_s3_results.json` |
| S4/S5 results | `results/phase2/cube/c6_audit/s4_s5_results.json` |
| S6 results | `results/phase2/cube/c6_audit/s6_random_arch_results.json` |
| Summary | `results/phase2/cube/c6_audit/cube_c6_audit_summary.json` |
| Stage 1A Cube reference | `results/phase2/cube/cube_stage1a.json` |
| Memo | `docs/phase2/cube/cube_c6_audit_memo.md` |
