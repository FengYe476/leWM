# C6 Audit Memo

## 1. Provenance and Config

- Purpose: decide whether the C6 random-init anti-correlation is a real architecture-induced effect, an implementation artifact, or ambiguous.
- Git HEAD at memo write time: `e1f97f303cb4f038cd5fd63e299f7ce491e1caf4`
- Input latent artifact: `results/phase2/p2_0/track_a_latents.pt`
- Track A size: `8000` endpoint records, `100` pairs, `80` actions per pair.
- Latent dimension: `192`
- Top-k overlap values: `5`, `10`, `20`
- False-elite k: `30`
- LeWM config for S1-S3: `checkpoints/lewm-pusht/config.json`
- S1 seeds: `0,1,2,3,4,5,6,7,8,9`
- S2/S3/S6 seed: `0`
- Pixel source: simulator replay. S4-S6 raw pixels have shape `[8000, 224, 224, 3]`.
- C6 audit JSON outputs:
  - `results/phase2/stage1/c6_audit/s1_random_init_10seed.json`
  - `results/phase2/stage1/c6_audit/s2_s3_results.json`
  - `results/phase2/stage1/c6_audit/s4_s5_results.json`
  - `results/phase2/stage1/c6_audit/s6_random_arch_results.json`
- Stage 1A reference JSON: `results/phase2/stage1/stage1a_full.json`

Anchor subsets used in the audit:

| Subset | Definition | Pair count |
|---|---|---:|
| Invisible quadrant | all-fail + strong-rho pairs | 16 |
| Sign-reversal | negative C_real_z vs C_real_state per-pair Spearman pairs | 21 |
| Latent-favorable | D0xR1 + D1xR0 | 12 |
| V1-favorable | D3xR0 + D3xR3 | 13 |
| Ordinary | complement of the four named audit subsets | 47 |

## 2. S1-S6 Results

Reference Stage 1A values:

| Control | Spearman | Pairwise Acc | Per-pair rho | False Elite Rate |
|---|---:|---:|---:|---:|
| C0 trained LeWM | +0.506 | 0.632 | +0.333 | 0.700 |
| C4 Gaussian null | -0.006 +/- 0.012 | n/a | n/a | n/a |
| C5 shuffled latent | +0.004 +/- 0.006 | n/a | n/a | n/a |
| C7_mean DINOv2 mean-pool | +0.286 | n/a | n/a | n/a |

### S1: 10-seed random-init LeWM (ViT-tiny)

Evidence: `s1_random_init_10seed.json`, paths `summary_table` and `aggregate`.

| Seed | Spearman | Pairwise Acc | Per-pair rho | False Elite Rate |
|---:|---:|---:|---:|---:|
| 0 | -0.2005 | 0.4753 | -0.0664 | 0.9043 |
| 1 | -0.2136 | 0.4727 | -0.0754 | 0.9113 |
| 2 | -0.2219 | 0.4670 | -0.0886 | 0.9070 |
| 3 | -0.2121 | 0.4656 | -0.0875 | 0.9117 |
| 4 | -0.1152 | 0.4914 | -0.0278 | 0.8983 |
| 5 | +0.0124 | 0.5178 | +0.0501 | 0.8863 |
| 6 | -0.1483 | 0.4749 | -0.0668 | 0.9043 |
| 7 | -0.2100 | 0.4805 | -0.0555 | 0.9087 |
| 8 | -0.2236 | 0.4835 | -0.0469 | 0.9083 |
| 9 | -0.0285 | 0.5140 | +0.0374 | 0.8857 |
| Aggregate | -0.1561 +/- 0.086 | 0.4843 +/- 0.018 | -0.0427 +/- 0.049 | 0.9026 +/- 0.010 |

S1 aggregate per-subset metrics. Evidence: `s1_random_init_10seed.json`, path `aggregate.anchors.*`.

| Subset | Spearman | Pairwise Acc | Per-pair rho mean | Per-pair rho std | False Elite | Top-k LeWM 5/10/20 | Top-k V1 5/10/20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Invisible quadrant | +0.245 +/- 0.101 | 0.541 +/- 0.024 | +0.115 +/- 0.066 | 0.337 +/- 0.023 | 1.000 +/- 0.000 | 0.071/0.139/0.263 | 0.076/0.164/0.326 |
| Sign-reversal | +0.148 +/- 0.071 | 0.562 +/- 0.012 | +0.172 +/- 0.033 | 0.374 +/- 0.039 | 0.995 +/- 0.002 | 0.039/0.100/0.182 | 0.115/0.182/0.333 |
| Latent-favorable | -0.237 +/- 0.060 | 0.417 +/- 0.030 | -0.207 +/- 0.073 | 0.409 +/- 0.028 | 0.744 +/- 0.023 | 0.087/0.112/0.210 | 0.043/0.075/0.187 |
| V1-favorable | +0.145 +/- 0.093 | 0.561 +/- 0.026 | +0.181 +/- 0.070 | 0.487 +/- 0.048 | 0.959 +/- 0.007 | 0.046/0.150/0.252 | 0.111/0.219/0.327 |
| Ordinary | -0.357 +/- 0.090 | 0.440 +/- 0.025 | -0.167 +/- 0.065 | 0.414 +/- 0.018 | 0.872 +/- 0.014 | 0.052/0.085/0.177 | 0.051/0.096/0.192 |

### S2: pre-projector vs post-projector

Evidence: `s2_s3_results.json`, path `S2`.

| Variant | Spearman | Pairwise Acc | Per-pair rho | False Elite Rate |
|---|---:|---:|---:|---:|
| pre_projector | -0.2009 | 0.4815 | -0.0535 | 0.9060 |
| post_projector | -0.2005 | 0.4753 | -0.0664 | 0.9043 |

S2 conclusion: the projector is not the source. The pre-projector and post-projector variants are effectively identical at the global level.

### S3: eval mode vs train mode

Evidence: `s2_s3_results.json`, path `S3`.

| Variant | Spearman | Pairwise Acc | Per-pair rho | False Elite Rate |
|---|---:|---:|---:|---:|
| eval_mode | -0.2005 | 0.4753 | -0.0664 | 0.9043 |
| train_mode | +0.0254 | 0.5148 | +0.0429 | 0.8553 |

S3 conclusion: the effect is tied to eval-mode normalization behavior. The train-mode run nearly removes the global anti-correlation.

### S4: raw pixel baselines

Evidence: `s4_s5_results.json`, path `S4`.

| Cost signal | Spearman | Pairwise Acc | Per-pair rho | False Elite Rate |
|---|---:|---:|---:|---:|
| raw_pixel_l2 | +0.6967 | 0.7145 | +0.5363 | 0.6897 |
| mean_rgb_diff | +0.7053 | 0.7259 | +0.5576 | 0.6853 |
| edge_density_diff | -0.2698 | 0.4331 | -0.1692 | 0.9107 |
| foreground_mass_diff | -0.3327 | 0.4351 | -0.1631 | 0.9207 |

S4 conclusion: the raw RGB input has strong positive task signal. Some low-level summary statistics invert the signal, but raw pixel L2 and mean RGB difference do not.

### S5: hand-crafted block center distance

Evidence: `s4_s5_results.json`, path `S5.block_center_distance`.

| Cost signal | Spearman | Pairwise Acc | Per-pair rho | False Elite Rate |
|---|---:|---:|---:|---:|
| block_center_distance | +0.4258 | 0.6453 | +0.3135 | 0.7767 |

S5 conclusion: a pixel-only parsed block-center feature is positively aligned with task success. The audit therefore does not require simulator state to recover positive visual task signal.

### S6: random CNN + ResNet18

Evidence: `s6_random_arch_results.json`, paths `small_cnn` and `resnet18`.

| Architecture | Mode | Spearman | Pairwise Acc | Per-pair rho | False Elite Rate |
|---|---|---:|---:|---:|---:|
| small_cnn | eval | -0.1815 | 0.4776 | -0.0534 | 0.9140 |
| small_cnn | train | -0.0521 | 0.4864 | -0.0303 | 0.8747 |
| resnet18 | eval | +0.3043 | 0.5576 | +0.1608 | 0.7743 |
| resnet18 | train | +0.1196 | 0.5295 | +0.0834 | 0.8177 |

S6 conclusion: the eval-mode anti-correlation is not unique to the random LeWM ViT-tiny stack. A shallow random CNN with BatchNorm also inverts the signal. Random ResNet18 does not: in eval mode it preserves a positive task signal.

## 3. Interpretation Table

Binding readout from the Step 1 plan, populated with the observed C6 audit data:

| Condition | Observed? | Evidence | Interpretation |
|---|---|---|---|
| C6 < -0.15 stable across seeds | Yes | S1 aggregate Spearman `-0.156`; 8/10 seeds negative | The single-seed C6 anti-correlation is not just a one-off seed accident. |
| Raw pixel outcome > 0 | Yes | raw_pixel_l2 `+0.697`, mean_rgb_diff `+0.705` | The visual input distribution itself is strongly task-positive. |
| Hand-crafted outcome > 0 | Yes | block_center_distance `+0.426` | Pixel-derived object geometry also carries positive task signal. |
| Pre/post projector switch brings C6 to zero | No | S2 pre `-0.201`, post `-0.200` | The projector head is not the source. |
| Eval/train mode removes the global anti-correlation | Yes | S3 eval `-0.200`, train `+0.025` | The effect is tied to eval-mode normalization behavior. |
| Random CNN also < -0.15 | Yes | small_cnn eval `-0.182` | The phenomenon generalizes to at least one shallow random visual encoder. |
| Random ResNet18 also < -0.15 | No | resnet18 eval `+0.304` | The phenomenon is architecture-dependent, not generic to all random visual encoders. |

The matching interpretation is a hybrid of rows 3 and 6/7 in the plan table:

- Random LeWM in eval mode actively flips a positive pixel signal.
- The same kind of inversion appears in a shallow random CNN with normalization layers.
- The inversion does not appear in random ResNet18; residual structure and/or depth partially preserves the positive visual signal.

## 4. Per-subset Analysis

Headline per-subset Spearman comparison. Evidence: `s1_random_init_10seed.json` path `aggregate.anchors.*`, `s4_s5_results.json` paths `S4.*.metrics.anchors.*` and `S5.block_center_distance.metrics.anchors.*`, and `s6_random_arch_results.json` paths `small_cnn.eval_mode.metrics.anchors.*` and `resnet18.eval_mode.metrics.anchors.*`.

| Subset | S1 aggregate | Raw L2 | Mean RGB | Block center | Small CNN eval | ResNet18 eval |
|---|---:|---:|---:|---:|---:|---:|
| Invisible quadrant | +0.245 +/- 0.101 | +0.498 | +0.505 | +0.385 | +0.384 | +0.158 |
| Sign-reversal | +0.148 +/- 0.071 | +0.371 | +0.344 | -0.022 | +0.234 | +0.168 |
| Latent-favorable | -0.237 +/- 0.060 | +0.632 | +0.678 | +0.492 | -0.255 | +0.199 |
| V1-favorable | +0.145 +/- 0.093 | +0.603 | +0.615 | +0.540 | +0.076 | +0.148 |
| Ordinary | -0.357 +/- 0.090 | +0.789 | +0.804 | +0.526 | -0.451 | +0.352 |

The C6 anti-correlation is not uniform across the anchor subsets. S1 is positive in invisible-quadrant, sign-reversal, and V1-favorable subsets, but strongly negative in latent-favorable and ordinary subsets. The global `-0.156` result is therefore driven primarily by ordinary records (`-0.357`) and latent-favorable records (`-0.237`).

This pattern matters because the raw visual baselines show the opposite sign in those same subsets. Raw L2 and mean RGB are positive in all five subsets, including ordinary (`+0.789`, `+0.804`) and latent-favorable (`+0.632`, `+0.678`). The block-center signal is also positive in ordinary (`+0.526`) and latent-favorable (`+0.492`). The random encoders are not merely exposing a naturally anti-task pixel geometry.

S3 subset behavior supports the eval-mode normalization diagnosis:

| Subset | Eval Spearman | Train Spearman |
|---|---:|---:|
| Invisible quadrant | +0.238 | +0.067 |
| Sign-reversal | +0.061 | +0.201 |
| Latent-favorable | -0.259 | -0.039 |
| V1-favorable | +0.118 | -0.052 |
| Ordinary | -0.387 | -0.018 |

The train-mode run mostly neutralizes the two subsets that drive the global inversion: ordinary moves from `-0.387` to `-0.018`, and latent-favorable moves from `-0.259` to `-0.039`. It does not create a strong positive representation; it mainly removes the eval-mode inversion.

S6 shows the same subset signature in the shallow CNN. Small CNN eval is positive on invisible-quadrant (`+0.384`), sign-reversal (`+0.234`), and V1-favorable (`+0.076`), but negative on latent-favorable (`-0.255`) and ordinary (`-0.451`). ResNet18 eval is positive in all five anchor subsets, including ordinary (`+0.352`) and latent-favorable (`+0.199`).

False-elite rates remain high for the random encoders. S1 aggregate false-elite rate is `0.903`, and small CNN eval is `0.914`, compared with trained LeWM C0 at `0.700`. The invisible-quadrant false-elite rate is saturated at `1.000` for many controls, so it is less diagnostic than Spearman and pairwise accuracy in that subset.

## 5. C6 Verdict

Verdict: **C6-REAL**, with scoped interpretation.

The C6 anti-correlation is real as an eval-mode random visual encoder phenomenon, not a single-seed accident and not a projector-head artifact. The strongest precise statement is:

> Random-init visual encoders with normalization layers in eval mode can invert task-relevant pixel signals on PushT endpoint ranking. Severity depends on architecture depth and structure; residual connections in random ResNet18 partially preserve the signal.

This is not a claim that all random encoders invert the signal. It is also not a claim that the raw visual input distribution is anti-task. S4 and S5 rule that out: raw pixels and a simple pixel-derived block-center feature are both positively aligned with C_real_state.

The S3 train-mode result narrows the scope. The random LeWM eval-mode behavior is deployment-relevant for the Stage 1A C6 control, but train mode nearly removes the inversion. This makes the mechanism more specific: eval-mode normalization behavior in untrained visual stacks can turn a positive visual signal into anti-task geometry.

## 6. Implications for Stage 2 Paper Framing

The Stage 1A C6 result should be framed as stronger than "random initialization is uninformative." The random-init LeWM baseline is worse than random: it starts below the C4/C5 null floors and below zero in aggregate Spearman. Training therefore does not merely add a `+0.506` Spearman signal over a neutral encoder. Relative to S1's random-init aggregate (`-0.156`), trained LeWM C0 (`+0.506`) represents an effective shift of about `+0.662` Spearman.

S4 and S5 make the paper framing cleaner. The pixels already contain strong task-positive geometry, and a simple pixel-derived object center recovers a meaningful positive signal. The learned LeWM representation should therefore be described as overcoming a worse-than-random random-encoder geometry while preserving and task-shaping a signal that is already present in the visual input.

S6 broadens the claim but also constrains it. The inversion is not unique to the LeWM ViT-tiny stack because small CNN eval also goes negative. It is not universal because random ResNet18 stays positive. The right framing is architecture- and mode-dependent, not "random visual encoders are anti-task" in general.

For Stage 2, the useful bridge is:

- C0 vs S1: learned LeWM overcomes an anti-task random starting point.
- C0 vs C2/C3: once the learned latent space exists, much of its endpoint-ranking signal is distributed and robust to random projections.
- C0 vs S4/S5/S6: the learned representation is not merely copying raw pixel distance; it converts a positive but shallow visual signal into a stronger task-specific latent geometry.

This supports the Stage 2 paper theme: planning-compatible geometry should be evaluated both at the representation-starting-point level and at the planning-cost-landscape level. Endpoint ranking alone is not enough, but the C6 audit clarifies that LeWM training creates real representational structure rather than inheriting a benign random visual geometry.
