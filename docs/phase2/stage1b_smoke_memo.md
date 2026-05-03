# Stage 1B-Smoke Memo

## 1. Provenance and Config

- Purpose: test whether dimensional projections of LeWM predictor latents preserve CEM planning success, and whether endpoint ranking predicts planning success.
- Full sweep JSON: `results/phase2/stage1/stage1b_smoke.json`
- Sanity JSON: `results/phase2/stage1/stage1b_sanity.json`
- C6 audit memo: `docs/phase2/c6_audit_memo.md`
- Stage 1A reference JSON: `results/phase2/stage1/stage1a_full.json`
- Full sweep git commit recorded in JSON: `7848d5b00f28534155909c9fd956b5a235d3fa27`
- Sanity git commit recorded in JSON: `dac7a06ec626ce125701c5b3f24b40a8cebe79a6`
- C6 verdict from Step 1: **C6-REAL**
- Pair selection: `63` deduplicated pairs from invisible quadrant, sign-reversal, latent-favorable, V1-favorable, plus a deterministic ordinary sample.
- Dimensions tested: `m = 8, 32, 64, 192`
- Projection seeds: `0, 1, 2`
- Projection matrix: Gaussian `P_m` with shape `[192, m]`, entries scaled by `1/sqrt(m)`, fixed per `(m, projection_seed)` across all pairs and CEM iterations.
- CEM config: `300` samples, `30` iterations, `30` elites, planning horizon `5`, action block `5`.
- Efficient scoring design: one simulator-scored default LeWM final 300-candidate pool per pair, plus one projected rank-1 simulator rollout per `(pair, m, projection_seed)`.
- Expected simulator rollouts in full sweep: `63 * 300 + 63 * 4 * 3 = 19,656`.

Reference values:

| Reference | Value |
|---|---:|
| Track A CEM_late success | 0.336 |
| Full-smoke selected-pair default rank-1 success | 0.302 |
| m=192 projected CEM success | 0.344 |
| Stage 1A C0 endpoint Spearman | 0.506 |
| Stage 1A C2 m=64 endpoint Spearman | 0.473 +/- 0.026 |
| Stage 1A C2 m=32 endpoint Spearman | 0.444 +/- 0.033 |

The selected-pair default rank-1 success in `stage1b_smoke.json` is `0.302`; the broader Track A CEM_late reference is `0.336`. The decision matrix comparison uses the user-facing Track A reference and the m=192 smoke result, which agree within noise.

## 2. Sanity Check

Sanity passed. Evidence: `results/phase2/stage1/stage1b_sanity.json`.

| Check | Result | Pass? |
|---|---:|---|
| Baseline success | 5/5 | yes |
| Projected m=192 success | 5/5 | yes |
| Success delta | 0.000 | yes |
| False-elite rate | 0.000 | yes |
| Mean elite-std ratio | 0.972 | yes |
| Mean blocked action L2 | 8.842 | yes |
| Relaxed action L2 threshold | 15.0 | yes |

One sanity pair had blocked action L2 `27.287`, but the mean stayed below the relaxed `15.0` threshold and all five projected rank-1 rollouts succeeded.

## 3. Headline Results

Overall full-sweep metrics by projection dimension:

| Dim | N | Success | Spearman | Pairwise | False elite | Action L2 | Top-k LeWM 5/10/30 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| m=8 | 189 | 0.2063 | 0.0801 | 0.5296 | 0.6894 | 14.8979 | 0.087/0.119/0.235 |
| m=32 | 189 | 0.2804 | 0.1111 | 0.5436 | 0.6857 | 12.2091 | 0.251/0.294/0.424 |
| m=64 | 189 | 0.2963 | 0.0405 | 0.5165 | 0.6894 | 11.1518 | 0.343/0.379/0.498 |
| m=192 | 189 | 0.3439 | 0.0674 | 0.5260 | 0.6889 | 10.1191 | 0.476/0.549/0.642 |

Planning success by dimension and subset:

| Dim | Invisible | Sign-reversal | Latent-fav | V1-fav | Ordinary |
|---:|---:|---:|---:|---:|---:|
| m=8 | 0.021 | 0.032 | 0.444 | 0.103 | 0.567 |
| m=32 | 0.000 | 0.063 | 0.556 | 0.154 | 0.767 |
| m=64 | 0.042 | 0.048 | 0.583 | 0.128 | 0.867 |
| m=192 | 0.021 | 0.079 | 0.694 | 0.154 | 0.967 |

The main success trend is monotonic with dimension at the aggregate level: `20.6%`, `28.0%`, `29.6%`, `34.4%`. The gain is driven primarily by ordinary and latent-favorable pairs. Invisible-quadrant and V1-favorable pairs remain difficult.

## 4. Landscape Diagnostics

Each cell reports projected final top-30 elite cost std / projected final candidate dynamic range.

| Dim | Invisible | Sign-reversal | Latent-fav | V1-fav | Ordinary |
|---:|---:|---:|---:|---:|---:|
| m=8 | 0.610 / 96.7 | 0.520 / 79.6 | 0.506 / 88.2 | 0.741 / 121.0 | 0.431 / 87.9 |
| m=32 | 0.904 / 110.5 | 0.650 / 74.1 | 0.726 / 98.6 | 0.916 / 100.1 | 0.322 / 61.4 |
| m=64 | 0.741 / 103.6 | 0.708 / 80.8 | 0.694 / 95.8 | 0.780 / 89.5 | 0.374 / 68.2 |
| m=192 | 0.836 / 102.5 | 0.849 / 86.2 | 0.640 / 100.5 | 0.775 / 96.3 | 0.354 / 76.8 |

There is no P2-0-style landscape collapse. Top-30 elite std remains nonzero across all dimensions and subsets, and dynamic range stays large. False-elite rate is also stable across dimensions: `0.6894`, `0.6857`, `0.6894`, `0.6889`.

Top-k overlap with LeWM increases with dimension:

| Dim | LeWM top-5 | LeWM top-10 | LeWM top-30 | V1 top-5 | V1 top-10 | V1 top-30 |
|---:|---:|---:|---:|---:|---:|---:|
| m=8 | 0.087 | 0.119 | 0.235 | 0.023 | 0.045 | 0.110 |
| m=32 | 0.251 | 0.294 | 0.424 | 0.040 | 0.060 | 0.097 |
| m=64 | 0.343 | 0.379 | 0.498 | 0.035 | 0.043 | 0.077 |
| m=192 | 0.476 | 0.549 | 0.642 | 0.040 | 0.046 | 0.075 |

The m=64 LeWM top-30 overlap is `0.498`, while success is within the 10pp decision threshold. V1 overlap stays low, which is consistent with the known gap between LeWM's latent planning objective and V1 oracle elites.

## 5. Decoupling Analysis

The central finding is endpoint-planning decoupling. Stage 1A C2 m=64 endpoint Spearman was `0.473 +/- 0.026`, but Stage 1B m=64 planning-pool Spearman is only `0.0405`. Stage 1A C2 m=32 endpoint Spearman was `0.444 +/- 0.033`, but Stage 1B m=32 planning-pool Spearman is `0.1111`.

This is a massive gap. Endpoint ranking preservation on the fixed Track A endpoint artifact does not predict ranking over the final CEM candidate pool. The CEM loop reshapes the candidate distribution over 30 iterations; by the final iteration, candidates are clustered around the search mean rather than sampled broadly from the original action distribution. Spearman over this clustered pool can be near zero while planning success is still meaningful, because local cost differences inside the collapsed search neighborhood are much smaller and more path-dependent than fixed endpoint ranking differences.

The table below is the scatter-plot data requested in the Step 2 plan: each row is one `(m, subset)` cell with planning success and final-pool endpoint Spearman.

| Dim | Subset | Success | Endpoint Spearman | Pairwise | False elite |
|---:|---|---:|---:|---:|---:|
| m=8 | Invisible | 0.021 | 0.038 | 0.513 | 0.995 |
| m=8 | Sign-reversal | 0.032 | 0.078 | 0.528 | 0.990 |
| m=8 | Latent-fav | 0.444 | 0.133 | 0.547 | 0.293 |
| m=8 | V1-fav | 0.103 | 0.050 | 0.521 | 0.915 |
| m=8 | Ordinary | 0.567 | 0.165 | 0.562 | 0.003 |
| m=32 | Invisible | 0.000 | 0.138 | 0.557 | 0.993 |
| m=32 | Sign-reversal | 0.063 | -0.005 | 0.499 | 0.990 |
| m=32 | Latent-fav | 0.556 | 0.179 | 0.569 | 0.289 |
| m=32 | V1-fav | 0.154 | 0.139 | 0.555 | 0.903 |
| m=32 | Ordinary | 0.767 | 0.240 | 0.593 | 0.000 |
| m=64 | Invisible | 0.042 | 0.010 | 0.508 | 0.994 |
| m=64 | Sign-reversal | 0.048 | -0.048 | 0.483 | 0.989 |
| m=64 | Latent-fav | 0.583 | 0.207 | 0.577 | 0.281 |
| m=64 | V1-fav | 0.128 | -0.029 | 0.491 | 0.928 |
| m=64 | Ordinary | 0.867 | 0.194 | 0.575 | 0.000 |
| m=192 | Invisible | 0.021 | 0.038 | 0.514 | 0.994 |
| m=192 | Sign-reversal | 0.079 | -0.018 | 0.494 | 0.990 |
| m=192 | Latent-fav | 0.694 | 0.169 | 0.562 | 0.284 |
| m=192 | V1-fav | 0.154 | 0.046 | 0.518 | 0.922 |
| m=192 | Ordinary | 0.967 | 0.214 | 0.583 | 0.000 |

The subset table makes the decoupling visible. Ordinary m=64 has only `0.194` endpoint Spearman but `0.867` planning success. Latent-favorable m=64 has `0.207` endpoint Spearman and `0.583` success. Invisible quadrant m=32 has higher endpoint Spearman (`0.138`) than sign-reversal m=192 (`-0.018`), but both have very low success. Local CEM geometry, subset difficulty, and elite false-positive structure dominate the final-pool Spearman alone.

## 6. Per-subset Narrative

Baseline selected-pair success by subset:

| Subset | Default selected-pair success |
|---|---:|
| Invisible | 0/16 |
| Sign-reversal | 1/21 |
| Latent-favorable | 9/12 |
| V1-favorable | 0/13 |
| Ordinary | 10/10 |

**Invisible quadrant.** Projections do not meaningfully rescue this subset. Default CEM succeeds on `0/16` invisible pairs. Projected CEM success is `2.1%`, `0.0%`, `4.2%`, and `2.1%` across m=8/32/64/192. At the pair level, m=64 rescues only `2/16` default-fail invisible pairs at least once across three projection seeds.

**Sign-reversal.** Projection does not invert the failure mode into a high-success regime. Default CEM succeeds on `1/21` sign-reversal pairs. Projection success stays between `3.2%` and `7.9%`, and m=64 has negative final-pool Spearman (`-0.048`). The projected objective slightly changes which rare failures are rescued, but does not repair the cluster.

**Latent-favorable.** Projection preserves much of LeWM's advantage in D0xR1/D1xR0, though not all of it. Default selected-pair success is `9/12 = 75%`. Projected success rises with dimension from `44.4%` to `69.4%`, with m=64 at `58.3%`. False-elite rate is low relative to hard subsets (`0.281` at m=64), and endpoint Spearman is the highest among the named subsets at m=64 (`0.207`).

**V1-favorable.** Projection does not preserve a high-success regime here because the selected-pair LeWM baseline itself is poor on these pairs: default selected-pair success is `0/13`. Projected success remains low, from `10.3%` to `15.4%`, and false-elite rates stay above `0.90`. This subset remains a planning failure region despite being V1-favorable by oracle-cell definition.

**Ordinary.** Ordinary pairs behave as the sanity-check baseline. Default selected-pair success is `10/10`, and projected success increases with dimension from `56.7%` to `96.7`. m=64 reaches `86.7%`, with false-elite rate `0.000`. The projected cost changes the final action, but it mostly preserves success once enough dimensions are retained.

## 7. Verdict and Joint Decision Matrix

Binding decision: **Continue B**.

The applicable row is the first row of the joint decision matrix:

| Condition | Required | Observed | Pass? |
|---|---:|---:|---|
| C6 verdict | C6-REAL | C6-REAL | yes |
| m=64 success within 10pp of LeWM | within 10pp | m=64 `29.6%` vs m=192 `34.4%`, gap `4.8pp`; vs Track A `33.6%`, gap `4.0pp` | yes |
| Elite std/dynamic range not collapsed | non-collapsed | m=64 elite std mean `0.694-0.780` on named subsets, dynamic range `68.2-103.6`; false elite stable at `0.689` | yes |
| Top-k overlap with LeWM remains high | high enough for smoke | m=64 LeWM overlap `0.343/0.379/0.498` for k=5/10/30 | yes |

This matches:

> **Continue B**: full Stage 1B + start Cube cross-environment (Path A) + consider Subspace-CEM (Path B). Top conference push.

The key nuance is that Continue B does not mean endpoint ranking predicts planning ranking. It means dimensional projection preserves enough CEM planning success at m=64, without cost-landscape collapse, even though final-pool Spearman is weak. That combination is stronger and more interesting than a simple endpoint-dimensionality story.

## 8. Implications for Next Steps

Per the decision matrix, the next path is:

- Run the full Stage 1B sweep on all 100 pairs with the full dimension ladder.
- Start Cube cross-environment setup.
- Consider Subspace-CEM as the method-development branch.
- Keep the endpoint-planning decoupling result central: Stage 1A endpoint Spearman survives projection, but Stage 1B final-pool Spearman does not explain planning success.

This is the top-conference push path. The story is now joint:

- Step 1: C6-REAL shows LeWM training overcomes an actively bad random visual starting point.
- Stage 1A: learned LeWM endpoint geometry is distributed enough to survive random projection.
- Stage 1B-Smoke: projected latent costs preserve planning success at m=64 within the decision threshold, even though endpoint ranking on CEM's final clustered candidate pool is weak.

The paper framing should therefore move from "endpoint rank preservation" to "planning-compatible geometry under dimensional constraint." Endpoint metrics remain useful diagnostics, but the decisive object is the CEM cost landscape induced by projected predictor latents.
