## 1. Stage 1A Summary Table

| Control | Config | Seeds | Global Spearman (mean±std) | Pairwise Accuracy (mean±std) | Per-pair rho mean | False Elite Rate |
|---|---:|---:|---:|---:|---:|---:|
| C0 | lewm_192 | 1 | 0.506±n/a | 0.632±n/a | 0.333 | 0.700 |
| C1 | orthogonal_192 | 10 | 0.506±0.000 | 0.632±0.000 | 0.333 | 0.700 |
| C2 | gaussian_m=1 | 10 | 0.196±0.082 | 0.554±0.014 | 0.145 | 0.790 |
| C2 | gaussian_m=2 | 10 | 0.235±0.049 | 0.559±0.017 | 0.157 | 0.768 |
| C2 | gaussian_m=4 | 10 | 0.323±0.047 | 0.595±0.007 | 0.251 | 0.743 |
| C2 | gaussian_m=8 | 10 | 0.379±0.035 | 0.603±0.013 | 0.266 | 0.723 |
| C2 | gaussian_m=16 | 10 | 0.407±0.037 | 0.605±0.011 | 0.266 | 0.715 |
| C2 | gaussian_m=32 | 10 | 0.444±0.033 | 0.621±0.011 | 0.309 | 0.706 |
| C2 | gaussian_m=64 | 10 | 0.473±0.026 | 0.626±0.007 | 0.320 | 0.704 |
| C2 | gaussian_m=128 | 10 | 0.477±0.019 | 0.628±0.007 | 0.323 | 0.701 |
| C2 | gaussian_m=192 | 10 | 0.495±0.019 | 0.629±0.007 | 0.326 | 0.701 |
| C3 | coords_m=8 | 10 | 0.367±0.029 | 0.603±0.011 | 0.270 | 0.726 |
| C3 | coords_m=16 | 10 | 0.423±0.032 | 0.615±0.011 | 0.297 | 0.716 |
| C3 | coords_m=32 | 10 | 0.455±0.027 | 0.621±0.013 | 0.309 | 0.709 |
| C3 | coords_m=64 | 10 | 0.487±0.033 | 0.629±0.011 | 0.327 | 0.702 |
| C4 | gaussian_null_192 | 10 | -0.006±0.012 | 0.498±0.003 | -0.005 | 0.838 |
| C5 | independent_row_shuffle | 10 | 0.004±0.006 | 0.500±0.003 | -0.000 | 0.837 |
| C6 | random_init_seed0 | 1 | -0.200±n/a | 0.475±n/a | -0.066 | 0.904 |
| C7_cls | dinov2_cls | 1 | 0.238±n/a | 0.582±n/a | 0.218 | 0.747 |
| C7_mean | dinov2_mean | 1 | 0.286±n/a | 0.595±n/a | 0.253 | 0.750 |

Evidence: [results/phase2/stage1/stage1a_full.json](../../../results/phase2/stage1/stage1a_full.json), path `summary_table`.

## 2. Key Findings

**F1. C4/C5 floors are established.**
C4 Gaussian-null costs produce global Spearman `-0.006±0.012` and pairwise accuracy `0.498±0.003`; C5 independent latent shuffling produces global Spearman `0.004±0.006` and pairwise accuracy `0.500±0.003`. These are the expected near-zero-correlation and chance-pairwise floors, validating the Stage 1A metric framework. Evidence: [stage1a_full.json](../../../results/phase2/stage1/stage1a_full.json), paths `controls.C4.aggregate`, `controls.C5.aggregate`, and `summary_table`.

**F2. C6 random-init LeWM is below the null floor.**
C6 has global Spearman `-0.200`, pairwise accuracy `0.475`, and false-elite rate `0.904`; the random-init encoder produced `[8000, 192]` terminal and goal embeddings through `model.encode(...)["emb"]` with `checkpoint_weights_loaded=false`. The untrained LeWM architecture does not provide goal ranking by itself; learned weights are essential. Evidence: [stage1a_full.json](../../../results/phase2/stage1/stage1a_full.json), paths `controls.C6.metrics`, `controls.C6.config`, and `controls.C6.encode_metadata`.

**F3. C2 `m=192` closely matches C0.**
The same-dimensional Gaussian projection reaches global Spearman `0.495±0.019` versus C0's `0.506`, a gap of `0.011`. This replicates the Track B random-projection observation with 10-seed robustness. Evidence: [stage1a_full.json](../../../results/phase2/stage1/stage1a_full.json), paths `controls.C2.by_dim.192.aggregate.global_spearman` and `controls.C0.metrics.global_spearman`.

**F4. Ranking signal degrades gradually with projection dimension.**
C2 global Spearman is `0.379±0.035` at `m=8`, `0.444±0.033` at `m=32`, and `0.473±0.026` at `m=64`. The ranking signal is distributed across dimensions rather than concentrated in a tiny coordinate set. Evidence: [stage1a_full.json](../../../results/phase2/stage1/stage1a_full.json), paths `controls.C2.by_dim.8.aggregate`, `controls.C2.by_dim.32.aggregate`, and `controls.C2.by_dim.64.aggregate`.

**F5. Coordinate subsets match or slightly exceed Gaussian projections at the same dimension.**
C3 coordinate subsets at `m=64` reach global Spearman `0.487±0.033`, slightly above C2 Gaussian projection at `m=64` (`0.473±0.026`). This supports a fairly uniform distribution of ranking information across learned latent coordinates. Evidence: [stage1a_full.json](../../../results/phase2/stage1/stage1a_full.json), paths `controls.C3.by_dim.64.aggregate` and `controls.C2.by_dim.64.aggregate`.

**F6. DINOv2 underperforms LeWM and sits between low-dimensional LeWM projections.**
C7 mean-pool gives global Spearman `0.286` and C7 CLS gives `0.238`, both below C0's `0.506`. In global Spearman, C7 mean-pool falls between C2 `m=2` (`0.235±0.049`) and C2 `m=4` (`0.323±0.047`). This is only a ranking-position analogy: DINOv2 is a separate 768-d representation space, so it should not be interpreted as a direct effective-dimensionality estimate for LeWM. Evidence: [stage1a_full.json](../../../results/phase2/stage1/stage1a_full.json), paths `controls.C7_mean.metrics.global_spearman`, `controls.C7_cls.metrics.global_spearman`, `controls.C2.by_dim.2.aggregate.global_spearman`, and `controls.C2.by_dim.4.aggregate.global_spearman`.

**F7. The random-projection result has a precise interpretation.**
Stage 1A does not show that random geometry is as good as LeWM: C6 disproves that. It shows that once observations are embedded into LeWM's learned latent space, Euclidean goal ranking is robust to random linear transformations because the ranking signal is distributed across the 192-d representation. Evidence: [stage1a_full.json](../../../results/phase2/stage1/stage1a_full.json), paths `controls.C6.metrics`, `controls.C2.by_dim.192.aggregate`, and `controls.C0.metrics`.

## 3. Anchor Subset Analysis

C2 `m=192` values below are aggregate means over 10 projection seeds; C0, C6, and C7_mean are single-run metrics. Evidence: [stage1a_full.json](../../../results/phase2/stage1/stage1a_full.json), paths `controls.*.metrics.anchors` and `controls.C2.by_dim.192.aggregate.anchors`.

**Invisible quadrant**

| Control | Global Spearman | Pairwise Accuracy | Per-pair rho mean | False Elite Rate |
|---|---:|---:|---:|---:|
| C0 | 0.462 | 0.689 | 0.509 | 1.000 |
| C2 m=192 | 0.408 | 0.672 | 0.457 | 1.000 |
| C6 | 0.238 | 0.528 | 0.078 | 1.000 |
| C7_mean | 0.370 | 0.606 | 0.297 | 1.000 |

**Sign-reversal**

| Control | Global Spearman | Pairwise Accuracy | Per-pair rho mean | False Elite Rate |
|---|---:|---:|---:|---:|
| C0 | -0.034 | 0.340 | -0.448 | 0.984 |
| C2 m=192 | 0.019 | 0.367 | -0.379 | 0.984 |
| C6 | 0.061 | 0.553 | 0.144 | 0.995 |
| C7_mean | 0.255 | 0.493 | -0.041 | 0.986 |

The C0 pairwise accuracy of `0.340` is below chance because these pairs were selected for negative encoder-physics alignment; LeWM's latent cost tends to invert the physical ranking in this cluster.

**Latent-favorable**

| Control | Global Spearman | Pairwise Accuracy | Per-pair rho mean | False Elite Rate |
|---|---:|---:|---:|---:|
| C0 | 0.519 | 0.672 | 0.452 | 0.364 |
| C2 m=192 | 0.499 | 0.661 | 0.421 | 0.367 |
| C6 | -0.259 | 0.391 | -0.281 | 0.764 |
| C7_mean | 0.119 | 0.629 | 0.327 | 0.417 |

## 4. Preliminary Decision Gate Assessment

The pre-registered Strong B gate requires: (a) `m<=8` random projection matches LeWM within 5pp on Stage 1B planning and within `0.05` Spearman on Stage 1A; (b) C6 matches LeWM within `0.05` Spearman while exceeding C5/C4 floors; and (c) seed robustness. Stage 1B planning data are not yet available, so only the Stage 1A parts can be assessed here. Evidence: [stage1a_full.json](../../../results/phase2/stage1/stage1a_full.json), paths `summary_table`, `controls.C2.by_dim.*.aggregate`, and `controls.C6.metrics`.

Criterion (a) is not met for Strong B on Stage 1A ranking. C2 `m=8` trails C0 by `0.127` Spearman (`0.506 - 0.379`), far outside the `0.05` threshold. C2 `m=32` is marginally outside with gap `0.062`; C2 `m=64` is inside with gap `0.034`.

Criterion (b) is definitively not met. C6 Spearman is `-0.200`, not within `0.05` of C0's `0.506`; the absolute gap is about `0.71`. C6 also differs from C4/C5 floors in the wrong direction: it is anti-correlated, not merely uninformative.

Criterion (c) is met for the same-dimensional projection: C2 `m=192` has seed std `0.019`, and its mean Spearman remains close to C0.

Preliminary verdict: Strong B is ruled out by C6. The interesting Stage 1A result is not "random geometry matches LeWM"; it is "random linear transformations of LeWM's learned latent space preserve most endpoint ranking." This is a statement about the geometry of the learned space, not about random spaces being equally good.

Implication for Stage 1B: CEM planning with projected costs remains valuable. If C2 `m=32` or `m=64` projections preserve planning success within 5pp of C0, that would support a Medium B finding about the effective planning dimensionality of LeWM's latent space.

## 5. Revised Interpretation

Stage 1A reframes the original question, "does LeWM's ranking exceed random geometry baselines?", into three separate comparisons:

- vs C6 random-init encoder: yes, dramatically. LeWM learned essential goal-ranking structure.
- vs C2 random projections of LeWM's own space: no, random projections preserve most of the ranking signal once the learned representation already exists.
- vs C7 DINOv2: yes, LeWM outperforms a strong generic encoder on this task-specific endpoint-ranking objective.

The productive framing is now: what is the effective dimensionality of LeWM's goal-ranking subspace, and does this low effective dimensionality explain the CEM cost-landscape failures identified in Phase 2? Stage 1A alone cannot determine whether the effective dimensionality finding has practical consequences for planning. Stage 1B is required to test whether dimensionally-reduced projections preserve CEM planning dynamics, not just endpoint ranking.

Evidence: [stage1a_full.json](../../../results/phase2/stage1/stage1a_full.json), paths `controls.C0.metrics`, `controls.C2.by_dim.*.aggregate`, `controls.C6.metrics`, and `controls.C7_mean.metrics`.

## 6. Open Questions for Stage 1B

- Do C2 `m=32` and `m=64` projected costs preserve CEM success within 5pp of C0?
- Does low-dimensional ranking preservation survive iterative CEM optimization, or only endpoint re-ranking?
- Which projection dimensions preserve top-k elite sets enough for planning?
- Do projected costs change false-elite selection in invisible-quadrant and latent-favorable pairs?
- Is planning degradation monotonic with projection dimension?
- Are failures driven by loss of ranking signal or by altered local cost smoothness/compression?
