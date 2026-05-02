# Gaussian Latents Are Not Goal Metrics: Large-Displacement Geometry Failure in SIGReg-Regularized JEPA World Models

**Phase 1 Research Proposal**

---

## 1. Background

LeWorldModel (LeWM) is a recent lightweight Joint-Embedding Predictive Architecture (JEPA) world model that learns from raw pixels using only two loss terms: a next-embedding prediction loss and SIGReg, a Gaussian-distribution regularizer. Its central appeal is simplicity: it avoids stop-gradient, exponential moving averages, pretrained encoders, reconstruction losses, and reward supervision, while training a 15M-parameter model on a single GPU with competitive control performance across PushT, TwoRoom, Reacher, and OGBench-Cube.

However, the published evaluations operate at short horizons. The default planning horizon is 5 latent steps with frame-skip 5, corresponding to roughly 25 environment timesteps. The paper's own limitations section explicitly identifies long-horizon planning as an open problem.

This proposal builds on a completed Phase 0 audit. Phase 0 was a pre-registered diagnostic protocol with no method commitment, designed to localize where LeWM's long-horizon failures actually originate. Its findings substantially redirected the project away from our original hypotheses (multi-step rollout drift, closed-loop SIGReg failure, planner-induced exploitation) and toward a different mechanism that we describe below.

## 2. Phase 0 Findings

Phase 0 produced four findings that jointly constrain the Phase 1 direction.

**Finding 1: Long-horizon failure is real but is a proxy for displacement-driven failure.**
On PushT, success rate degrades from 96% at goal offset 25, to 58% at offset 50, 16% at offset 75, and 10% at offset 100. The Spearman correlation between per-pair success count and physical pose distance is rho = -0.751, between success and block displacement is rho = -0.741, and between success and required rotation is rho = -0.627. The mechanism is not horizon length per se but the magnitude of physical displacement and rotation that the goal demands.

**Finding 2: The predictor is not the dominant failure source.**
Three-cost attribution at PushT offset 50 yields correlation of 0.864 between the model's imagined terminal cost and the cost computed from the encoder applied to real simulator observations. The predictor faithfully preserves encoder geometry across rollout horizons. Aggregate latent diagnostics confirm this: temporal straightness (real 0.516, imagined 0.531), effective rank (real 107.3, imagined 106.1), and SIGReg normality statistic (real 0.0016, imagined 0.0017) are essentially identical between encoded and imagined latent batches.

**Finding 3: The planner is not the dominant failure source.**
Of 30 evaluated initial-goal pairs, 23 contained at least one action sequence that succeeded under the fixed candidate set. Late-stage CEM candidates succeeded at 52.7%, compared with 15.7% for random actions. The planner's search behavior is functioning; the model assigns wrong scores to candidates the planner finds.

**Finding 4: Encoder goal geometry fails under large displacement and rotation.**
Per-pair correlation between the encoder-space terminal distance and the privileged physical task cost is mean 0.353 with standard deviation 0.486. On the seven hardest pairs (large displacement, large required rotation), this correlation is -0.046, indicating that latent distance to goal carries essentially no useful ranking information about physical proximity to goal. The aggregate distribution properties enforced by SIGReg remain intact, but the task-relevant geometry in the latent space does not.

**Finding 5 (boundary observation): OGBench-Cube exhibits a different failure mode.**
Cube success rates across offsets 25, 50, 75, 100 are 68%, 50%, 58%, 50%. There is no horizon-dependent monotonic degradation. This is consistent with the published probing results showing LeWM is weak on rotational and dynamic variables in 3D scenes. Cube failures appear to originate from fine-grained 3D visual encoding rather than from the planar large-displacement geometry failure observed on PushT.

## 3. Research Question

Phase 1 investigates a single research question:

**Why does LeWM's encoder produce a latent space in which Euclidean terminal distance to a goal embedding fails to rank candidate states by physical proximity to that goal under large displacement and rotation, and what is the minimum modification that restores this property?**

This question is more specific than the Phase 0 audit question. It commits to the encoder-geometry hypothesis identified by the audit, but it deliberately keeps two sub-questions open until further evidence is collected:

- whether the failure is specific to SIGReg or shared by other latent encoders such as DINOv2;
- whether the failure represents true information loss in the encoder, or merely a poor choice of distance function in latent space.

These two sub-questions structure the Phase 1 experimental tracks below.

## 4. Three Claims to be Tested

The proposal commits to three claims, each tied to a specific experimental track and each falsifiable.

**C1 (Phenomenon).** LeWM's encoder geometry degrades systematically with physical displacement and rotation magnitude, while predictor rollout fidelity and aggregate latent distribution properties remain intact. Phase 0 already provides initial evidence; Phase 1A extends sample size and stratification to bring this to publication-grade evidence.

**C2 (SIGReg specificity, conditional).** This degradation is more pronounced in SIGReg-regularized LeWM than in alternative latent encoders. Phase 1B tests this by applying the same diagnostic protocol to DINOv2-based encoders. C2 is reported with whatever sign the data supports; we do not commit to it succeeding.

**C3 (Mechanism and minimal fix).** The failure can be localized along a ladder: linear metric calibration, nonlinear metric calibration, or representation retraining. Phase 1C identifies which level of the ladder is necessary to restore goal ranking, and Phase 1D applies a minimal fix at that level. The expected outcome is a lightweight, SIGReg-compatible regularizer that does not break LeWM's two-term simplicity.

## 5. Experimental Design

Phase 1 is organized as four tracks (A, B, C, D) with explicit sequential dependencies and decision points.

### 5.1 Track A: Phenomenon Sharpening

Track A extends Phase 0's PushT three-cost attribution from 30 initial-goal pairs to approximately 100 pairs, and from 40 candidate action sequences per pair to approximately 80. The 100 pairs are stratified across displacement and rotation bins to ensure balanced coverage of Easy, Hard, and Impossible categories rather than relying on random sampling.

Outputs are: a per-pair heatmap of encoder-physics correlation as a function of displacement and rotation; scatter plots of latent distance against physical distance, separated by difficulty stratum; and an updated Failure Atlas with representative trajectories.

Track A runs locally on Apple Silicon (MPS). It does not require the 5090 GPU. Estimated wall-clock: 1 to 2 days.

### 5.2 Track B: SIGReg Specificity via DINOv2 Encoder Control

Track B applies the same per-pair three-cost protocol to a DINOv2-based encoder, in two levels of increasing cost.

**Level 1.** The frozen DINOv2 encoder is used to compute a single vector per observation (CLS token or mean-pooled patch tokens), and this vector is used in place of the LeWM encoder for the purpose of computing the encoder-physics correlation. This level uses no DINO-WM predictor and no CEM. It answers a single binary question: does a non-SIGReg encoder, trained on a different objective on a different dataset, exhibit the same geometric failure under large displacement?

**Level 2 (conditional).** If Level 1 results are ambiguous (DINOv2 and LeWM perform similarly), a small projection head is trained on top of frozen DINOv2 features to test whether the relevant geometric information is present in DINOv2 features but not directly accessible in the chosen aggregation. Level 2 is not run if Level 1 already produces a clear directional answer.

**Full DINO-WM planning is explicitly out of scope for Phase 1A and Phase 1B.** It is a conditional Phase 1D experiment, triggered only if Level 1 and Level 2 results jointly require it for the paper.

Track B Level 1 requires the 5090 GPU for DINOv2 forward passes on the full pair set (estimated several hours). Level 2, if needed, requires additional 5090 time on the order of half a day.

### 5.3 Track C: Frozen Metric Calibration Ladder

Track C tests whether the failure observed in Track A reflects (i) a poor choice of Euclidean distance in an otherwise informative latent space, or (ii) genuine information loss in the encoder. The LeWM encoder and predictor are held frozen throughout. Only a goal-cost function on top of the encoder latent space is learned.

The ladder consists of three layers:

**Layer 1 — Linear metric (global Mahalanobis).**
A learned positive-semidefinite matrix M parameterizes the cost C_M(z, z_g) = (z - z_g)^T M (z - z_g). Three parameterizations are tested: diagonal, low-rank, and full PSD. Training is by ranking loss: for triples (z_+, z_-, z_g) where z_+ is physically closer to z_g than z_-, the loss enforces C_M(z_+, z_g) + m < C_M(z_-, z_g) for a margin m.

**Layer 2 — Small nonlinear head.**
A 1-2 layer MLP with hidden dimension 64-128 takes [z, z_g, z - z_g, |z - z_g|] as input and outputs a scalar cost. Same ranking loss objective. This represents a candidate lightweight method.

**Layer 3 — Large nonlinear head (diagnostic upper bound).**
A 3-4 layer MLP with hidden dimension 256-512. Same input and objective. This is explicitly framed in the paper as a diagnostic, not a proposed method. Its purpose is to upper-bound what any nonlinear function on top of the frozen encoder can achieve.

**Held-out evaluation.** All three layers are evaluated on three held-out splits, none of which are random:
- Split 1: train on Easy + Hard pairs, test on Impossible (displacement extrapolation).
- Split 2: train on offset 25 pairs, test on offset 75 pairs (horizon extrapolation).
- Split 3: train on low-rotation pairs, test on high-rotation pairs (rotation extrapolation).

Metric-level evaluation (ranking correlation, AUROC, pairwise accuracy) is run on all three splits. Planning-level evaluation (CEM with the calibrated cost replacing the Euclidean cost) is run only on the most informative stress regime, to control compute.

**Interpretation logic.**
- If Layer 1 (Mahalanobis) succeeds: the failure is purely a metric calibration problem. A lightweight global metric is sufficient.
- If Layer 1 fails but Layer 2 succeeds: the failure requires a lightweight nonlinear metric.
- If Layer 2 fails but Layer 3 succeeds on at least metric-level splits: the encoder retains the information, but it is not easily decodable. This still does not strictly require encoder retraining; it suggests an inference-time module.
- If Layer 3 also fails on held-out splits (especially Split 1 and Split 3): the encoder has lost task-relevant information under large displacement. Encoder retraining is then justified.

Track C runs primarily on CPU and MPS for training the small modules. The 5090 is required only for the planning-level CEM sanity check on the strongest calibrated cost (estimated half a day).

### 5.4 Track D: Representation Retraining (Conditional)

Track D is triggered only if Track C concludes that information loss in the encoder is the operative failure mode. It is not run otherwise.

If triggered, Track D introduces a Goal-Ranking Consistency (GRC) regularizer that supplements the original LeWM training objective:

L = L_pred + lambda_1 * SIGReg(Z) + lambda_2 * L_GRC

where L_GRC is a ranking hinge loss enforcing that for triples (z_+, z_-, z_g) with z_+ physically closer to z_g, the latent Euclidean distance respects the same ordering. Two variants of L_GRC are evaluated:

**GRC-alpha (oracle).** Uses privileged physical state to construct ranking triples. This breaks LeWM's task-agnostic, reward-free design and is explicitly framed in the paper as an oracle upper bound on what representation-level repair can achieve, not as the proposed final method.

**GRC-beta (task-agnostic, PushT-only).** Uses trajectory ordering: for an expert trajectory, a state at later time t+k_2 is treated as physically closer to a future goal at time t+K than a state at earlier time t+k_1 (k_1 < k_2 < K). This proxy is reasonable on PushT, where expert trajectories are approximately monotonic in physical progress toward the goal. It is not used on OGBench-Cube, where multi-stage pick-and-place trajectories are non-monotonic and the proxy would supply incorrect supervision.

**Cube treatment within Track D.** GRC-alpha is evaluated on Cube as a diagnostic counterfactual: if even oracle privileged-state ranking fails to repair Cube performance, this provides strong evidence that Cube failures originate from fine-grained 3D encoding rather than goal-metric geometry, sharpening the paper's scope claim.

Track D is the largest 5090 consumer in Phase 1. Each retraining run is estimated at 2-3 days of 5090 time. Multiple lambda settings, two variants, and held-out evaluation imply the track is only initiated when prior evidence has made it the necessary path.

## 6. Decision Points and Resource Discipline

The 5090 GPU is shared and not always available. Phase 1 is structured around explicit decision points to ensure each 5090 use answers a question that cannot be answered without it.

**DP1 (after Track A).** Are the per-pair correlation statistics from Phase 0 (mean 0.353, std 0.486) reproduced at 100-pair scale? If the standard deviation drops substantially below 0.3 at larger sample size, the heterogeneity that motivates the entire project is weaker than Phase 0 suggested, and the framing must be re-evaluated.

**DP2 (after Track B Level 1).** Three possible outcomes:
- DINOv2 encoder geometry is significantly better than LeWM's: the SIGReg-specificity claim (C2) is supported. Track C proceeds.
- DINOv2 encoder geometry is similar to LeWM's: the failure is not SIGReg-specific. The paper claim narrows to "Euclidean latent goal metrics fail under large displacement" without singling out SIGReg. Track C still proceeds.
- DINOv2 encoder geometry is worse than LeWM's: an unexpected result. Pause Track C; convene to reinterpret.

**DP3 (after Track C ladder).** Determines whether Track D is triggered. If Layer 1 or Layer 2 already restores ranking on all three held-out splits and yields a planning improvement of at least 10 percentage points on the chosen stress regime, Track D is not initiated. If Layer 3 succeeds only with extensive capacity, or fails on Splits 1 and 3, Track D is initiated.

**Resource principle.** Each 5090 session must be tied in advance to one of these decision points. Exploratory 5090 use is not authorized.

## 7. Frozen Exclusions

The following directions are explicitly excluded from Phase 1, based on Phase 0 evidence that contradicts or fails to support them:

- Self-rollout or multi-step rollout training losses, in any form. Phase 0 showed predictor rollout is faithful.
- Closed-loop SIGReg, RDC-SIG, two-sample distribution matching, or any related distributional regularization on imagined latents. Phase 0 showed SIGReg statistics are essentially identical between encoded and imagined latents.
- Drift-Aware CEM, kNN transition support penalty, or other planner-side pessimism methods. Phase 0 showed planner search is not the bottleneck.
- Curvature regularization, residual predictor architectures, or other dynamics-side modifications.
- EMA or stop-gradient training tricks introduced as a method line.
- Magnitude regularization on the calibrated cost. Pure ranking loss is the main objective; cost normalization, if needed, is an implementation-level numerical hygiene step and not a method claim.
- Action-norm or temporal-distance proxies as the main self-supervised signal for GRC-beta. Trajectory ordering on monotonic-progress tasks is the only proxy used.

These exclusions are based on empirical evidence from Phase 0 and are not subject to revision without new contradicting evidence.

## 8. Expected Outcomes and Risk Profile

The expected primary outcome is one of the following three paper framings, determined by which tier of the Track C ladder turns out to be the operative one.

**Outcome A (linear metric sufficient).** The paper argues that SIGReg ensures distributional anti-collapse but does not enforce a calibrated goal metric, and a lightweight global Mahalanobis cost suffices. Method novelty is moderate. Phenomenon and analysis carry the paper.

**Outcome B (nonlinear head needed).** The paper argues that LeWM's encoder retains the relevant information but in a form not accessible via Euclidean distance, and proposes a small inference-time cost head. Method novelty is moderate to strong, depending on whether the head structure is principled.

**Outcome C (representation retraining needed).** The paper argues that SIGReg-induced Gaussianity actively destroys task-relevant geometry under large displacement, and proposes GRC as a SIGReg-compatible repair. Method novelty is strongest. Risk is highest because retraining is required and timeline lengthens.

**Fallback.** If Phase 1 fails to produce a clean method-level result, the Phase 0 Failure Atlas combined with the Track A and Track C diagnostic results is sufficient to support an empirical study submission to TMLR or a workshop venue. This fallback is a deliberate part of the project structure, not an afterthought.

The team explicitly does not subscribe to the position that any specific outcome is preferable for paper novelty reasons. The Track C ladder is designed to honestly distinguish the three cases; the operative case is determined by the data.

## 9. Boundary Conditions and Cube

OGBench-Cube is included in Phase 1 only as a boundary analysis, not as a primary evaluation environment. Three variants of the privileged state cost (cube position only; cube position plus orientation; full benchmark task cost) are reported separately, in light of the published probing results showing LeWM is weaker on rotational and dynamic variables in 3D.

If the proposed method (regardless of which Track C tier) succeeds on PushT but not on Cube, this is reported as a scope limitation: the method addresses planar large-displacement goal geometry, not 3D fine-grained encoding deficiency. This boundary is part of the paper's contribution, not a limitation to hide.

## 10. Communication with the LeWM Team

The LeWM authors will be contacted after Track A produces 100-pair statistics. The contact is brief, includes specific numbers from Phase 0 and Track A, and offers to share the Failure Atlas. The intent is twofold: to disclose the project direction transparently before a paper submission, and to surface any overlap with their own follow-up work (the published limitations section identifies hierarchical world modeling as their direction, which is orthogonal to the encoder geometry direction proposed here).

## 11. Pre-Registered Stop Condition

Phase 1 is terminated as a method paper if any of the following occur:

- DP1 reveals that Phase 0 phenomenon does not replicate at larger sample size.
- DP3 reveals that the entire Track C ladder fails to improve held-out ranking by at least 0.15 in Spearman correlation, AND fails to improve planning success on the stress regime by at least 10 percentage points.
- A method paper from another group preempts the encoder-geometry framing during Phase 1 execution.

In any of these cases, the project pivots to publishing the Failure Atlas plus Track A and Track C diagnostics as an empirical study, without a method contribution.

---

## Appendix A: Summary of Phase 0 Quantitative Findings

| Quantity | Value | Source |
|---|---|---|
| PushT success at offset 25 | 96% (48/50) | Track A baseline |
| PushT success at offset 50 | 58% (29/50) | Track A sweep |
| PushT success at offset 75 | 16% (8/50) | Track A sweep |
| PushT success at offset 100 | 10% (5/50) | Track A sweep |
| corr(C_model, C_real_z) at offset 50 | 0.864 | Three-cost attribution |
| corr(C_real_z, C_real_state), per-pair mean | 0.353 | Three-cost attribution |
| corr(C_real_z, C_real_state), per-pair std | 0.486 | Three-cost attribution |
| corr on Impossible pairs | -0.046 | Per-pair characterization |
| Spearman(success, block displacement) | -0.741 | Per-pair characterization |
| Spearman(success, physical pose distance) | -0.751 | Per-pair characterization |
| Spearman(success, required rotation) | -0.627 | Per-pair characterization |
| Real latent straightness | 0.516 | Aggregate diagnostics |
| Imagined latent straightness | 0.531 | Aggregate diagnostics |
| Real latent effective rank | 107.3 | Aggregate diagnostics |
| Imagined latent effective rank | 106.1 | Aggregate diagnostics |
| Real SIGReg statistic | 0.0016 | Aggregate diagnostics |
| Imagined SIGReg statistic | 0.0017 | Aggregate diagnostics |
| Cube success at offset 25 | 68% | Track A sweep |
| Cube success at offset 50 | 50% | Track A sweep |
| Cube success at offset 75 | 58% | Track A sweep |
| Cube success at offset 100 | 50% | Track A sweep |

## Appendix B: Track-to-Resource Mapping

| Track | Compute target | 5090 use | Estimated time | Trigger |
|---|---|---|---|---|
| A: Phenomenon sharpening | Local MPS / CPU | None | 1-2 days | Default start |
| B Level 1: DINOv2 encoder only | 5090 | Yes, short window | Several hours | After A |
| B Level 2: DINOv2 + small head | 5090 | Yes, half day | Half day | Conditional on B Level 1 ambiguity |
| C Layer 1: Mahalanobis | CPU | None | 1 day | Parallel with A finalization |
| C Layer 2: Small nonlinear head | CPU / MPS | None | 1-2 days | After C Layer 1 |
| C Layer 3: Large nonlinear head | MPS / 5090 | Optional, short | 1-2 days | After C Layer 2 |
| C planning sanity check | 5090 | Yes, half day | Half day | Best calibrated cost only |
| D: GRC retraining | 5090 | Yes, multi-day | 2-3 days per run | Conditional on Track C |
| D: Cube boundary analysis | MPS | None | 1 day | Reporting only |

## Appendix C: Frozen Exclusions Reference

For traceability, the directions excluded from Phase 1 and the Phase 0 evidence that excludes them:

- **Multi-step rollout loss / RDC-SIG / closed-loop SIGReg.** Excluded by aggregate diagnostics: real and imagined latents have nearly identical SIGReg statistic, effective rank, and straightness.
- **Drift-Aware CEM / planner-side pessimism.** Excluded by per-pair analysis: 23 of 30 pairs have at least one successful action in the candidate set; planner search is not the bottleneck.
- **Predictor rollout improvement.** Excluded by three-cost attribution: corr(C_model, C_real_z) = 0.864 indicates the predictor is faithful.
- **Stop-gradient or EMA as method axis.** Out of scope; current evidence does not motivate them.
- **OGBench-Cube as primary environment.** Excluded by sweep results: Cube does not show horizon-dependent degradation; failure mode differs from PushT.
