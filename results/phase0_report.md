# Phase 0 Failure Mode Audit: Where Does LeWorldModel's Long-Horizon Planning Fail?

## 1. Executive Summary

LeWorldModel's long-horizon failure in PushT originates from encoder goal geometry, not predictor rollout drift or planner weakness. The SIGReg-regularized encoder produces latent distances and rankings that decouple from physical task distances at large block displacements and rotations. The predictor faithfully preserves this flawed encoder geometry: imagined terminal latent costs correlate strongly with real encoded terminal observations, and aggregate latent diagnostics remain nearly identical between real and imagined trajectories. The CEM planner is also not the primary bottleneck because most diagnostic initial-goal pairs contain at least one successful action, and converged CEM candidates outperform random actions. Cross-validation on OGBench-Cube reveals a different failure pattern: Cube starts from a lower baseline but remains roughly flat across offsets, indicating horizon-independent 3D visual encoding difficulty rather than displacement-dependent long-horizon breakdown. The binding Phase 0 classification is therefore a Case B/E hybrid for PushT: encoder goal geometry fails under large physical displacements with event-localized heterogeneity, while Cube is a separate baseline representation limitation.

## 2. Background and Motivation

LeWorldModel (LeWM) is a lightweight JEPA-style world model for goal-conditioned planning from pixels. The model learns an encoder that maps observations into a latent space and a predictor that rolls those latents forward under candidate action sequences. At evaluation time, a CEM planner searches over action blocks and minimizes terminal latent distance to a visual goal. This makes LeWM attractive as a compact planning model: it avoids reconstructing pixels, uses a small encoder-predictor stack, and can plan directly in representation space.

The paper reports strong short-horizon performance, including a near-ceiling PushT result around the standard offset-25 setting. But the paper also motivates the question this audit targets: what happens when the goal is farther away? A failure at longer offsets could arise from several distinct places. The encoder might not preserve task-relevant geometry. The predictor might accumulate rollout error. The planner might fail to search a useful action sequence. Or the failure might only occur around specific events, such as contact, rotation, or large displacement.

Those mechanisms imply different next steps. If the planner is the bottleneck, Phase 1 should change search, horizon, or action parameterization. If predictor drift dominates, Phase 1 should focus on rollout objectives. If encoder geometry is already misleading, planner and predictor changes can spend compute optimizing the wrong objective. The purpose of Phase 0 was therefore diagnostic rather than interventionist: reproduce the published path, stress the horizon, isolate the component responsible for failure, and avoid committing prematurely to a method.

## 3. Experimental Setup

### Environments

The audit used two environments.

| Environment | Role in audit | Success criterion | Dataset |
|---|---|---|---|
| PushT | Primary long-horizon stress environment | `pos_diff < 20` and `angle_diff < pi/9` | `stablewm_cache/pusht_expert_train.h5` |
| OGBench-Cube | Cross-environment validation | cube position within `0.04m` of target | `stablewm_cache/ogbench/cube_single_expert.h5` |

PushT is a 2D pushing task with a visually simple scene and clear physical displacement and rotation variables. It is useful for asking whether the latent goal geometry remains aligned with physical block pose as the goal moves farther away. OGBench-Cube is a visually richer 3D manipulation environment. It is useful as a secondary check because it stresses visual representation capacity differently from PushT.

The Cube dataset contains `10,000` fixed-length episodes of `201` rows each, for `2,010,000` total rows. PushT contains `18,685` episodes and `2,336,736` rows, with a mean episode length of `125.06` rows.

### Model and Planner

The converted LeWM checkpoints use an approximately `18.0M` parameter model with a `192`-dimensional latent. The encoder is a ViT-Tiny stack of about `5.5M` parameters, and the predictor is a ViT-S stack of about `12.5M` parameters.

The planning configuration matched the reproduced stable-worldmodel evaluation path:

| Setting | Value |
|---|---:|
| CEM samples | 300 |
| CEM iterations | 30 |
| CEM elites | 30 |
| Planning horizon | 5 |
| Receding horizon | 5 |
| Action block | 5 |
| PushT raw action dim | 2 |
| PushT effective action dim | 10 |
| Cube raw action dim | 5 |
| Cube effective action dim | 25 |

The effective action dimension equals raw action dimension times `action_block`. `goal_offset_steps` is measured in environment and dataset rows, not latent planner steps. Therefore offset `25` corresponds to five planner blocks with `action_block=5`.

### Evaluation Protocol

Phase 0 used four complementary diagnostics.

The first diagnostic was an offset sweep over `goal_offset_steps = {25, 50, 75, 100}` with `eval_budget = 2 * offset`. This tested whether success degraded as the goal moved farther away. Each offset used `50` evaluation episodes.

The second diagnostic was a three-cost attribution study on PushT at offset `50`, the first stress point where performance fell into a useful diagnostic band. For `30` initial-goal pairs and `40` fixed action sequences per pair, the audit measured:

| Cost | Meaning |
|---|---|
| `C_model` | terminal latent cost from imagined predictor rollout |
| `C_real_z` | terminal latent cost after executing the action and encoding the real observation |
| `C_real_state` | physical terminal-state cost in task state space |

The third diagnostic was per-pair failure characterization. Initial-goal pairs were grouped by how many of the `40` fixed actions succeeded: Easy (`>25/40`), Hard (`1-25/40`), and Impossible (`0/40`). Physical displacement, rotation, and encoder-to-physics correlation were compared across these groups.

The fourth diagnostic was aggregate latent analysis. The audit compared real encoded trajectories and imagined predicted trajectories using temporal straightness, effective rank, covariance spectrum, SIGReg-style normality, and horizon-dependent prediction error.

## 4. Results

### 4.1 Long-Horizon Stress Test

The reproduced PushT baseline stayed near the paper reference point at the standard short offset. The MPS baseline at offset `25` reached `98%` (`49/50`), while the sweep run reached `96%` (`48/50`), matching the paper's `96%` reference within finite-sample variation.

The PushT offset sweep then revealed a steep horizon-dependent failure:

| Offset | Budget | Success rate | Episode count | Result file |
|---:|---:|---:|---:|---|
| 25 | 50 | 96% | 48/50 | `results/pusht_sweep_offset25.json` |
| 50 | 100 | 58% | 29/50 | `results/pusht_sweep_offset50.json` |
| 75 | 150 | 16% | 8/50 | `results/pusht_sweep_offset75.json` |
| 100 | 200 | 10% | 5/50 | `results/pusht_sweep_offset100.json` |

This pattern localized the useful diagnostic region. Offset `25` is too easy because the model is near ceiling. Offsets `75` and `100` are severe stress tests where most runs fail. Offset `50` is the best first attribution setting because its `58%` success rate lies in the intended `40-70%` band: enough failures to analyze, but enough successes to compare against.

OGBench-Cube behaved differently:

| Offset | Budget | Success rate | Episode count | Result file |
|---:|---:|---:|---:|---|
| 25 | 50 | 68% | 34/50 | `results/cube_sweep_offset25.json` |
| 50 | 100 | 50% | 25/50 | `results/cube_sweep_offset50.json` |
| 75 | 150 | 58% | 29/50 | `results/cube_sweep_offset75.json` |
| 100 | 200 | 50% | 25/50 | `results/cube_sweep_offset100.json` |

Cube starts lower than PushT but does not degrade with offset. Its success rate remains between `50%` and `68%`, and offset `100` is essentially tied with offset `50`. The reproduced Cube baseline at offset `25` was `66%` (`33/50`), below the paper's `74%` reference but close to the sweep offset-25 result.

This contrast is important. PushT shows a classic long-horizon stress signature: near-ceiling short-offset performance followed by collapse under larger displacement. Cube shows a baseline representation difficulty signature: lower performance from the start, but no systematic horizon dependence.

### 4.2 Three-Cost Attribution

The three-cost study tested whether the PushT offset-50 failures were caused by predictor rollout drift, planner weakness, or encoder geometry. The global rank correlations were:

| Correlation | Value | Interpretation |
|---|---:|---|
| `corr(C_model, C_real_z)` | 0.864 | Predictor faithfully preserves encoder geometry |
| `corr(C_real_z, C_real_state)` | 0.669 global | Encoder-to-physics alignment is only moderate globally |
| `corr(C_model, C_real_state)` | 0.596 | Model cost is less aligned with true physical cost |
| Per-pair `corr(C_real_z, C_real_state)` mean | `0.353 +/- 0.486` | Encoder geometry is highly pair-dependent |

The strongest result is `corr(C_model, C_real_z) = 0.864`. Imagined terminal costs track real encoded terminal observations well. That makes predictor rollout drift an unlikely primary explanation for the failure: the predictor is preserving the representation it was trained to preserve.

The weaker link is between encoded terminal distance and physical terminal-state distance. The global `C_real_z` to `C_real_state` correlation is moderate, but the per-pair mean is low and highly variable. Some pairs have sensible latent-to-physics ranking. Others have near-zero or negative ranking, meaning the encoded goal distance can prefer physically worse states.

Planner diagnostics also argue against planner-dominated failure. In the `30` initial-goal pairs, `23/30` had at least one successful action among the fixed candidate set. This means the action space and execution path often contain a successful candidate. The planner's converged `CEM_late` candidates succeeded `52.7%` of the time, compared with `15.7%` for random candidates. CEM is not perfect, but it is doing useful search rather than behaving randomly.

The action-source analysis therefore supports a representation-side failure. Dataset actions, smooth random actions, early CEM candidates, and late CEM candidates were all evaluated under the same three costs. The useful contrast is that CEM improves success substantially over random search while still failing on pairs whose latent rankings disagree with physical progress. The planner can exploit the model objective when the objective is aligned; when the latent geometry misorders the physical goal relation, planner quality cannot fully rescue the trajectory.

The supporting figures are:

| Figure | What it shows |
|---|---|
| `results/figures/scatter_cmodel_vs_crealz.png` | High model-to-real-latent agreement |
| `results/figures/scatter_crealz_vs_crealstate.png` | Weaker real-latent to real-state alignment |
| `results/figures/scatter_cmodel_vs_crealstate.png` | Model cost to physical cost mismatch |
| `results/figures/correlation_by_pair.png` | Large per-pair variation |

### 4.3 Per-Pair Failure Characterization

The per-pair analysis split offset-50 PushT pairs into Easy, Hard, and Impossible groups by successful actions out of `40`.

| Category | Count | Success definition | Mean block displacement | Mean rotation required | Mean encoder corr |
|---|---:|---|---:|---:|---:|
| Easy | 8 | `>25/40` successes | 2.6 px | 0.082 rad | 0.408 |
| Hard | 15 | `1-25/40` successes | 66.0 px | 0.809 rad | 0.510 |
| Impossible | 7 | `0/40` successes | 123.3 px | 1.145 rad | -0.046 |

The categories reveal the physical structure of the failure. Easy pairs require almost no block displacement and little rotation. Hard pairs require moderate displacement and rotation. Impossible pairs require very large displacement and larger rotation, and their mean encoder-to-physics correlation is slightly negative.

Physical features strongly predict success count:

| Feature vs. success count | Spearman rho |
|---|---:|
| Physical pose distance | -0.751 |
| Block displacement | -0.741 |
| Required rotation | -0.627 |

These correlations show that the failure is not random. It tracks exactly the variables a goal-conditioned pushing representation should preserve: how far the block must move and how much it must rotate. Latent endpoint distances compress these large physical changes: Easy pairs average about `2.6` px displacement and latent distance `12.5`, while Impossible pairs average about `123.3` px displacement but only latent distance `19.4`. The latent space grows only weakly as physical task difficulty explodes.

The failure atlas captures three PushT motifs:

| Atlas page | Purpose |
|---|---|
| `results/failure_atlas/01_large_displacement_encoder_geometry_failure.md` | Large block displacement breaks latent-to-physical ranking |
| `results/failure_atlas/02_rotation_dependent_encoding_failure.md` | Large orientation changes can be underweighted or misordered |
| `results/failure_atlas/03_easy_baseline_control.md` | Easy near-goal pairs confirm the pipeline and planner can succeed |

### 4.4 Aggregate Latent Diagnostics

Aggregate latent diagnostics asked whether the predictor changes the global geometry of the latent space. It does not. Real encoder latents and imagined predictor latents have nearly identical summary structure:

| Diagnostic | Real encoder latents | Imagined predictor latents | Interpretation |
|---|---:|---:|---|
| Temporal straightness | 0.516 | 0.531 | Nearly identical |
| Effective rank | 107.3 | 106.1 | Nearly identical |
| SIGReg normality | 0.0016 | 0.0017 | Nearly identical |
| Step-10 prediction error L2 | - | 5.83 | Error grows, but structure remains |

The covariance spectra are also nearly identical, as shown in `results/figures/covariance_spectrum.png`. Horizon metrics and per-step error are plotted in `results/figures/metrics_vs_horizon.png` and `results/figures/per_step_error.png`.

This resolves an important ambiguity. A long-horizon planning failure could occur because predicted latents progressively leave the real latent manifold, collapse in rank, become anisotropic, or lose SIGReg-like normality. None of those aggregate pathologies appear. The predictor does accumulate stepwise error, with step-10 mean latent L2 error `5.83`, but it preserves the global geometry of the encoder space. The problem is that the preserved geometry is not sufficiently aligned with task-relevant physical geometry for large PushT displacements.

Put differently: the predictor is faithful to the wrong map. It is not inventing a new distorted latent world at rollout time. It is carrying forward the encoder's own metric, and that metric can decouple from physical goal progress.

### 4.5 Cross-Environment Comparison

PushT and Cube now form the central cross-environment contrast:

| Environment | Offset pattern | Failure interpretation |
|---|---|---|
| PushT | `96 -> 58 -> 16 -> 10%` | Genuine long-horizon encoder-geometry failure under large displacement |
| OGBench-Cube | `68 -> 50 -> 58 -> 50%` | Horizon-independent baseline 3D encoding difficulty |

Cube validates the PushT interpretation by showing a different signature. If the entire LeWM stack had a generic inability to plan at long offsets, Cube should degrade as offset increases. It does not. If CEM horizon or predictor rollout length were the universal bottleneck, offset `100` should be dramatically worse than offset `25`. It is not.

Instead, Cube starts lower and stays flat. That pattern is more consistent with baseline representation capacity in a visually complex 3D scene. The paper's probing results point in the same direction: block quaternion, block yaw, and end-effector yaw probing accuracy are poor on Cube in Table 4. The Cube encoder appears to struggle with task-relevant 3D orientation variables before the long-horizon question even becomes decisive.

The new atlas page `results/failure_atlas/04_cube_baseline_encoding_difficulty.md` records this as a separate phenomenon. It is not the PushT Case B/E failure. It is a baseline representation limitation that strengthens the causal claim: PushT's collapse is specific to displacement-dependent latent geometry, not a generic property of all LeWM evaluations.

## 5. Decision Tree Classification

The Phase 0 decision tree considered six broad explanations.

| Case | Question | Evidence | Outcome |
|---|---|---|---|
| Case A | Is there no meaningful long-horizon failure? | PushT falls from `96%` to `10%` | Excluded |
| Case B | Is encoder goal geometry the main failure source? | Physical distance and latent cost decouple at large displacement | Supported for PushT |
| Case C | Is predictor rollout drift the main failure source? | `corr(C_model, C_real_z) = 0.864`; aggregate geometry preserved | Excluded as primary cause |
| Case D | Is planner search the main failure source? | `23/30` pairs have a successful candidate; CEM_late beats random | Excluded as primary cause |
| Case E | Is the failure heterogeneous or event-localized? | Impossible pairs cluster around displacement and rotation motifs | Supported as refinement |
| Case F | Is this a generic cross-environment model weakness? | Cube is flat rather than horizon-degrading | Excluded |

The binding label is Case B/E hybrid for PushT. Case B captures the representation-side failure: the encoder's latent goal geometry becomes unreliable under large physical displacements. Case E captures the fact that this is not uniform global collapse. Failures are pair-dependent and event-localized, especially around large block translation and rotation.

Cube is a separate baseline representation limitation. It is not Case B/E because it does not show displacement-dependent or horizon-dependent collapse. Its lower success appears to come from visual encoding difficulty in 3D scenes rather than a long-horizon planning failure.

## 6. Discussion

The central lesson is that a SIGReg-style isotropic Gaussian prior does not guarantee task-metric alignment. A representation can have healthy aggregate statistics while still misordering the distances that matter for control. In this audit, real and imagined latents have similar straightness, rank, covariance spectra, and normality. Those are useful sanity checks, but they do not ensure that latent distance to a goal is monotonic with physical progress toward that goal.

This matters because LeWM plans by optimizing a latent cost. If the encoder maps physically different block poses to distances that compress, saturate, or mis-rank large displacements, then the planner can faithfully optimize a misleading objective. The predictor can also be excellent by its own criterion while preserving a flawed geometry. That is exactly what the PushT evidence suggests: the predictor tracks real encoded latents, and CEM improves over random, yet selected actions can still be poor under physical state cost.

The Cube result connects this audit to the paper's own probing observations. The paper reports that Cube probing is difficult for orientation-related variables such as block quaternion, block yaw, and end-effector yaw. That aligns with the flat Cube sweep: Cube does not collapse as the offset increases, but its baseline success is already limited. The difficulty appears to be extracting and preserving 3D state variables from the visual scene with a compact ViT-Tiny encoder, not handling longer offsets per se.

There are several implications for JEPA planning cost functions. First, representation learning objectives that produce useful predictive latents may still need explicit metric structure for planning. Second, terminal latent L2 distance is a strong assumption: it treats the encoder's geometry as the task geometry. Third, aggregate latent health metrics are insufficient as planning diagnostics. A latent space can be globally well behaved while locally or conditionally misaligned with physical success.

The audit has limitations. Most headline success rates use `50` evaluation episodes and a single seed, so small differences such as Cube `50%` versus `58%` should not be overinterpreted. The three-cost attribution is PushT-specific and uses `30` pairs with `40` actions per pair rather than a full exhaustive policy analysis. Physical state costs are diagnostic proxies, not a replacement for the environment's binary success criterion. Cube did not receive the same three-cost attribution pass, so the Cube interpretation relies on the sweep pattern, baseline reproduction, and the paper's probing evidence. These limitations do not overturn the main conclusion, but they bound how strongly Phase 0 should generalize.

## 7. Phase 1 Recommendations

Phase 1 should pivot to representation and SIGReg geometry research. The evidence does not warrant planner redesign or rollout-loss work as the first intervention, because the planner and predictor are not the primary failure sources in the PushT diagnostic regime.

Promising directions include:

| Direction | Rationale |
|---|---|
| Metric-aware regularization | Encourage latent distances or rankings to preserve physical pose structure under large displacement |
| Task-conditioned latent costs | Avoid assuming one global latent L2 metric is the right planning cost for all states |
| Multi-scale or hierarchical planning | Separate coarse displacement progress from fine contact and rotation correction |
| Improved 3D encoder capacity for Cube | Address horizon-independent Cube limitations through stronger visual/orientation encoding |

The cleanest first intervention is to keep the predictor and CEM planner fixed and modify only the representation objective or latent cost. That preserves attribution: if PushT offset-50 and high-offset performance improve, the improvement can be tied to representation geometry rather than search or rollout changes.

Recommended diagnostics for Phase 1 are displacement-binned rank correlations, rotation-binned rank correlations, controlled pose-interpolation probes, and repeat three-cost attribution after any representation change. Offset `50` should remain the main PushT stress setting, while offsets `75` and `100` should be retained as out-of-distribution stress checks.

## 8. Artifacts and Reproducibility

### Primary Result Artifacts

| Artifact | Contents |
|---|---|
| `results/decision_memo.md` | Binding Phase 0 classification and recommendations |
| `results/three_cost_offset50.json` | Raw PushT three-cost attribution records |
| `results/three_cost_analysis.json` | Three-cost analysis summary |
| `results/per_pair_analysis.json` | Per-pair Easy/Hard/Impossible characterization |
| `results/aggregate_latent_diagnostics.json` | Aggregate real-vs-imagined latent diagnostics |
| `results/cube_baseline_eval_mps.json` | Cube baseline reproduction |
| `results/cube_sweep_offset25.json` | Cube offset-25 sweep result |
| `results/cube_sweep_offset50.json` | Cube offset-50 sweep result |
| `results/cube_sweep_offset75.json` | Cube offset-75 sweep result |
| `results/cube_sweep_offset100.json` | Cube offset-100 sweep result |

### Figures

| Figure | Use |
|---|---|
| `results/figures/scatter_cmodel_vs_crealz.png` | Predictor-to-real-latent agreement |
| `results/figures/scatter_crealz_vs_crealstate.png` | Encoder-to-physics alignment |
| `results/figures/scatter_cmodel_vs_crealstate.png` | Model cost versus physical cost |
| `results/figures/correlation_by_pair.png` | Per-pair correlation heterogeneity |
| `results/figures/latent_vs_physical_distance.png` | Latent compression of physical distances |
| `results/figures/covariance_spectrum.png` | Real vs imagined covariance spectrum |
| `results/figures/metrics_vs_horizon.png` | Horizon-dependent latent metrics |
| `results/figures/per_step_error.png` | Prediction error over rollout steps |

### Failure Atlas

| Page | Description |
|---|---|
| `results/failure_atlas/01_large_displacement_encoder_geometry_failure.md` | PushT large-displacement failure motif |
| `results/failure_atlas/02_rotation_dependent_encoding_failure.md` | PushT rotation-sensitive failure motif |
| `results/failure_atlas/03_easy_baseline_control.md` | PushT easy-pair control regime |
| `results/failure_atlas/04_cube_baseline_encoding_difficulty.md` | Cube horizon-independent baseline encoding difficulty |

### Scripts

| Script | Purpose |
|---|---|
| `scripts/eval_pusht_baseline.py` | Reproduce PushT baseline |
| `scripts/eval_pusht_sweep.py` | Run PushT offset sweep |
| `scripts/analyze_three_cost.py` | Analyze PushT three-cost attribution |
| `scripts/analyze_per_pair.py` | Analyze PushT pair-level failure structure |
| `scripts/aggregate_latent_diagnostics.py` | Compare real and imagined latent trajectory geometry |
| `scripts/eval_cube_baseline.py` | Reproduce OGBench-Cube baseline |
| `scripts/eval_cube_sweep.py` | Run OGBench-Cube offset sweep |

### Reproduction Commands

PushT baseline:

```bash
conda run -n lewm-audit python scripts/eval_pusht_baseline.py \
    --cache-dir stablewm_cache \
    --results-path results/pusht_baseline_eval.json \
    --num-eval 50 \
    --device cpu
```

PushT offset sweep:

```bash
bash scripts/run_pusht_sweep.sh
```

OGBench-Cube baseline:

```bash
conda run -n lewm-audit python scripts/eval_cube_baseline.py \
    --cache-dir stablewm_cache \
    --results-path results/cube_baseline_eval_mps.json \
    --num-eval 50 \
    --device mps
```

OGBench-Cube offset sweep:

```bash
bash scripts/run_cube_sweep.sh
```

Aggregate latent diagnostics:

```bash
conda run -n lewm-audit python scripts/aggregate_latent_diagnostics.py
```

The full Phase 0 trail is documented in `docs/progress_log.md`, with setup notes in `docs/mps_setup.md` and the original plan in `docs/research_plan.md`.
