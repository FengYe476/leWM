# Progress Log

## 2026-04-30 — Day 0: Environment and artifact verification

### Repository and environment bring-up

- Created the audit workspace structure and helper scripts for setup, checkpoint download, install verification, and checkpoint verification.
- Built the `lewm-audit` conda environment on macOS 26.3 ARM with Python `3.11`.
- Installed and verified `stable-worldmodel==0.0.6`.
- Installed and verified editable `stable-pretraining==0.1.7.dev44+gd8d8f568d.d20260501`.
- Installed and verified `ogbench==1.2.1`.
- Verified local PyTorch runtime as `2.13.0.dev20260430`.

### Environment verification

- Confirmed `swm/PushT-v1` instantiates successfully.
- Confirmed `swm/OGBCube-v0` instantiates successfully.
- Confirmed MuJoCo-backed Cube evaluation works with `MUJOCO_GL=disabled` for verification.

### Checkpoint verification

- Downloaded the official public LeWM PushT and Cube checkpoints from Hugging Face.
- Confirmed both checkpoint weight files are about `69 MB`.
- Converted both checkpoints into `stable_worldmodel.policy.AutoCostModel`-compatible object checkpoints.
- Verified encoder forward pass shape `(1, 192)` for both checkpoints.
- Verified predictor forward pass shape `(1, 3, 192)` for both checkpoints.

### Key findings

- The deployed model is about `18.0M` parameters, not `15M`.
- Encoder size is about `5.5M` parameters.
- Predictor stack size is about `12.5M` parameters.
- Latent embedding dimension is `192`.
- PushT raw action dimension is `2`, but the effective action dimension used by the checkpoint is `10`.
- Cube raw action dimension is `5`, but the effective action dimension used by the checkpoint is `25`.
- The effective action dimension equals raw action dimension times planner `action_block`.

### MPS finding

- Apple Silicon MPS works for evaluation from a normal terminal.
- A dtype patch is required because the upstream path forwards `float64` tensors that MPS rejects.
- The reusable patch was captured in `patches/lewm_mps_fix.patch`.

## 2026-05-01 — Day 1: Baseline reproduction and long-goal sweep

### PushT baseline reproduction

- Reproduced the published PushT evaluation using `stable_worldmodel.World.evaluate_from_dataset()`.
- Matched the LeWM evaluation path with converted checkpoints, the same CEM settings, and the same PushT environment.
- Confirmed the baseline semantics of `goal_offset_steps`.

### Baseline results

- CPU baseline result: `98%` success (`49/50`) with `goal_offset_steps=25` and `eval_budget=50`.
- MPS baseline result: `98%` success (`49/50`) with the same evaluation settings.
- The reproduced baseline is consistent with the paper's reported `96%`, within normal finite-sample variance.

### Goal-offset semantics

- Confirmed that `goal_offset_steps` is measured in dataset rows and PushT environment steps.
- Confirmed that it is not measured in latent planner steps.
- With `action_block=5`, an offset of `25` corresponds to `5` planner blocks, not `25` planner blocks.

### MPS evaluation notes

- Batched MPS evaluation ran at about `295s` for 50 episodes.
- Steady-state time is about `5.9s` per episode in the batched setup.
- The first CEM solve pays a large one-time shader warmup cost on MPS.

### Long-goal sweep infrastructure

- Added `scripts/eval_pusht_sweep.py`.
- Added `scripts/run_pusht_sweep.sh`.
- Implemented resumable per-offset JSON outputs.
- Added valid-start-point accounting and warnings for large offsets.

### Offset sweep status

- Offset `25`: completed, `96%` success (`48/50`) in the sweep runner.
- Offset `50`: completed, `58%` success (`29/50`) in the sweep runner.
- Offset `75`: pending at the time of this log entry.
- Offset `100`: pending at the time of this log entry.

### Dataset facts used in the sweep

- PushT dataset rows: `2,336,736`.
- PushT dataset episodes: `18,685`.
- Mean episode length: `125.06` rows.
- Median episode length: `123` rows.
- Minimum episode length: `49` rows.
- Maximum episode length: `246` rows.

### Valid starting points by offset

- Offset `25`: `1,869,611`.
- Offset `50`: `1,402,587`.
- Offset `75`: `950,309`.
- Offset `100`: `559,843`.

### Current interpretation

- Offset `25` remains near-ceiling and is too easy for detailed failure attribution.
- Offset `50` already lands in the intended diagnostic regime at about `58%` success.
- Pending confirmation from offsets `75` and `100`, offset `50` is the leading candidate for the first aggregate and per-trajectory failure analysis pass.

## 2026-05-01 — Day 1: PushT long-goal sweep completion

- Long-goal offset sweep complete on PushT.
- Results: offset `25` -> `96%`, offset `50` -> `58%`, offset `75` -> `16%`, offset `100` -> `10%`.
- Offset `50` selected as primary diagnostic setting because `58%` falls in the `40-70%` target range.
- Total sweep time was about `1 hour` on MPS.

## 2026-05-01 — Day 1: Three-cost attribution analysis

- Completed the offset-`50` three-cost attribution run on PushT.
- Evaluated `30` initial-goal pairs with `40` fixed action sequences per pair, for `1200` total action records.
- Compared `C_model`, `C_real_z`, and `C_real_state` across data actions, smooth random actions, early CEM candidates, and converged CEM candidates.
- Found strong model-to-real-latent agreement: `corr(C_model, C_real_z) = 0.864`, indicating that predictor rollout is not the dominant failure source.
- Found globally moderate encoder-to-physics agreement: `corr(C_real_z, C_real_state) = 0.669`.
- Found weak and highly variable per-pair encoder-to-physics agreement: mean per-pair correlation `0.353 +/- 0.486`.
- Found that `23/30` pairs had at least one successful action, which argues against planner/action-parameterization failure as the primary explanation.
- Found that `CEM_late` actions succeeded at `52.7%`, compared with `15.7%` for random actions.
- Saved detailed output to `results/three_cost_analysis.json` and figures under `results/figures/`.

## 2026-05-01 — Day 1: Per-pair failure characterization

- Completed per-pair failure characterization at offset `50`.
- Categorized pairs into `Easy`, `Hard`, and `Impossible` using the number of successful actions out of `40`.
- Category counts: `8` Easy, `15` Hard, `7` Impossible.
- Easy pairs averaged `2.6` px block displacement and `0.082` rad required rotation.
- Hard pairs averaged `66.0` px block displacement and `0.809` rad required rotation.
- Impossible pairs averaged `123.3` px block displacement and `1.145` rad required rotation.
- Impossible pairs had mean encoder-to-physics correlation `-0.046`, indicating near-zero or inverted latent-to-physical ranking on the hardest configurations.
- Physical pose distance correlated strongly with success count: Spearman `rho = -0.751`.
- Block displacement correlated strongly with success count: Spearman `rho = -0.741`.
- Required rotation correlated strongly with success count: Spearman `rho = -0.627`.
- Confirmed PushT state layout as `[agent_x, agent_y, block_x, block_y, block_angle, agent_vx, agent_vy]`; the goal is represented by a second future state row, not by fields inside the same 7-D state.
- Saved detailed output to `results/per_pair_analysis.json`.

## 2026-05-01 — Day 1: Binding Phase 0 decision memo and failure atlas

- Wrote the binding decision memo at `results/decision_memo.md`.
- Classified the primary failure as a `Case B/E` hybrid under the current analysis protocol: encoder goal geometry fails under large physical displacements, with event-localized characteristics.
- Bound the next action to representation and SIGReg-style geometry research rather than planner redesign or rollout-loss work.
- Started the failure atlas in `results/failure_atlas/`.
- Added atlas pages for large-displacement encoder geometry failure, rotation-dependent encoding failure, and easy baseline controls.
- Representative failure pairs include pair `21` for large displacement, pair `2` for rotation-sensitive impossible behavior, and pairs `25`, `26`, `28`, and `29` as easy controls.
- The current Phase 0 interpretation is that LeWM's predictor is faithful and CEM improves over random search, but encoder goal geometry compresses or misorders physically meaningful differences when displacement and rotation grow.

## 2026-05-01 — Day 1: Aggregate latent diagnostics and Cube pipeline

- Aggregate latent diagnostics complete on PushT: predictor faithfully preserves encoder geometry.
- Temporal straightness, effective rank, covariance spectrum, and SIGReg-style normality are all near-identical between real encoder latents and imagined predictor latents.
- Prediction error grows with horizon, with step-10 mean latent L2 error `5.83`, but aggregate latent structure is maintained.
- Saved aggregate diagnostics to `results/aggregate_latent_diagnostics.json`.
- Saved covariance spectrum and horizon-metric plots to `results/figures/`.
- OGBench-Cube evaluation pipeline created in `scripts/eval_cube_baseline.py`.
- Cube dataset downloaded and extracted to `stablewm_cache/ogbench/cube_single_expert.h5`; extracted HDF5 size is `95GB`.
- OGBench-Cube smoke test completed with `2/3` success on CPU.
- Full OGBench-Cube baseline evaluation is pending.

## 2026-05-01 — Day 1: Cube baseline and aggregate diagnostics finalization

- OGBench-Cube baseline reproduced: `66%` success (`33/50`) at offset=`25`. Paper reports `74%`.
- Aggregate latent diagnostics complete on PushT: predictor faithfully preserves encoder geometry. Straightness real=`0.516` imagined=`0.531`, effective rank real=`107.3` imagined=`106.1`, SIGReg normality real=`0.0016` imagined=`0.0017`. Step-10 prediction error L2=`5.83`.
- Added resumable OGBench-Cube offset-sweep infrastructure for offsets `25`, `50`, `75`, and `100`, with budget set to `2x` offset and per-offset JSON outputs.

## 2026-05-01 — Day 1: OGBench-Cube offset sweep completion

- OGBench-Cube offset sweep complete. Results: offset 25→68%, 50→50%, 75→58%, 100→50%. Unlike PushT's steep degradation (96→58→16→10%), Cube shows no horizon-dependent failure — success rate is essentially flat across offsets. This indicates Cube failures originate from 3D visual encoding difficulty rather than long-horizon planning breakdown, consistent with the paper's own observation that encoder training is more challenging in visually complex 3D environments.

## 2026-05-01 — Day 1: Phase 0 finalization

- Phase 0 report finalized. All experimental data collected and analyzed across PushT and OGBench-Cube. Classification: Case B/E hybrid. Decision memo and failure atlas complete.

## Update policy

This file is intended to be append-only. Future entries should:

- start with an ISO date
- include the project day
- record both actions taken and concrete findings
- distinguish completed evidence from planned next steps

## 2026-05-01 — Day 15: Track A consolidation

- Accepted V1/V2 oracle ablation as the final Phase 1 Track A experimental finding.
- Wrote the canonical Track A summary at `docs/phase1/track_a_summary.md`.
- Wrote Phase 0 claim revisions at `docs/phase1/phase0_revisions.md` without modifying frozen Phase 0 source documents.
- Added new Failure Atlas pages `05` through `09` for predictor sharpness, cost-criterion misalignment, V2 indicator degeneracy, R3 apparent infeasibility revision, and the D3 alignment paradox.
- Updated the README conclusion note, Phase 1 Track A conclusion, status table, and Additional Docs links.
- Net revision: Phase 0's predictor-fidelity and per-pair heterogeneity findings hold, while the planner-not-bottleneck and large-displacement-encoder-collapse framings are revised or falsified by Track A.

## 2026-05-01 — Day 16: Track A cleanup

- Replaced the F6 per-trajectory anchor: pair `6` was retained as `*_replaced` and pair `8` became the D0xR1 visual anchor.
- Reran only the four pair `8` per-trajectory rollouts and rendered its three replacement figures.
- Removed gitignored smoke-run links from the Track A summary evidence inventory.
- Corrected the Track A mechanism-map label for D2xR0 from `both` to `cost misalignment`.
- Updated Failure Atlas page `09` with the pair-weighted D3 latent CEM_late convention and value.
- Added D3xR2 representative pair `92` to the D3 row alignment paradox examples.
- No Phase 0 source files or earlier atlas pages were changed.
