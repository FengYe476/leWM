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

## Update policy

This file is intended to be append-only. Future entries should:

- start with an ISO date
- include the project day
- record both actions taken and concrete findings
- distinguish completed evidence from planned next steps
