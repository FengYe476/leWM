# Phase 0 Failure Mode Audit: LeWorldModel Long-Horizon Planning

This repository tracks a focused Phase 0 audit of LeWorldModel (LeWM), a lightweight JEPA world model for goal-conditioned planning from pixels. The project goal is not to improve the model yet, but to localize where long-horizon failures originate once the task moves beyond the near-ceiling PushT baseline regime. The audit starts from a faithful reproduction of the published evaluation pipeline, then applies controlled long-goal stress tests and structured diagnostics to separate failures caused by encoder geometry, predictor rollout quality, planner behavior, and event-localized breakdowns along individual trajectories.

## Research Question

Where does LeWM's long-horizon goal-conditioned planning failure originate?

## Links

- LeWM paper: https://arxiv.org/abs/2603.19312
- Audit repository: https://github.com/FengYe476/leWM
- Upstream LeWM code: https://github.com/lucas-maes/le-wm
- stable-worldmodel: https://github.com/galilai-group/stable-worldmodel
- stable-pretraining: https://github.com/galilai-group/stable-pretraining
- OGBench: https://github.com/seohongpark/ogbench
- LeWM project page: https://le-wm.github.io/
- Hugging Face LeWM collection: https://huggingface.co/collections/quentinll/lewm

## Repository Overview

```text
lewm-failure-audit/
├── scripts/
│   ├── setup_env.sh
│   ├── download_checkpoints.sh
│   ├── verify_install.py
│   ├── verify_checkpoint.py
│   ├── eval_pusht_baseline.py
│   ├── eval_pusht_sweep.py
│   ├── run_pusht_full_eval.sh
│   └── run_pusht_sweep.sh
├── third_party/
│   ├── le-wm/
│   └── stable-pretraining/
├── checkpoints/
│   ├── lewm-pusht/
│   ├── lewm-cube/
│   └── converted/
├── stablewm_cache/
├── patches/
├── docs/
├── results/
├── configs/
└── notebooks/
```

## Setup

### 1. Create the environment

```bash
cd /Users/fengye/Desktop/Project/leWM/lewm-failure-audit
bash scripts/setup_env.sh
```

This creates the `lewm-audit` conda environment and installs:

- `stable-worldmodel==0.0.6`
- `stable-pretraining==0.1.7.dev44+gd8d8f568d.d20260501` in editable mode
- `ogbench==1.2.1`
- PyTorch `2.13.0.dev20260430` on the current Apple Silicon setup

### 2. Download checkpoints

```bash
cd /Users/fengye/Desktop/Project/leWM/lewm-failure-audit
bash scripts/download_checkpoints.sh
```

This fetches the official public LeWM checkpoints:

- `checkpoints/lewm-pusht/weights.pt`
- `checkpoints/lewm-cube/weights.pt`

Each checkpoint file is about 69 MB and includes a matching `config.json`.

### 3. Provide the PushT dataset

The PushT dataset is expected at:

```text
stablewm_cache/pusht_expert_train.h5
```

On the current workstation, the cache contains:

- `pusht_expert_train.h5`
- `pusht_expert_train.h5.zst`

The audit uses the extracted HDF5 file directly. If you are setting up a fresh clone, place the extracted dataset under `stablewm_cache/` before running evaluations.

### 4. Verify the install

```bash
cd /Users/fengye/Desktop/Project/leWM/lewm-failure-audit
conda run -n lewm-audit python scripts/verify_install.py
conda run -n lewm-audit python scripts/verify_checkpoint.py
```

Expected verified components:

- `stable-worldmodel` import
- `stable-pretraining` import
- `ogbench` import
- `torch` import
- `swm/PushT-v1`
- `swm/OGBCube-v0`
- PushT and Cube checkpoint load plus forward pass

### 5. Apple Silicon / MPS note

LeWM requires a local dtype fix for MPS evaluation on this machine. The patch is tracked in:

```text
patches/lewm_mps_fix.patch
```

Setup notes are in [docs/mps_setup.md](/Users/fengye/Desktop/Project/leWM/lewm-failure-audit/docs/mps_setup.md:1). MPS evaluation works from a normal host terminal; the Codex sandbox may not expose Metal devices.

## Reproduce the Baseline

The reproduced PushT baseline uses:

- `goal_offset_steps=25`
- `eval_budget=50`
- CEM with `300` samples, `30` iterations, `30` elites
- planning horizon `5`
- receding horizon `5`
- `action_block=5`

Run the CPU baseline:

```bash
cd /Users/fengye/Desktop/Project/leWM/lewm-failure-audit
conda run -n lewm-audit python scripts/eval_pusht_baseline.py \
    --cache-dir stablewm_cache \
    --results-path results/pusht_baseline_eval.json \
    --num-eval 50 \
    --device cpu
```

Run the MPS baseline:

```bash
cd /Users/fengye/Desktop/Project/leWM/lewm-failure-audit
bash scripts/run_pusht_full_eval.sh
```

Important semantic note:

- `goal_offset_steps` is measured in dataset rows and PushT environment steps.
- It is not a latent-step offset.
- Because `action_block=5`, an offset of `25` corresponds to `5` planner blocks.

## Run the Long-Goal Offset Sweep

The long-horizon stress test sweeps:

- `goal_offset_steps in {25, 50, 75, 100}`
- `eval_budget = 2 x goal_offset_steps`

Run the sweep on MPS:

```bash
cd /Users/fengye/Desktop/Project/leWM/lewm-failure-audit
bash scripts/run_pusht_sweep.sh
```

Equivalent direct command:

```bash
cd /Users/fengye/Desktop/Project/leWM/lewm-failure-audit
conda run -n lewm-audit python scripts/eval_pusht_sweep.py \
    --cache-dir stablewm_cache \
    --offsets 25,50,75,100 \
    --num-eval 50 \
    --device mps
```

Sweep runner features:

- comma-separated `--offsets`
- `--device auto|mps|cpu`
- `--num-eval`
- per-offset JSON outputs in `results/`
- resume support if `results/pusht_sweep_offset{N}.json` already exists
- valid-start-point diagnostics and warnings

## Current Results Summary

### Environment and model facts

- Conda env: `lewm-audit`
- Python: `3.11`
- OS / hardware: macOS 26.3 ARM on Apple M5 Pro
- Dataset: `stablewm_cache/pusht_expert_train.h5`
- Dataset size: 18,685 episodes, 2,336,736 rows
- Mean episode length: `125.06` rows
- Model size: `18.0M` parameters, not `15M`
- Encoder size: `5.5M`
- Predictor stack size: `12.5M`
- Latent embedding dimension: `192`

### Action-space finding

- PushT raw action dimension: `2`
- PushT effective action dimension in the checkpoint: `10`
- This comes from raw action dimension times `action_block=5`

### PushT baseline

- CPU baseline: `98%` success (`49/50`) in [results/pusht_baseline_eval.json](/Users/fengye/Desktop/Project/leWM/lewm-failure-audit/results/pusht_baseline_eval.json:1)
- MPS baseline: `98%` success (`49/50`) in [results/pusht_baseline_eval_mps.json](/Users/fengye/Desktop/Project/leWM/lewm-failure-audit/results/pusht_baseline_eval_mps.json:1)
- Paper reference point: `96%`
- MPS wall-clock for 50 episodes: about `295s`
- First batched CEM solve pays a large one-time shader warmup cost

### Long-goal sweep status

- Offset `25`: `96%` success (`48/50`) in [results/pusht_sweep_offset25.json](/Users/fengye/Desktop/Project/leWM/lewm-failure-audit/results/pusht_sweep_offset25.json:1)
- Offset `50`: `58%` success (`29/50`) in [results/pusht_sweep_offset50.json](/Users/fengye/Desktop/Project/leWM/lewm-failure-audit/results/pusht_sweep_offset50.json:1)
- Offset `75`: pending
- Offset `100`: pending

Current valid starting points by offset:

- `25 -> 1,869,611`
- `50 -> 1,402,587`
- `75 -> 950,309`
- `100 -> 559,843`

The offset-50 result already lands in the target diagnostic band of roughly `40-70%` success.

## Status Tracker

| Day | Status | Planned deliverable |
| --- | --- | --- |
| Day 0 | ✅ Complete | Environment verification, checkpoint verification, MPS setup notes |
| Day 1 | 🟡 In progress | Baseline reproduction, long-goal sweep infrastructure, first sweep outputs |
| Day 2 | ⬜ Pending | Finish PushT offset sweep and summarize offset-dependent failure onset |
| Day 3 | ⬜ Pending | Freeze the diagnostic offset and prepare aggregate trajectory slices |
| Day 4 | ⬜ Pending | Implement aggregate encoder / predictor / planner attribution metrics |
| Day 5 | ⬜ Pending | Run aggregate diagnostics on PushT at the selected stress offset |
| Day 6 | ⬜ Pending | Add per-trajectory logging and event-localized failure traces |
| Day 7 | ⬜ Pending | Build decision-tree labeling pass for Cases A-F |
| Day 8 | ⬜ Pending | Review whether failures cluster in encoder geometry, rollout drift, or planner misspecification |
| Day 9 | ⬜ Pending | Extend the same audit protocol to OGBench-Cube |
| Day 10 | ⬜ Pending | Compare PushT and Cube failure signatures |
| Day 11 | ⬜ Pending | Consolidate frozen exclusions and guard against scope drift |
| Day 12 | ⬜ Pending | Produce final tables, plots, and representative trajectories |
| Day 13 | ⬜ Pending | Draft interpretation of failure origin and residual ambiguity |
| Day 14 | ⬜ Pending | Final Phase 0 writeup and handoff package |

## Additional Docs

- [docs/research_plan.md](/Users/fengye/Desktop/Project/leWM/lewm-failure-audit/docs/research_plan.md:1)
- [docs/progress_log.md](/Users/fengye/Desktop/Project/leWM/lewm-failure-audit/docs/progress_log.md:1)
- [docs/day0_status_table.md](/Users/fengye/Desktop/Project/leWM/lewm-failure-audit/docs/day0_status_table.md:1)
- [docs/mps_setup.md](/Users/fengye/Desktop/Project/leWM/lewm-failure-audit/docs/mps_setup.md:1)
