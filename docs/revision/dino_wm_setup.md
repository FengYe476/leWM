# DINO-WM PushT Pool Audit Setup

This memo documents the prepared DINO-WM integration for the 30-pair PushT endpoint-pool audit. It does not require modifying the cloned DINO-WM repository.

## Repository Snapshot

Clone location:

```bash
cd /Users/fengye/Desktop/Project/leWM
git clone https://github.com/gaoyuezhou/dino_wm.git dino_wm
cd dino_wm
git rev-parse --short HEAD
```

Inspected commit: `0a9492f` (`add checkpoints`).

Top-level structure:

- `conf/`: Hydra configs for training/planning/envs/encoders/planners.
- `plan.py`: planning entrypoint, checkpoint/model loading, `PlanWorkspace`.
- `planning/`: CEM, MPC wrapper, objective functions, evaluator.
- `models/`: DINOv2 encoder, ViT predictor, visual world model wrapper.
- `datasets/`: PushT dataset loader and normalization constants.
- `env/`: gym env registration, vector envs, PushT simulator wrapper.
- `environment.yaml`: DINO-WM conda environment.

## Key Code Locations

- PushT config: `conf/env/pusht.yaml`
  - `with_velocity: true`, `with_target: true`.
  - Dataset path uses `${DATASET_DIR}/pusht_noise`.
- PushT planning config: `conf/plan_pusht.yaml`
  - `ckpt_base_path: ./checkpoints`
  - `model_name: pusht`
  - `planner.sub_planner`: CEM with 300 samples, 30 elites, 30 iterations, horizon 5.
- CEM implementation: `planning/cem.py`
  - Samples `torch.randn(num_samples, horizon, action_dim) * sigma + mu`.
  - Forces candidate 0 to current `mu`.
  - Updates `mu/std` from top-30 elites.
  - Returns final post-update `mu`, not necessarily a sampled candidate.
- Cost/objective: `planning/objectives.py`
  - `mode=last`: terminal visual MSE plus `alpha *` terminal proprio MSE.
- Model loading: `plan.py`
  - `load_ckpt()`, `load_model()`, `planning_main()`.
  - Checkpoint expected at `${ckpt_base_path}/outputs/${model_name}/checkpoints/model_${model_epoch}.pth`.
- Encoder: `models/dino.py`
  - `DinoV2Encoder`, default `dinov2_vits14`, `x_norm_patchtokens`.
  - PushT visual latent is 196 patch tokens x 384 dims.
- Predictor: `models/vit.py`
  - `ViTPredictor`, causal attention mask.
  - Contains the hardcoded CUDA mask that needs a device-agnostic patch or runtime monkey patch.
- Rollout interface: `models/visual_world_model.py`
  - `VWorldModel.rollout(obs_0, act)` returns predicted observation embeddings.
- PushT simulator: `env/pusht/pusht_wrapper.py`, `env/pusht/pusht_env.py`
  - `PushTWrapper.prepare()` and `rollout()` expose controlled state resets.

## Integration Script

Prepared script:

```bash
/Users/fengye/Desktop/Project/leWM/lewm-failure-audit/scripts/revision/dino_wm_pool_audit.py
```

What it does:

1. Imports DINO-WM from `--dino-repo` at runtime.
2. Loads the DINO-WM PushT checkpoint from `--ckpt-base-path/outputs/pusht`.
3. Selects the same 30 Track A pairs used by the MPPI comparison:
   - 8 invisible-quadrant
   - 12 ordinary
   - 5 latent-favorable
   - 5 V1-favorable
4. Reads Track A start/goal states directly from the HDF5 file in `track_a_pairs.json`, avoiding `stable_worldmodel` imports.
5. Runs an instrumented DINO-WM CEM loop for each pair.
6. Saves the final iteration's 300-candidate pool with:
   - `z_pred.visual`, `z_pred.proprio`
   - `z_goal.visual`, `z_goal.proprio`
   - normalized blocked actions `(300, 5, 10)`
   - raw simulator actions `(300, 25, 2)`
   - DINO-WM `default_costs` / `C_model`
   - simulator-scored `c_real_state`, V1 hinge, success, terminal states
7. Scores real terminal DINO-WM endpoint embeddings for endpoint diagnostics.
8. Writes per-pair and aggregate `Rendpoint`, `Rpool_Cmodel`, and `Delta_CEM`.

Default output paths:

```bash
results/revision/dino_wm_pool_audit.json
results/revision/dino_wm_pusht_pools/pair_XXX_seed_Y.pt
```

Pair list smoke command, safe to run before checkpoint setup:

```bash
cd /Users/fengye/Desktop/Project/leWM/lewm-failure-audit
python scripts/revision/dino_wm_pool_audit.py --list-pairs
```

## Environment Setup

On the GPU server:

```bash
cd /path/to/leWM
git clone https://github.com/gaoyuezhou/dino_wm.git dino_wm
cd dino_wm
conda env create -f environment.yaml
conda activate dino_wm
export SDL_VIDEODRIVER=dummy
export PYTHONPATH=/path/to/leWM/dino_wm:/path/to/leWM/lewm-failure-audit:${PYTHONPATH}
```

The audit script reads Track A states directly with `h5py`, which is already in DINO-WM's environment file. It does not need the DINO-WM PushT dataset for the 30-pair audit, but it does need the leWM Track A HDF5 path referenced by `results/phase1/track_a_pairs.json`.

Verify CUDA and DINO-WM imports:

```bash
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
import env
import models.vit
print("dino_wm imports ok")
PY
```

## Checkpoint Download

DINO-WM README points to OSF node `bmw48`; the PushT checkpoint is inside `/checkpoints/outputs.zip`.

Download command:

```bash
cd /path/to/leWM/dino_wm
mkdir -p checkpoints
curl -L 'https://osf.io/download/xvzs4/' -o checkpoints/outputs.zip
```

Known OSF metadata:

- File: `/checkpoints/outputs.zip`
- Size: `953204628` bytes
- MD5: `5f21e8306df578d292f92d0028c4cc31`
- SHA256: `425b0d8c2a4194d3ec996b553575f69409aa5714176b063d8401789db9179482`

Verify and extract:

```bash
cd /path/to/leWM/dino_wm
echo '425b0d8c2a4194d3ec996b553575f69409aa5714176b063d8401789db9179482  checkpoints/outputs.zip' | sha256sum -c -
unzip -q checkpoints/outputs.zip -d checkpoints
test -f checkpoints/outputs/pusht/hydra.yaml
test -f checkpoints/outputs/pusht/checkpoints/model_latest.pth
```

On macOS, replace `sha256sum -c -` with:

```bash
shasum -a 256 checkpoints/outputs.zip
```

## Device Compatibility Patch

DINO-WM contains one relevant PushT planning blocker:

```python
# dino_wm/models/vit.py
self.bias = generate_mask_matrix(NUM_PATCHES, NUM_FRAMES).to('cuda')
```

Problems:

- Fails on CPU-only smoke tests.
- Can place the attention mask on `cuda:0` even when running on another CUDA device.
- The tensor is not registered as a buffer, so `model.to(device)` will not move it if instantiated from config.

The audit script applies a runtime-only monkey patch by default:

```bash
--patch-device-compat
```

It replaces the attention init so the causal mask is a non-persistent registered buffer on the requested device. No DINO-WM source file is edited.

If you decide to patch DINO-WM directly on the server, the minimal source-level form is:

```python
self.register_buffer(
    "bias",
    generate_mask_matrix(NUM_PATCHES, NUM_FRAMES),
    persistent=False,
)
```

and ensure the mask used in `forward()` is on `dots.device`.

Other `.to("cuda")` occurrences are in distributed/R3M/wall/deformable paths or comments and are not on this PushT planning path.

## Planned Run Command

Default single-seed 30-pair run:

```bash
cd /path/to/leWM/lewm-failure-audit
python scripts/revision/dino_wm_pool_audit.py \
  --dino-repo /path/to/leWM/dino_wm \
  --ckpt-base-path /path/to/leWM/dino_wm/checkpoints \
  --device cuda:0
```

Conservative disk-saving variant:

```bash
python scripts/revision/dino_wm_pool_audit.py \
  --dino-repo /path/to/leWM/dino_wm \
  --ckpt-base-path /path/to/leWM/dino_wm/checkpoints \
  --device cuda:0 \
  --save-float16-latents
```

One-pair smoke after checkpoint setup:

```bash
python scripts/revision/dino_wm_pool_audit.py \
  --dino-repo /path/to/leWM/dino_wm \
  --ckpt-base-path /path/to/leWM/dino_wm/checkpoints \
  --device cuda:0 \
  --pair-ids 25 \
  --output results/revision/dino_wm_pool_audit_smoke.json \
  --pool-dir results/revision/dino_wm_pusht_pools_smoke
```

The default `Rendpoint` uses `first_iter_real_z`: it scores the broad first CEM sample pool with real terminal DINO-WM embeddings. This gives a same-pair endpoint reference without needing to regenerate the full historical Track A 80-action endpoint artifact. The final-pool real-endpoint correlation is also saved as `Rpool_Creal_z_final`.

## What To Verify Before Running

- `python scripts/revision/dino_wm_pool_audit.py --list-pairs` prints the 30 expected pair IDs:
  `25,46,60,61,67,70,71,73,0,1,2,3,4,5,12,13,14,16,19,30,6,7,8,9,10,74,75,76,77,78`.
- DINO checkpoint files exist:
  - `dino_wm/checkpoints/outputs/pusht/hydra.yaml`
  - `dino_wm/checkpoints/outputs/pusht/checkpoints/model_latest.pth`
- Track A HDF5 exists at the path in `results/phase1/track_a_pairs.json`.
- CUDA is available.
- `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")` can either reach the network or has a cached DINOv2 model.
- `SDL_VIDEODRIVER=dummy` is set on headless servers.
- Available disk is at least 5 GB for float32 final-pool latents; more if `--save-endpoint-pools` is enabled.

## Runtime Estimate

Default one-seed run:

- DINO CEM model evaluations: `30 pairs * 30 iterations * 300 candidates`.
- Simulator scoring: `30 pairs * 300 final candidates`, plus another `30 * 300` if `Rendpoint=first_iter_real_z` remains enabled.
- Raw simulator steps: each candidate has 25 PushT actions.

Expected on one modern CUDA GPU plus CPU simulator:

- Smoke pair: about 2-8 minutes, depending on DINOv2 cache and CPU simulator speed.
- Full 30-pair default run: about 2-5 hours.
- Disk: about 3 GB for final-pool float32 DINO patch latents; about half with `--save-float16-latents`.

Use the one-pair smoke first, then run the full set once checkpoint/model/device loading is confirmed.
