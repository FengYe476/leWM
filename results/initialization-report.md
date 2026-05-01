# LeWM Failure Audit Initialization Report

Verified on 2026-04-30.

## 1. Repository scaffold

Created:

```text
lewm-failure-audit/
├── README.md
├── requirements.txt
├── scripts/
├── configs/
├── notebooks/
└── results/
```

`README.md` contains a one-line project description. `requirements.txt` is intentionally empty.

## 2. Dependency research

### stable-worldmodel

- GitHub: https://github.com/galilai-group/stable-worldmodel
- Paper: https://arxiv.org/abs/2602.08968
- Install:
  - PyPI: `pip install stable-worldmodel`
  - Source/dev:
    ```bash
    git clone https://github.com/galilai-group/stable-worldmodel
    cd stable-worldmodel
    uv venv --python=3.10 && source .venv/bin/activate
    uv sync --all-extras --group dev
    ```
- Python requirement:
  - Explicit in `pyproject.toml`: `>=3.10`
- Key dependencies:
  - Core: `torch`, `torchvision`, `numpy`, `gymnasium`, `einops`, `lancedb>=0.30.0`, `pylance>=4.0.0`, `pyarrow`, `pillow`, `tqdm`, `typer`, `rich`
  - Train extra: `transformers>=4.50.0`, `stable-pretraining`, `hydra-core`, `hydra-submitit-launcher`, `wandb`
  - Env extra: `pygame`, `pymunk`, `shapely`, `ogbench`, `minigrid`, `gymnasium[all]`, `gymnasium-robotics`, `opencv-python`, `stable_baselines3>=2.0.0`, `craftax>=1.5.0`, `ale-py`
- PyTorch version note:
  - The current dependency manifest requires `torch` and `torchvision` but does not pin an exact version.

### stable-pretraining

- GitHub: https://github.com/galilai-group/stable-pretraining
- Paper: https://arxiv.org/abs/2511.19484
- Install:
  - Source:
    ```bash
    git clone https://github.com/galilai-group/stable-pretraining.git
    cd stable-pretraining
    pip install -e .
    ```
  - The README also documents a `uv` workflow after manually installing PyTorch.
- Python requirement:
  - No explicit `requires-python` field is currently declared in the repo's `pyproject.toml`.
  - The README's environment example uses `python=3.11`.
  - Inference: for LeWM integration, `Python 3.10+` is the practical target because `stable-worldmodel` requires `>=3.10`.
- Key dependencies:
  - Core: `torch`, `torchvision`, `torchmetrics`, `lightning`, `hydra-core`, `omegaconf`, `timm`, `transformers`, `wandb`, `datasets`, `pyarrow==20.0.0`, `minari[hdf5]>=0.5.3`, `scikit-learn>=1.7.0`, `opencv-python-headless`, `pylance`
  - CI pins for deterministic integration tests:
    - `torch==2.11.0`
    - `torchvision==0.26.0`
    - `lightning==2.6.1`
    - `transformers==4.55.4`
    - `timm==1.0.15`
- PyTorch version note:
  - The package install manifest leaves `torch` unpinned, but the repo's CI currently pins `torch==2.11.0`.

## 3. LeWM checkpoints (PushT and OGBench-Cube)

- LeWM GitHub: https://github.com/lucas-maes/le-wm
- Official website: https://le-wm.github.io/
- Paper: https://arxiv.org/abs/2603.19312

### Are pretrained checkpoints public?

Yes.

- The official website links a Hugging Face "Data & Checkpoints" collection:
  - https://huggingface.co/collections/quentinll/lewm
- The GitHub README explicitly lists pretrained model repos:
  - PushT: https://huggingface.co/quentinll/lewm-pusht
  - OGBench-Cube: https://huggingface.co/quentinll/lewm-cube
  - Also available: `lewm-tworooms`, `lewm-reacher`

### Download instructions

- The README provides an explicit Hugging Face CLI example for PushT:
  ```bash
  hf download quentinll/lewm-pusht --local-dir $STABLEWM_HOME/hf_pusht
  ```
- The model repos provide `weights.pt` plus `config.json`.
- The README then converts those files into the `_object.ckpt` format expected by `eval.py` and `stable_worldmodel`.
- Inference: the same workflow should apply to OGBench-Cube using `quentinll/lewm-cube`, but the README only spells out the exact command for PushT.

### Other checkpoint availability

- The GitHub README also links a Google Drive folder with baseline checkpoints:
  - https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e
- The website states additional baseline checkpoints are available there as well.

### If checkpoints had not been public, were training scripts available?

This is not needed because checkpoints are public, but training scripts are also provided:

- `train.py` in the LeWM repo
- Hydra configs under `config/train/`
- Evaluation entrypoint `eval.py`
- Baseline training scripts in `stable-worldmodel/scripts/train/`

## 4. OGBench environment

- GitHub: https://github.com/seohongpark/ogbench
- Paper/project: https://seohong.me/projects/ogbench/

### Install

- Base package:
  ```bash
  pip install ogbench
  ```
- Editable local install with training extras:
  ```bash
  pip install -e ".[train]"
  ```
- Reference implementation setup:
  ```bash
  cd impls
  pip install -r requirements.txt
  ```

### Python requirement

- Base package `pyproject.toml`: `>=3.8`
- Reference implementations / data generation README note: `Python 3.9+`

### External dependencies

- Yes, MuJoCo is required.
- The base package depends on:
  - `mujoco >= 3.1.6`
  - `dm_control >= 1.0.20`
  - `gymnasium[mujoco]`
- For the reference implementations and dataset generation, extra dependencies include:
  - `jax[cuda12] >= 0.4.26`
  - `flax >= 0.8.4`
  - `distrax >= 0.1.5`
  - `ml_collections`
  - `matplotlib`
  - `moviepy`
  - `wandb`
- Headless note:
  - The README recommends `MUJOCO_GL=egl` on remote/headless servers.

## Short takeaways

- `stable-worldmodel` is the cleanest dependency anchor for LeWM and has the clearest version floor: `Python >=3.10`.
- `stable-pretraining` is source-installable and mature enough for research use, but its Python floor is currently implicit rather than formally declared in `pyproject.toml`.
- LeWM checkpoints for both PushT and OGBench-Cube are publicly available through the official Hugging Face collection.
- OGBench is easy to install for environment use, but serious reproduction work still pulls in MuJoCo and a JAX stack.

## Sources

- https://github.com/galilai-group/stable-worldmodel
- https://github.com/galilai-group/stable-pretraining
- https://github.com/lucas-maes/le-wm
- https://le-wm.github.io/
- https://huggingface.co/collections/quentinll/lewm
- https://github.com/seohongpark/ogbench
- https://arxiv.org/abs/2602.08968
- https://arxiv.org/abs/2511.19484
- https://arxiv.org/abs/2603.19312
