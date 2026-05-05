# Phase F PLDM Feasibility Memo

Date: 2026-05-04

## Verdict

PLDM cross-model validation is feasible in principle, but it is not a same-day drop-in experiment. The original PLDM repository is public, but it does not appear to contain PushT-specific checkpoints or PushT configs. A public PushT PLDM checkpoint is advertised through the LeWorldModel baseline checkpoint suite on Google Drive, so the next concrete step is to download the `pldm` PushT archive, load it through `stable_worldmodel`, and inspect the object interface and latent shape on CPU.

## Local Search

Command:

```bash
grep -ri "pldm\|sobal" --include="*.py" --include="*.md" --include="*.txt" --include="*.yaml" . 2>/dev/null | head -20
find . -iname "*pldm*" 2>/dev/null
```

Findings:

- Local references are planning notes and paper-revision text marking PLDM as a future cross-model target.
- `third_party/le-wm/README.md` says the full baseline checkpoint suite includes PLDM for `two-room`, `pusht`, `cube`, and `reacher`.
- No local PLDM checkpoint or PLDM-specific file exists in this audit checkout.

`find . -iname "*pldm*"` returned no files.

## Online Findings

### Repository

Official PLDM repo:

```text
https://github.com/vladisai/PLDM
```

The PLDM project page links to the paper, arXiv, code, and data. It describes the model as a JEPA-style latent dynamics model trained from offline trajectories with an L2 prediction loss plus variance-covariance regularization, then planned in latent space by minimizing distance to a goal latent. Sources: [PLDM project page](https://latent-planning.github.io/) and [PLDM GitHub](https://github.com/vladisai/PLDM).

The PLDM GitHub README gives setup and training instructions, but the repo has no GitHub releases and no visible checkpoint files. Its environment tree contains `wall` and `diverse_maze`; no `pusht` files were found in the repository tree.

### Public PushT Checkpoint

The PLDM authors' repository does not expose a PushT checkpoint. However, the LeWorldModel release explicitly advertises a public baseline checkpoint suite containing PLDM for PushT:

- LeWM project page: “Additional checkpoints for baselines are available on Google Drive.”
- LeWM GitHub README: “The full baseline checkpoint suite (PLDM, LeJEPA, IVL, IQL, GCBC, DINO-WM, DINO-WM-noprop) is available on Google Drive,” with `pldm` marked available for `pusht`.

Source: [LeWM project page](https://le-wm.github.io/) and [LeWM GitHub README](https://github.com/lucas-maes/le-wm).

### Checkpoint Download Instructions

Use the LeWM baseline checkpoint suite rather than the original PLDM repo:

1. Open the LeWM baseline Google Drive linked from the LeWM README.
2. Download the PLDM PushT archive.
3. Extract it under `$STABLEWM_HOME`, preserving the archive's directory structure.
4. Load the object checkpoint using the stable-worldmodel convention:

```python
import stable_worldmodel as swm

cost = swm.policy.AutoCostModel("pusht/pldm")
cost.eval()
```

The exact `run_name` should be confirmed after extraction. The LeWM README documents the convention as a path relative to `$STABLEWM_HOME`, without the `_object.ckpt` suffix; object checkpoints are named `<name>_object.ckpt`.

### Encoder Interface

Two relevant interfaces exist:

1. `stable_worldmodel` checkpoint interface: `swm.policy.AutoCostModel(run_name)` scans the loaded checkpoint and returns the module with a `get_cost` method. This is the interface used by LeWM evaluation and should allow reuse of existing CEM-style planners once the checkpoint is downloaded.

2. Original PLDM repo interface: the PLDM model uses a backbone/encoder and predictor. The planner code encodes observations with `model.backbone(...)` and the backbone returns `BackboneOutput(encodings=...)`. The original PLDM planner is MPPI-based, not CEM-only.

For the audit protocol, the practical integration path is:

```python
cost = swm.policy.AutoCostModel("pusht/pldm")
# Use cost.get_cost(info_dict, action_candidates) for planning costs.
# Then inspect cost/model internals on CPU to expose image -> latent embeddings
# for R_endpoint, R_pool, and Delta_CEM.
```

### Latent Dimension

The PushT PLDM latent dimension is not stated in the public PLDM repo or LeWM README. It should be verified by loading the PushT PLDM checkpoint on CPU and running one transformed PushT observation through the returned encoder.

Known related facts:

- Original PLDM Two-Room config uses an IMPALA image encoder whose implementation maps to a 512-dimensional output vector.
- Original PLDM Diverse Maze config uses a different `menet6` / convolutional setup with proprioception.
- LeWM itself uses a single 192-dimensional token, but that value should not be assumed for PLDM.

Therefore, record the latent dimension as:

```text
PushT PLDM latent dimension: unknown until checkpoint load; verify by CPU forward pass.
```

### Planner Reuse

PLDM has an MPPI planner implementation in its public repo:

```text
pldm/planning/planners/mppi_planner.py
pldm/planning/planners/mppi_torch.py
```

For our audit, the easier path is probably to reuse the existing `stable_worldmodel` / audit CEM infrastructure around the checkpoint's `get_cost` method, then add pool capture. If we want optimizer-matched PLDM behavior, use PLDM's MPPI planner and add final-pool logging analogously to Phase D.

## Integration Effort Estimate

Estimated effort: 2-4 days after checkpoint download.

Breakdown:

- 0.5 day: download/extract PLDM PushT archive, confirm load path, CPU-load checkpoint.
- 0.5 day: inspect encoder and latent shape; write a small image-to-latent adapter.
- 1 day: adapt endpoint and final-pool metric extraction to PLDM's object interface.
- 0.5-1 day: run a small smoke test on 5-10 PushT pairs.
- 0.5-1 day: full 30-100 pair run depending on planner speed and whether we use CEM or PLDM MPPI.

## Recommendation

Do not include PLDM results in the current revision unless the LeWM baseline archive is downloaded and the PushT checkpoint loads cleanly. The memo supports a feasibility statement: PLDM is a good cross-model target, a public PushT baseline checkpoint appears to exist through the LeWM release, and the diagnostic protocol is directly applicable once final candidate pools and terminal latent embeddings are exposed.

## LaTeX-Ready Future-Work Text

Use this if PLDM is not integrated before submission:

```latex
We also investigated PLDM as a cross-model target for the same endpoint-pool diagnostic. A public PLDM implementation is available, and the LeWM release advertises PLDM baseline checkpoints for PushT, but the checkpoint was not integrated before the revision deadline. The proposed diagnostic protocol is model-agnostic: any latent world model that exposes terminal embeddings, a terminal-cost planner, and its optimizer-produced candidate pool can be evaluated with $\Rendpoint$, $\Rpool$, and $\DeltaCEM$. We therefore leave cross-model validation on PLDM as future work.
```

## Sources

- PLDM project page: https://latent-planning.github.io/
- PLDM GitHub repository: https://github.com/vladisai/PLDM
- PLDM arXiv paper: https://arxiv.org/abs/2502.14819
- LeWM project page: https://le-wm.github.io/
- LeWM GitHub repository and checkpoint README: https://github.com/lucas-maes/le-wm
