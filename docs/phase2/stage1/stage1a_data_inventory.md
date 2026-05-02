# Stage 1A Data Inventory

Date: 2026-05-02

Scope: read-only inventory for Stage 1A feature/endpoint ranking controls C0-C7. No controls are implemented here.

## Phase 2 Script Inventory

`scripts/phase2/` currently contains:

- `analyze_all_splits.py`
- `analyze_split1.py`
- `analyze_track_b.py`
- `cost_head_model.py`
- `dataloader.py`
- `deep_diagnosis.py`
- `diagnose_planning_gap.py`
- `eval_oracle_budget_cem.py`
- `eval_planning.py`
- `extract_dinov2_features.py`
- `extract_latents.py`
- `extract_pixels.py`
- `extract_predicted_latents.py`
- `mahalanobis_baseline.py`
- `random_projection_control.py`
- `splits.py`
- `track_b_common.py`
- `train_cem_aware.py`
- `train_cost_head.py`

## Track A Three-Cost JSON Schema

Path: `results/phase1/track_a_three_cost.json`

Top-level object:

- `metadata`: run metadata object.
- `pairs`: list of 100 pair records.

`metadata` fields:

| Field | Type | Meaning |
|---|---:|---|
| `pairs_path` | string | Source Track A pair spec path. |
| `n_pairs_requested` | number | Requested pair count; 100. |
| `n_pairs_completed` | number | Completed pair count; 100. |
| `device` | string | Evaluation device; `mps`. |
| `seed` | number | Run seed; 0. |
| `action_counts` | object | Candidate counts: 20 each for `data`, `smooth_random`, `CEM_early`, `CEM_late`. |
| `cem_config` | object | CEM parameters: 300 samples/iter, 30 iterations, 30 elites, planning/receding horizon 5, action block 5, early iter 3, late iter 30. |
| `git_commit` | string | Source commit for the run. |
| `timestamp_started` | string | UTC ISO timestamp. |
| `timestamp_finished` | string | UTC ISO timestamp. |
| `fixed_sequence_length_raw_steps` | number | Raw sequence length; 50. |
| `fixed_sequence_length_action_blocks` | number | Action blocks; 10. |
| `offset_steps_at_runtime` | number | Goal offset; 50. |

Each `pairs[]` record represents one stratified PushT start-goal pair at offset 50. Pair fields:

| Field | Type | Meaning |
|---|---:|---|
| `pair_id` | number | Stable pair ID, 0-99. |
| `cell` | string | Stratification cell, e.g. `D0xR0`. |
| `episode_id` | number | Dataset episode ID. |
| `start_row` | number | Dataset row for the initial state. |
| `goal_row` | number | Dataset row for the goal state. |
| `block_displacement_px` | number | Initial-to-goal block displacement in pixels. |
| `required_rotation_rad` | number | Initial-to-goal block rotation in radians. |
| `wallclock_seconds` | number | Per-pair evaluation wallclock time. |
| `actions` | list | 80 executed candidate action records for this pair. |

Each `actions[]` record represents one candidate action sequence executed from that pair's initial state. Action fields:

| Field | Type | Meaning |
|---|---:|---|
| `action_id` | number | Candidate index within pair. |
| `source` | string | Candidate source: `data`, `smooth_random`, `CEM_early`, or `CEM_late`. |
| `C_real_z` | number | Squared Euclidean distance between terminal LeWM latent and goal LeWM latent after real environment rollout. |
| `C_model` | number | LeWM model/planner predicted cost for the candidate sequence. |
| `C_real_state` | number | Physical state cost from terminal state to goal state; lower is better. |
| `success` | boolean | Success flag from physical pose metrics. |

Action ID/source layout is fixed for every pair:

| Source | Count | `action_id` range |
|---|---:|---:|
| `data` | 20 | 0-19 |
| `smooth_random` | 20 | 20-39 |
| `CEM_early` | 20 | 40-59 |
| `CEM_late` | 20 | 60-79 |

Cell coverage:

- 100 pairs total, 8000 candidate records total.
- Most cells contain 6 pairs; `D2xR2`, `D2xR3`, `D3xR1`, and `D3xR3` contain 7 pairs.
- Anchor latent-favorable cells are present as `D0xR1` pair IDs 6-11 and `D1xR0` pair IDs 24-29.

Important limitation: this JSON stores scalar costs only. It does not store raw action sequences, terminal pixels, terminal latents, or goal latents.

## Existing Latent and Feature Artifacts

All `.pt` artifacts were inspected with `conda run -n lewm-audit python` and `torch.load(..., map_location="cpu", weights_only=False)`.

### Stage 1A-Relevant Artifacts

| Path | Format / role | Keys and shapes |
|---|---|---|
| `results/phase2/p2_0/track_a_latents.pt` | Replayed Track A endpoint and goal LeWM latents. Primary C0 artifact. | `pair_id [8000] int64`, `action_id [8000] int64`, `source` list length 8000, `source_index [8000] int64`, `action_key` list length 8000, `cell` list length 8000, `z_terminal [8000,192] float32`, `z_goal [8000,192] float32`, `v1_cost [8000] float32`, `success [8000] bool`, `C_real_z [8000] float32`, `C_model [8000] float32`, `C_real_state [8000] float32`, `block_pos_dist [8000] float32`, `angle_dist [8000] float32`, `metadata`. |
| `results/phase2/p2_0/track_a_predicted_latents.pt` | Predicted latent artifact from P2-0, not needed for endpoint-only Stage 1A unless comparing predicted geometry later. | `pair_id [8000]`, `action_id [8000]`, `source`, `source_index [8000]`, `action_key`, `cell`, `z_predicted [8000,192] float32`, `z_goal_pred [8000,192] float32`, `metadata`. |
| `results/phase2/p2_0/track_a_predicted_latents_smoke.pt` | Smoke subset of predicted latents. | Same predicted-latent schema, but 80 records: `z_predicted [80,192]`, `z_goal_pred [80,192]`. |
| `results/phase2/track_b/dinov2_features.pt` | Cached DINOv2 Track B endpoint and goal features. C7 artifact. | `pair_id [8000]`, `action_id [8000]`, `source`, `source_index [8000]`, `cell`, `d_terminal_cls [8000,768] float32`, `d_goal_cls [8000,768] float32`, `d_terminal_mean [8000,768] float32`, `d_goal_mean [8000,768] float32`, `metadata`. |
| `results/phase2/track_b/random_projection_features.pt` | Existing seed-0 Gaussian random projection baseline from Track B. Partial precursor for C2, not full Stage 1A. | `pair_id [8000]`, `action_id [8000]`, `source`, `source_index [8000]`, `cell`, `r_terminal [8000,192] float32`, `r_goal [8000,192] float32`, `projection [192,192] float32`, `metadata`. |

`track_a_latents.pt` metadata confirms:

- format: `p2_0_track_a_latents`
- seed: 0
- device: `mps`
- offset: 50
- latent dim: 192
- image size: 224
- 100 completed pairs and 8000 records
- it was created by replaying Track A because Phase 1 scalar JSON did not store endpoint observations or embeddings

### Other `.pt` Files Under `results/phase2/`

There are 40 `.pt` files total. Aside from the five artifacts above, the remaining files are P2-0 learned metric/model checkpoints, not endpoint feature datasets:

- CEM-aware cost-head checkpoints under `results/phase2/p2_0/cem_aware_*`: top-level `model_state_dict`, `model_type`, `variant`, `split`, `epoch`, validation metric fields, and `args`. Small/CEM-aware models use tensors such as `net.0.weight [128,768]`, `net.2.weight [128,128]`, `net.4.weight [1,128]`.
- Mahalanobis checkpoints under `results/phase2/p2_0/mahalanobis_*`: `raw_d [192]`, `A [192,192]`, or `L [192,16]` depending on diagonal/full/lowrank variant.
- Split 1/2/3 cost-head checkpoints under `results/phase2/p2_0/split*_*/`: top-level `model_state_dict`, `variant`, `split`, `fold`, `epoch`, `val_pairwise_accuracy`, and `args`. Small models are 128-hidden MLPs; large models include `net.0.weight [512,768]`.

These checkpoints may be useful historical context but should not be used as C0-C7 feature controls.

## DINOv2 Track B Features

Path: `results/phase2/track_b/dinov2_features.pt`

Metadata:

- format: `phase2_track_b_dinov2_features`
- model repo: `facebookresearch/dinov2`
- model: `dinov2_vitb14`
- feature dim: 768
- feature variants: CLS and mean-pool
- source latent ordering: `results/phase2/p2_0/track_a_latents.pt`
- records: 8000
- raw pixels were replayed and encoded in batches; raw pixels were not stored

Existing Track B summary in `results/phase2/track_b/ranking_comparison.json` evaluated ranking against `v1_cost`, not `C_real_state`. Main rows:

| Encoder | Dim | Global Spearman | Pairwise accuracy | Per-pair rho mean | Per-pair rho std |
|---|---:|---:|---:|---:|---:|
| LeWM (SIGReg) | 192 | 0.5033 | 0.6470 | 0.3549 | 0.5021 |
| DINOv2 CLS | 768 | 0.2387 | 0.5943 | 0.2482 | 0.3651 |
| DINOv2 mean-pool | 768 | 0.2609 | 0.6099 | 0.2848 | 0.3686 |
| Random projection | 192 | 0.5010 | 0.6429 | 0.3493 | 0.4887 |

For Stage 1A, C7 can reuse `d_terminal_cls/d_goal_cls` and `d_terminal_mean/d_goal_mean`; metrics need to be rerun against the Stage 1A targets, especially `C_real_state`, top-k overlap, false-elite rate, anchor subsets, and 4x4 heatmaps.

## Existing Random Projection Code

Path: `scripts/phase2/random_projection_control.py`

Current behavior:

- Loads `results/phase2/p2_0/track_a_latents.pt` by default through `DEFAULT_LATENT_ARTIFACT`.
- Reads `z_terminal` and `z_goal`.
- Creates a seed-controlled Gaussian matrix with shape `[192,192]`:
  - `projection = torch.randn((dim, dim)) / sqrt(dim)`
  - applied as `z @ projection`
- Saves ordering fields from the latent artifact: `pair_id`, `action_id`, `source`, `source_index`, `cell`.
- Saves `r_terminal`, `r_goal`, and `projection`.

Reusable pieces:

- Artifact loading and ordering preservation.
- Seeded `torch.Generator` pattern.
- Metadata pattern for projected feature artifacts.

Limitations for Stage 1A:

- Only one seed is supported by default.
- Only 192-to-192 Gaussian projection is implemented.
- It is not an orthogonal projection control.
- It does not support lower-dimensional projections `m in {1,2,4,8,16,32,64,128,192}`.
- It does not implement coordinate subset, Gaussian null, shuffled latent, or random-init encoder controls.

Related reusable infrastructure:

- `scripts/phase2/track_b_common.py` provides Track A replay helpers, latent artifact loading, ordering validation, and pair/action indexing.
- `scripts/phase2/analyze_track_b.py` provides reusable metric functions: squared L2 cost, within-pair pairwise accuracy, per-pair Spearman, feature-order validation, and per-cell pairwise accuracy.

## V1 Oracle Data

Directory: `results/phase1/v1_oracle_ablation/`

Files:

| Path | Pairs | Cells | Actions per pair |
|---|---:|---|---:|
| `v1_d0.json` | 24 | `D0xR0`, `D0xR1`, `D0xR2`, `D0xR3` | 40 |
| `v1_d1.json` | 24 | `D1xR0`, `D1xR1`, `D1xR2`, `D1xR3` | 40 |
| `v1_d2.json` | 26 | `D2xR0`, `D2xR1`, `D2xR2`, `D2xR3` | 40 |
| `v1_d3.json` | 26 | `D3xR0`, `D3xR1`, `D3xR2`, `D3xR3` | 40 |

Top-level schema matches the Track A JSON pattern:

- `metadata`
- `pairs`

V1 metadata includes:

- `variant`: `V1`
- `cost_formula`: hinge physical cost, `max(block_pos_dist - 20, 0) + alpha * max(angle_dist - pi/9, 0)`
- `alpha`: 57.29577951308232
- `planner_cost_source`: `oracle_real_state`
- `cells_evaluated`
- `n_pairs`
- CEM config matching Track A
- seed/device
- `action_counts`: `data=0`, `smooth_random=0`, `CEM_early=20`, `CEM_late=20`
- `actions_subset`: `cem_only`
- `data_random_storage`: `omitted_for_cem_only_outputs`

Each V1 pair record has:

| Field | Type | Meaning |
|---|---:|---|
| `pair_id` | number | Same Track A pair ID. |
| `cell` | string | Cell label. |
| `episode_id` | number | Dataset episode ID. |
| `start_row` | number | Initial dataset row. |
| `goal_row` | number | Goal dataset row. |
| `block_displacement_px` | number | Pair displacement. |
| `required_rotation_rad` | number | Pair rotation. |
| `physical_pose_distance` | number | Initial-goal physical pose distance. |
| `wallclock_seconds` | number | Per-pair wallclock. |
| `actions` | list | 40 CEM-only V1 oracle candidates. |

Each V1 action record has:

| Field | Type | Meaning |
|---|---:|---|
| `action_id` | number | 0-39 within the V1-only candidate set. |
| `source` | string | `CEM_early_V1` or `CEM_late_V1`. |
| `source_index` | number | 0-19 within each source. |
| `C_real_z` | number | LeWM endpoint latent cost after rollout. |
| `C_model` | number | LeWM model cost. |
| `C_real_state` | number | Physical state cost. |
| `C_variant` | number | V1 hinge cost used by oracle planner. |
| `success` | boolean | Success flag. |
| `cem_iter` | number | 3 or 30. |
| `cem_rank` | number | Rank within selected CEM elites. |
| `cem_oracle_cost` | number | Oracle cost from CEM selection. |

Stage 1A note: the V1 ablation JSONs contain a separate V1-planned CEM candidate pool, not the same 80 candidates per pair from `track_a_three_cost.json`. For top-k overlap with a V1 oracle ranking over the existing 8000 Track A candidates, the aligned source is `v1_cost [8000]` in `track_a_latents.pt`.

## Anchor Subsets Already Available

Invisible quadrant:

- Source: `results/phase1/track_a_analysis/failure_mode_decomposition.json`
- Definition: `all_fail + strong_rho`
- Pair IDs: 25, 46, 60, 61, 67, 70, 71, 73, 78, 86, 87, 93, 94, 96, 97, 99
- Count: 16
- Cells: `D1xR0`, `D1xR3`, `D2xR2`, `D2xR3`, `D3xR0`, `D3xR1`, `D3xR2`, `D3xR3`

Sign-reversal cluster:

- Source: `results/phase1/track_a_analysis/track_a_sign_reversal_pairs.json`
- Definition: negative per-pair Spearman rho for `C_real_z` vs `C_real_state`
- Pair IDs: 20, 40, 42, 53, 44, 49, 66, 17, 21, 62, 47, 69, 22, 37, 33, 45, 98, 23, 18, 29, 15
- Count: 21
- Cells: `D0xR2`, `D0xR3`, `D1xR0`, `D1xR1`, `D1xR2`, `D1xR3`, `D2xR0`, `D2xR2`, `D2xR3`, `D3xR3`

Latent-favorable cells:

- `D0xR1`: pair IDs 6, 7, 8, 9, 10, 11
- `D1xR0`: pair IDs 24, 25, 26, 27, 28, 29

## What Needs To Be Built For Stage 1A

Common Stage 1A harness:

- Load and validate alignment across `track_a_three_cost.json`, `track_a_latents.pt`, `dinov2_features.pt`, and optional projection artifacts by `(pair_id, action_id)`.
- Build a single 8000-row table with pair metadata, action source, `C_real_state`, `success`, `C_real_z`, `C_model`, `v1_cost`, LeWM endpoint/goal latents, and DINO endpoint/goal features.
- Compute all requested metrics for each control: Spearman vs `C_real_state`, within-pair pairwise accuracy, top-k overlap with LeWM ranking, top-k overlap with V1 ranking, false-elite rate, anchor-subset metrics, and 4x4 per-cell heatmaps.
- Aggregate multi-seed controls with mean/std and preserve per-seed rows.

Control-specific build status:

| Control | Status | Needed work |
|---|---|---|
| C0 LeWM 192-d Euclidean | Data exists | Use `z_terminal/z_goal` from `track_a_latents.pt`; validate against `C_real_z`. |
| C1 same-dim orthogonal `Q*z` | Not built | Add 10-seed orthogonal matrix generation. Euclidean distances should match C0 to numerical precision, so this is mainly a sanity/control check. |
| C2 Gaussian projection 192-to-`m` | Partially built | Generalize `random_projection_control.py` to `m in {1,2,4,8,16,32,64,128,192}` and 10 seeds. Existing artifact covers only seed 0, `m=192`. |
| C3 coordinate subset | Not built | Add 10-seed random coordinate selection for `m in {8,16,32,64}`. |
| C4 Gaussian null | Not built | Generate independent `N(0,I)` endpoint/goal features with matched dimensions/seeds and Track A ordering. |
| C5 shuffled latent | Not built | Define and implement a reproducible permutation control that preserves latent marginals while destroying observation-goal correspondence. |
| C6 random-init LeWM encoder | Not built | Instantiate the LeWM encoder architecture with random weights and encode real terminal/goal observations. Because raw pixels are not stored, this likely requires replaying Track A endpoints using `track_b_common.py` or adding a new cached feature artifact. |
| C7 DINOv2 reference | Data exists | Reuse `dinov2_features.pt` for CLS and mean-pool variants; rerun Stage 1A metrics against `C_real_state` and requested anchors. |

Recommended reuse:

- Use `track_a_latents.pt` as the canonical row order and endpoint feature source.
- Use `v1_cost` in `track_a_latents.pt` for V1 top-k ranking over the same Track A candidate set.
- Reuse `analyze_track_b.py` metric helpers, but extend them for Stage 1A's target/overlap/false-elite requirements.
- Reuse `track_b_common.py` replay utilities only for controls that need pixels or fresh encoder passes, especially C6.
