# Track A Stratification Proposal

## (a) Restated Goal

Track A should turn the Phase 0 PushT finding from a 30-pair diagnostic into a publication-grade 100-pair characterization of when LeWM's encoder geometry fails. Phase 0 used random offset-50 pairs, and the per-pair Spearman correlation between `C_real_z` and `C_real_state` had a large standard deviation of `0.486`, partly because Easy, Hard, and Impossible cases were not balanced across physical pose changes. Stratified sampling matters because the downstream Track A heatmaps, latent-vs-physical scatter plots, updated Failure Atlas, and later Track C calibration splits all depend on having enough examples in each displacement and rotation regime.

## (b) Stratification Axes

Use two axes: block displacement in pixels and required block rotation in radians. These are the two most interpretable physical drivers from Phase 0: block displacement had Spearman `rho = -0.741` against success count, and required rotation had Spearman `rho = -0.627`. Physical pose distance should remain a derived diagnostic, not a third stratification axis, because Phase 0 defined it as `block_displacement + rotation_required`; adding it as a third axis would mostly duplicate the first two axes while making a 100-pair design too sparse.

## (c) Bin Edges

Recommended grid: `4 x 4` using empirical Phase 0 quartiles rounded to stable physical thresholds.

Displacement bins:

| Bin | Range px | Phase 0 motivation |
| --- | ---: | --- |
| D0 | `[0, 10)` | covers zero/tiny controls; Phase 0 p25 was `5.89` px |
| D1 | `[10, 50)` | covers moderate moves around the Phase 0 median `39.26` px |
| D2 | `[50, 120)` | covers large moves around the Phase 0 p75 `104.87` px |
| D3 | `[120, inf)` | covers extreme moves; Phase 0 max was `233.94` px |

Rotation bins:

| Bin | Range rad | Phase 0 motivation |
| --- | ---: | --- |
| R0 | `[0, 0.25)` | tiny rotation; Phase 0 p25 was `0.167` rad |
| R1 | `[0.25, 0.75)` | moderate rotation around the Phase 0 median `0.485` rad |
| R2 | `[0.75, 1.25)` | large rotation around the Phase 0 p75 `1.069` rad |
| R3 | `[1.25, inf)` | extreme rotation; Phase 0 max was `2.033` rad |

Phase 0 30-pair counts under this grid:

| Displacement \\ Rotation | R0 `[0,0.25)` | R1 `[0.25,0.75)` | R2 `[0.75,1.25)` | R3 `[1.25,inf)` | Row total |
| --- | ---: | ---: | ---: | ---: | ---: |
| D0 `[0,10)` | 7 | 1 | 1 | 1 | 10 |
| D1 `[10,50)` | 3 | 4 | 0 * | 1 | 8 |
| D2 `[50,120)` | 1 | 0 * | 3 | 2 | 6 |
| D3 `[120,inf)` | 1 | 2 | 0 * | 3 | 6 |
| Column total | 12 | 7 | 4 | 7 | 30 |

Cells marked `*` were never sampled by Phase 0. Treat these as priority coverage gaps, not as evidence that the cells are physically unreachable.

## (d) Per-Cell Budget

Use the `4 x 4` grid with a target total of `100` pairs. The base allocation is `6` pairs per cell, with four extra pairs assigned to high-risk large-motion cells that Phase 0 suggests are most diagnostic.

Target budget:

| Displacement \\ Rotation | R0 `[0,0.25)` | R1 `[0.25,0.75)` | R2 `[0.75,1.25)` | R3 `[1.25,inf)` | Row total |
| --- | ---: | ---: | ---: | ---: | ---: |
| D0 `[0,10)` | 6 | 6 | 6 | 6 | 24 |
| D1 `[10,50)` | 6 | 6 | 6 | 6 | 24 |
| D2 `[50,120)` | 6 | 6 | 7 | 7 | 26 |
| D3 `[120,inf)` | 6 | 7 | 6 | 7 | 26 |
| Column total | 24 | 25 | 25 | 26 | 100 |

Implementation rule for unreachable cells: enumerate the eligible offset-50 dataset pool before sampling. If a cell has zero eligible pairs, allocate `0` to that cell and document it; do not force-fill from another cell while pretending the grid is balanced.

Implementation rule for rare cells: if a cell has fewer eligible pairs than its target budget, cap the cell at the actually available count and document the cap. Any redistribution of leftover budget should require sign-off, because redistribution changes the interpretation of the heatmap.

## (e) Eligibility Rules For A Pair

A pair is defined as one valid PushT dataset start row plus a future goal row at `start_row + offset`. The start row is valid only if the future goal row remains inside the same episode, matching the Phase 0 valid-start-point constraint `step_idx <= episode_length - offset - 1`. The pair stores the start state/pixels and the future goal state/pixels; the goal is not encoded as fields inside the same 7-D state row.

Track A should use a single offset, `offset = 50`, because Phase 0 selected it as the diagnostic offset: it is hard enough to expose failures while still leaving both successes and failures. Offset `75` should be a stretch follow-up only after the offset-50 stratified analysis is stable, because mixing offsets would confound physical-pose coverage with horizon length.

## (f) Action-Sequence Count Per Pair

Use `80` action sequences per pair with an equal `20/20/20/20` split:

| Source | Count per pair | Reason |
| --- | ---: | --- |
| Data actions | 20 | preserves comparison to behavior-cloned/expert-like trajectories |
| Smooth random actions | 20 | measures action-space coverage independent of CEM |
| CEM-early candidates | 20 | probes the planner before convergence |
| CEM-late candidates | 20 | probes the planner-best distribution used in Phase 0 |

The equal split is the safest default because it preserves Phase 0's source balance while doubling per-source resolution. A CEM-heavy split such as `15/15/25/25` is defensible later if Track A becomes planner-focused, but the current hypothesis is encoder geometry rather than CEM search.

## (g) Expected Wall-Clock And Storage

Phase 0 evaluated `30 pairs * 40 actions = 1200` action records. Track A would evaluate `100 pairs * 80 actions = 8000` records, a `6.67x` increase.

The Phase 0 refactor reproduction took `148.6` seconds for the raw three-cost run on MPS, so a linear estimate for Track A is about `16.5` minutes for the raw cost pass. Add model warmup, stratified enumeration, per-pair diagnostics, plots, and likely reruns, and a practical local estimate is `30-60` minutes for one clean pass, with `1-2` days reserved for the full Track A workflow including QA and atlas updates.

Storage should remain modest if the JSON schema continues to store costs and metadata rather than pixels. The Phase 0 raw JSON was about `490 KB` for `1200` records, implying roughly `3.3 MB` for `8000` records. If Track A also stores raw action tensors in HDF5, `8000 * 50 * 2` float32 action values are about `3.2 MB` before compression; with metadata and optional latent summaries, expected storage is still likely below `25 MB` unless rendered pixels or videos are added.

## (h) Open Questions For Sign-Off

- Confirm the `4 x 4` bin edges: displacement `[0, 10, 50, 120, inf]` px and rotation `[0, 0.25, 0.75, 1.25, inf]` rad.
- Confirm the target allocation is exactly `100` pairs with the budget matrix above.
- Confirm `offset = 50` only for Track A, with offset `75` deferred as a stretch follow-up.
- Confirm the equal `20/20/20/20` action-source split.
- Decide whether leftover budget from truly empty or rare cells should remain unfilled or be redistributed to neighboring cells after explicit review.
- Decide whether zero-displacement controls should be capped further if the D0/R0 pool is very large.
