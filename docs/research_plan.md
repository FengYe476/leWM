# Phase 0 Research Plan: LeWM Long-Horizon Failure Audit

## Background

LeWorldModel (LeWM) is a lightweight JEPA world model that plans from pixels by combining a ViT-based encoder, an autoregressive predictor, and a planner over learned costs. The short-horizon PushT baseline is already strong, which makes it a poor regime for failure analysis: when success is near ceiling, errors are too rare and too weakly separated to attribute cleanly. Phase 0 therefore shifts evaluation into a deliberately harder long-goal regime, while holding the core model and planner fixed, so that failures become frequent enough to diagnose without changing the underlying system.

This audit is designed as a localization study, not an optimization study. The aim is to determine whether long-horizon degradation is dominated by encoder geometry, predictor rollout drift, planner weakness, or localized events that only become visible on individual trajectories.

## Research Question

Where does LeWM's long-horizon goal-conditioned planning failure originate?

## Phase 0 Objective

The specific goal of Phase 0 is to identify a diagnostic stress regime and then attribute failures using a fixed protocol. The intended output is a causal triage map, not a final architectural intervention.

## Environments

Phase 0 uses two environments:

- `swm/PushT-v1` as the primary diagnostic environment
- `swm/OGBCube-v0` as the secondary cross-check environment

PushT is the first target because it is already reproduced locally, evaluation is fast enough for repeated sweeps, and the goal-conditioned structure is clean enough to support trajectory-level inspection.

## Long-Horizon Stress Test

The stress-test variable is `goal_offset_steps`, measured in dataset rows and environment steps.

For PushT:

- baseline offset: `25`
- sweep offsets: `25, 50, 75, 100`
- evaluation budget: `2 x goal_offset_steps`

Because the planner uses `action_block = 5`, one planner block covers five environment steps. Therefore:

- `25` offset steps = `5` planner blocks
- `50` offset steps = `10` planner blocks
- `75` offset steps = `15` planner blocks
- `100` offset steps = `20` planner blocks

The target stress regime is the first offset range where success falls out of the near-ceiling regime and into the diagnostic band of roughly `40%` to `70%`.

## Diagnostic Protocol

### 1. Three-cost attribution

The audit uses a three-cost view of planning failure. Let the current observation at time $t$ be $o_t$, the goal observation be $g$, and an action sequence be $a_{t:t+H-1}$.

We track three conceptually distinct costs:

1. Encoder geometry cost, $c_{enc}$.
2. Predictor rollout cost, $c_{pred}$.
3. Executed trajectory or planner-selected cost, $c_{exec}$.

In words:

- $c_{enc}$ asks whether the encoder embeds current and goal observations in a geometry that preserves meaningful progress structure.
- $c_{pred}$ asks whether the learned dynamics preserve that structure under imagined rollouts.
- $c_{exec}$ asks whether the planner chooses and executes sequences that realize the imagined improvement in the real environment.

The attribution logic is comparative rather than absolute. Large disagreement between these costs is more diagnostic than any single scalar value in isolation.

### 2. Aggregate diagnostics

Aggregate diagnostics summarize behavior across many trajectories at a fixed offset:

- success rate
- cost-to-go trends
- cost agreement or disagreement between encoder and predictor
- failure rate as a function of horizon
- sensitivity to start state and goal displacement

Aggregate statistics answer the question:

Does the model fail in a stable, repeatable way once horizon pressure increases?

### 3. Per-trajectory diagnostics

Per-trajectory diagnostics answer a different question:

When a rollout fails, where along that rollout does the failure first become visible?

Trajectory-level inspection looks for:

- sudden jumps in latent goal distance
- steadily accumulating rollout drift
- planner commitment to a bad branch
- localized event failures such as missing contact, mis-timing, or irreversible deviation

### 4. Aggregate-first, trajectory-second workflow

The order of operations is fixed:

1. Find the offset where failure rate is informative.
2. Run aggregate diagnostics there.
3. Sample representative successes and failures.
4. Inspect per-trajectory behavior only after the aggregate picture is stable.

This prevents anecdotal trajectory reading from dominating the early audit.

## Decision Tree

The decision tree maps observed evidence to one of six Phase 0 cases.

### Case A: No meaningful long-horizon failure

Criteria:

- success remains near ceiling even at stressed offsets
- encoder, predictor, and execution costs remain aligned

Interpretation:

- Phase 0 on the current environment is under-stressed
- increase horizon pressure or move faster to the secondary environment

### Case B: Planner-dominated failure

Criteria:

- encoder and predictor costs indicate a good route
- executed behavior fails to realize that route
- failures look like action-sequence search or commitment errors

Interpretation:

- the representation may be adequate
- the planner or its search budget is the likely bottleneck

### Case C: Predictor-dominated failure

Criteria:

- encoder geometry remains plausible
- imagined futures drift away from what execution needs
- rollouts become unreliable before the planner can exploit them

Interpretation:

- long-horizon degradation is likely caused by model rollout error accumulation

### Case D: Encoder-geometry failure

Criteria:

- latent distances or goal relations are already misleading at the observation level
- planner and predictor appear coherent inside a distorted embedding

Interpretation:

- failure originates upstream in the representation geometry rather than downstream planning

### Case E: Mixed model-plus-planner failure

Criteria:

- predictor error and planner misspecification co-occur
- no single component explains the majority of failures

Interpretation:

- the failure source is coupled
- Phase 0 should report mixed attribution rather than forcing a single-cause claim

### Case F: Event-localized or heterogeneous failure

Criteria:

- aggregate metrics are ambiguous
- failures cluster into distinct trajectory motifs
- some failures are contact-localized, timing-localized, or state-dependent

Interpretation:

- the model may not have one dominant failure mode
- trajectory-level taxonomy becomes a primary output

## Aggregate vs. Per-Trajectory Outputs

Both views are required.

Aggregate outputs:

- offset-sweep tables
- success-versus-offset curves
- summary statistics for costs and cost gaps
- environment-level failure proportions

Per-trajectory outputs:

- representative success and failure episodes
- trajectory-aligned cost traces
- event annotations for first visible deviation
- case labels under the A-F decision tree

## Execution Schedule

### Day 0

- verify all dependencies
- verify environments
- verify checkpoint load and forward pass
- verify Apple Silicon MPS path

### Day 1

- reproduce PushT baseline
- build long-goal sweep infrastructure
- begin offset sweep on PushT

### Day 2

- finish offset sweep
- select the diagnostic offset where success enters the target band

### Day 3

- freeze the PushT stress offset
- define the aggregate diagnostic batch

### Day 4

- implement encoder, predictor, and execution cost extraction

### Day 5

- run aggregate diagnostics on PushT

### Day 6

- add per-trajectory logging and trajectory selection utilities

### Day 7

- inspect representative successful and failed trajectories

### Day 8

- assign provisional Cases A-F on PushT

### Day 9

- port the same audit recipe to OGBench-Cube

### Day 10

- run Cube baselines and stress tests

### Day 11

- compare failure signatures across PushT and Cube

### Day 12

- finalize tables, plots, and example trajectories

### Day 13

- write interpretation and residual uncertainty

### Day 14

- finalize the Phase 0 report and handoff package

## Frozen Exclusions

The following are explicitly out of scope during Phase 0:

- no model retraining
- no architecture changes
- no planner redesign beyond fixed evaluation settings already chosen for reproduction
- no broad hyperparameter search
- no task expansion beyond the planned environments
- no claim of final causal proof from a single metric

These exclusions exist to keep Phase 0 diagnostic rather than exploratory in an unconstrained way.

## Outputs

Phase 0 is expected to produce:

- a verified reproduction package for PushT evaluation
- an offset sweep identifying the long-horizon failure onset
- aggregate diagnostic summaries for the chosen stress regime
- per-trajectory failure examples
- A-F decision-tree labels or partial labels
- a concise statement of whether long-horizon failure appears encoder-dominated, predictor-dominated, planner-dominated, mixed, or heterogeneous

## Current Working Interpretation

As of Day 1, PushT offset `50` already produces a success rate of about `58%`, which places the model inside the intended diagnostic band. That result makes offset `50` the leading candidate for the first full attribution pass, pending completion of the remaining sweep offsets and review of failure heterogeneity.
