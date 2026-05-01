# Large-Displacement Encoder Geometry Failure

## 1. Failure Type Name

Large-displacement encoder geometry failure.

This failure occurs when the goal requires a large block translation. The latent goal geometry stops ranking physically better terminal states reliably, even though the predictor remains faithful to its own latent objective.

## 2. Representative Trajectories

| Pair | Category | Successes | Block displacement | Rotation required | Encoder corr |
|---:|---|---:|---:|---:|---:|
| 21 | Impossible | 0/40 | 208.2 | 1.938 | -0.180 |
| 6 | Impossible | 0/40 | 99.4 | 1.810 | -0.736 |
| 23 | Impossible | 0/40 | 142.9 | 0.433 | 0.542 |

Pair `21` is the clearest large-displacement example: it requires moving the block by `208.2` px and rotating by `1.938` rad, and no sampled action succeeds. Pair `6` combines large displacement with strongly negative encoder-to-physics rank correlation. Pair `23` shows the same zero-success outcome under large displacement even when the aggregate rank correlation is not negative, suggesting that large translation alone can leave all fixed candidate actions outside the success basin.

## 3. Blow-Up Step With Physical Event Annotation

Representative per-step diagnostics used the planner-best `CEM_late` action, selected by minimum `C_model`.

| Pair | Blow-up step t* | Mean predictor error | Max predictor error | Annotation |
|---:|---:|---:|---:|---|
| 6 | 7 | 4.619 | 9.422 | Error spike occurs late in the 10-step latent horizon after the block should have made substantial progress toward a far goal. |
| 21 | Not in step plot | Not sampled | Not sampled | Large-displacement atlas exemplar; recommended for the next video/event annotation pass because model-best CEM is ranked 38th by real-state cost. |

## 4. Three-Cost Ranking

For pair `21`, the planner-best `CEM_late` candidate has `C_model = 259.2`, `C_real_z = 348.2`, and `C_real_state = 219.4`. It is ranked `1st` by model cost but only `38th` by real-state cost. The best real-state action is a random candidate with `C_real_state = 3.735`, but it still fails the block-only success criterion because the angular error remains `0.493` rad.

For pair `6`, the planner-best `CEM_late` candidate has `C_model = 240.3`, `C_real_z = 301.6`, and `C_real_state = 133.4`. It is ranked `1st` by model cost but `32nd` by real-state cost. The best real-state action is random, with `C_real_state = 44.7`, still outside success.

These rankings show that the model objective is not selecting physically best terminal states for large-displacement pairs.

## 5. Decision-Tree Assignment

**Case B/E.** The failure is representation-side and event-localized: large block displacement breaks the relationship between encoded goal proximity and physical task progress. This page supports the Phase 1 pivot toward representation geometry rather than planner or rollout-loss changes.
