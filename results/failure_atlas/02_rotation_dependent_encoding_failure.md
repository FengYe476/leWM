# Rotation-Dependent Encoding Failure

## 1. Failure Type Name

Rotation-dependent encoding failure.

This failure appears when the required block translation is modest or moderate but the goal requires a large orientation change. The latent objective can become weakly or negatively aligned with the physical success metric, especially around angular success.

## 2. Representative Trajectories

| Pair | Category | Successes | Block displacement | Rotation required | Encoder corr |
|---:|---|---:|---:|---:|---:|
| 2 | Impossible | 0/40 | 33.3 | 1.317 | -0.751 |
| 3 | Hard | 8/40 | 6.4 | 1.136 | 0.197 |
| 12 | Hard | 10/40 | 4.2 | 1.430 | -0.434 |

Pair `2` is the strongest rotation-sensitive failure: displacement is only `33.3` px, but required rotation is `1.317` rad and the encoder correlation is strongly negative. Pairs `3` and `12` isolate rotation even more sharply: both have small displacement but require rotations above `1.1` rad, and both remain hard despite being spatially near the goal.

## 3. Blow-Up Step With Physical Event Annotation

| Pair | Blow-up step t* | Mean predictor error | Max predictor error | Annotation |
|---:|---:|---:|---:|---|
| 2 | 9 | 8.899 | 10.541 | Late-horizon mismatch after the block should have resolved a large angular change; the best CEM_late action remains far outside angular success. |
| 3 | Not in step plot | Not sampled | Not sampled | Recommended for the next event pass because it is a small-displacement, high-rotation hard pair. |
| 12 | Not in step plot | Not sampled | Not sampled | Useful control for rotation: model-best CEM succeeds, but rank correlation is negative across the candidate set. |

## 4. Three-Cost Ranking

For pair `2`, the planner-best `CEM_late` action has `C_model = 249.4`, `C_real_z = 339.3`, and `C_real_state = 92.6`. It is ranked `1st` by model cost but `25th` by real-state cost. The real-state-best action has `C_real_state = 32.4`, but it is ranked `35th` by model cost and still fails the angular threshold.

For pair `3`, the planner-best action has `C_model = 2.724`, `C_real_z = 307.4`, and `C_real_state = 8.68`. It fails because angular error remains `1.031` rad. The real-state-best action reduces `C_real_state` to `0.655`, but angular error is still `0.399` rad, slightly above the `pi/9` success threshold.

These pairs show that low latent cost can coexist with unresolved orientation error.

## 5. Decision-Tree Assignment

**Case B/E.** The representation appears to underweight or mis-order orientation-sensitive terminal states. This is not a planner-only failure because the candidate set includes lower real-state-cost actions that the model objective ranks poorly.
