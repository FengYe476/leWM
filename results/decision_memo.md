# Decision Memo: Phase 0 Failure Mode Audit

## Classification

**Case B/E hybrid: encoder goal geometry fails under large physical displacements, with event-localized characteristics.**

The binding Phase 0 conclusion is that PushT long-horizon degradation is not primarily caused by predictor rollout drift or planner search failure. The dominant failure signature is representation-side: LeWM's latent goal geometry becomes unreliable for large block displacements and substantial rotations. The failure is heterogeneous across initial-goal pairs, so the encoder diagnosis should be refined as event-localized rather than treated as a uniform global collapse.

## Evidence Summary

- Offset sweep localized the diagnostic regime: success falls from `96%` at offset `25` to `58%` at offset `50`, then to `16%` and `10%` at offsets `75` and `100`.
- Predictor is not the failure source: `corr(C_model, C_real_z) = 0.864`, so imagined terminal latent costs track encoded real terminal observations well.
- Planner is not the failure source: `23/30` offset-50 pairs have at least one successful action in the fixed evaluation set, and `CEM_late` succeeds on `52.7%` of candidate actions versus `15.7%` for random actions.
- Encoder geometry is pair-dependent: global `corr(C_real_z, C_real_state) = 0.669`, but per-pair mean is only `0.353 +/- 0.486`.
- Impossible pairs show near-zero or negative encoder-to-physics alignment: mean per-pair `corr(C_real_z, C_real_state) = -0.046`.
- Failure correlates strongly with physical difficulty: physical pose distance versus success has Spearman `rho = -0.751`; block displacement versus success has `rho = -0.741`; required rotation versus success has `rho = -0.627`.
- Latent endpoint distance compresses large physical changes: Easy pairs average `2.6` px block displacement and latent distance `12.5`; Impossible pairs average `123.3` px block displacement but only latent distance `19.4`.

## Binding Next Action

Case B applies: pivot to representation and SIGReg-style geometry research. Do not spend Phase 1 effort on planner redesign, CEM budget sweeps, or rollout-loss work as the primary intervention.

Case E refinement applies: the failure is event-localized and displacement-dependent. Phase 1 should investigate representation objectives that preserve task-relevant geometry under large block translations and rotations, rather than optimizing only aggregate latent prediction quality.

## Phase 1 Recommendations

- Prioritize displacement-aware representation diagnostics: measure whether latent distances and rank orderings remain monotonic with block-pose distance over controlled displacement and rotation bins.
- Evaluate geometry-correcting objectives such as SIGReg-style regularization, contrastive pose ordering, or privileged pose-distance calibration as diagnostic interventions.
- Add per-event atlas expansion: split failures by large translation, large rotation, contact/timing ambiguity, and near-goal angle mismatch.
- Keep the CEM planner and predictor fixed for the first Phase 1 intervention pass, so improvements can be attributed to representation geometry rather than search changes.
- Use offset `50` as the primary stress setting and retain offsets `75` and `100` as out-of-distribution stress checks after representation changes.
