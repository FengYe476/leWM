# Phase 0 Revisions After Phase 1 Track A

## 1. Purpose and Scope

This document records Phase 1 revisions to Phase 0 conclusions. The Phase 0 source documents, including [results/phase0_report.md](../../results/phase0_report.md), [results/decision_memo.md](../../results/decision_memo.md), and the original Failure Atlas pages under [results/failure_atlas/](../../results/failure_atlas/), are not modified. This document is the authoritative source for which Phase 0 claims still hold, which are revised, and which are falsified by the Phase 1 Track A 100-pair stratified evidence and oracle-CEM ablations.

## 2. Claim-by-Claim Audit

| Phase 0 claim | Phase 1 status | Phase 1 evidence | Notes |
|---|---|---|---|
| "Predictor faithfully preserves encoder geometry." | HOLDS | Track A global Pearson(C_model, C_real_z)=`0.814`; per-row Pearson remains high: D0=`0.865`, D1=`0.848`, D2=`0.747`, D3=`0.780`. Evidence: [docs/phase1/track_a_three_cost_run_report.md](track_a_three_cost_run_report.md), [docs/phase1/track_a_supplementary_findings.md](track_a_supplementary_findings.md). | Phase 1 confirms the predictor-to-real-latent link at larger sample size. This does not imply the latent objective is task-aligned. |
| "Planner is not the dominant bottleneck." | FALSIFIED | V3 oracle CEM beats latent CEM_late by `+69.17` pp in D3xR0 and by `+29.81` pp over the D3 row; V1 further improves all 16 cells over V3. Evidence: [docs/phase1/oracle_v3_row_comparison.md](oracle_v3_row_comparison.md), [docs/phase1/oracle_full_variant_comparison.md](oracle_full_variant_comparison.md). | The global Phase 0 claim is false in high-displacement regimes. The revised view is that the latent planning loop has a predictor-sharpness/objective-ranking bottleneck in D2-D3, especially D3. |
| "Encoder goal geometry fails under large physical displacement." | FALSIFIED | Row Pearson(C_real_z, C_real_state) increases with D row: D0=`0.323`, D1=`0.393`, D2=`0.451`, D3=`0.630`; D3 is the best-aligned row. Evidence: [docs/phase1/track_a_supplementary_findings.md](track_a_supplementary_findings.md). | The broad displacement-axis framing is wrong. Phase 1 still finds heterogeneous encoder-rho failures, but not a monotonic large-displacement encoder collapse. |
| "Failure correlates with required rotation." | REVISED | R3 latent success is low in several cells, but V1 reaches D0xR3=`66.7%`, D1xR3=`50.8%`, D2xR3=`71.4%`, D3xR3=`99.3%`. Evidence: [docs/phase1/oracle_full_variant_comparison.md](oracle_full_variant_comparison.md). | Rotation remains associated with hard cases and sign reversals, but Track A shows cost shape can turn many R3 cases from apparent failures into successes. |
| "Per-pair encoder-physics correlation is weak and heterogeneous (mean 0.353, std 0.486)." | HOLDS | Track A DP1: mean rho=`0.333`, std=`0.477`, 95% std CI=`[0.412, 0.529]`, threshold=`0.300`, verdict=`pass`. Evidence: [docs/phase1/track_a_analysis_report.md](track_a_analysis_report.md). | Phase 1 confirms this as a real property of the task/evaluation setting, not random-sampling noise from the 30-pair Phase 0 sample. |
| "Case B/E hybrid classification: encoder geometry failure with event-localized heterogeneity." | REVISED | V1/V2/V3 show separable predictor-sharpness and cost-criterion-alignment axes; all_fail + strong_rho adds a previously hidden quadrant of `16` pairs. Evidence: [docs/phase1/track_a_summary.md](track_a_summary.md), [docs/phase1/track_a_supplementary_findings.md](track_a_supplementary_findings.md). | Case B/E is no longer adequate as a single binding classification. Heterogeneity still holds, but the dominant mechanisms are multi-bottleneck rather than encoder-only. |
| "Most diagnostic pairs contain successful action candidates, so the task/action space is often feasible." | HOLDS | V1 reaches >=`95%` CEM_late success in 6/16 cells, including D3xR0=`99.2%` and D3xR3=`99.3%`. Evidence: [docs/phase1/oracle_full_variant_comparison.md](oracle_full_variant_comparison.md). | Phase 1 strengthens the feasibility point at offset 50. Oracle planners are privileged, so this is not a deployable-policy claim. |
| "CEM_late improves over random, so CEM search is doing useful work." | HOLDS | Track A latent source success: smooth_random=`6.2%`, CEM_early=`18.6%`, CEM_late=`33.6%`. Evidence: [docs/phase1/track_a_three_cost_run_report.md](track_a_three_cost_run_report.md). | CEM search is useful, but useful search does not rule out planner-objective bottlenecks. |
| "Do not spend Phase 1 effort on planner redesign or rollout-loss work as the primary intervention." | REVISED | V3/V1 oracle results show planner objective and cost shape materially change success; V1 beats V3 in `16/16` cells and V1 vs V2 differs by >5 pp in `14/16` cells. Evidence: [docs/phase1/oracle_full_variant_comparison.md](oracle_full_variant_comparison.md). | Track A does not prescribe interventions, but the Phase 0 exclusion of planner/objective work is too strong. |
| "OGBench-Cube is a separate horizon-independent baseline representation limitation." | HOLDS | Phase 1 Track A did not test Cube. Phase 0 Cube sweep remains `68%`, `50%`, `58%`, `50%` across offsets 25/50/75/100. Evidence remains [results/phase0_report.md](../../results/phase0_report.md). | No Phase 1 evidence revises Cube, environment, dataset, or checkpoint facts. |

## 3. Net Effect on Case Classification

The Phase 0 Case B/E classification is no longer adequate. Phase 1 supports a multi-bottleneck picture: predictor-sharpness/objective-ranking issues dominate high-displacement oracle-vs-latent gaps, cost-criterion misalignment explains much of the D0/D1 low-rotation latent edge over V3, and V2 shows that success alignment without a useful outside-region cost shape can break CEM. This document does not propose a new official case taxonomy.

## 4. What Does Not Change

- PushT baseline reproduction remains near the paper reference point: Phase 0 observed about `98%` at the short baseline and `96%` in the offset-25 sweep.
- PushT long-goal degradation remains real in the original latent evaluation: `96% -> 58% -> 16% -> 10%` over offsets 25/50/75/100.
- OGBench-Cube findings are unchanged because Phase 1 Track A did not test Cube.
- Environment, dataset, checkpoint, model-size, and setup facts from Phase 0 remain unchanged.
- Predictor-to-real-latent fidelity remains supported by Phase 1.

## 5. How To Read Together

Future readers should read the Phase 0 originals first for the original framing, baseline reproduction, 30-pair attribution, and cross-environment context. Then read this revisions document and [docs/phase1/track_a_summary.md](track_a_summary.md) for the 100-pair stratified evidence and oracle-ablation results that revise the Phase 0 interpretation.
