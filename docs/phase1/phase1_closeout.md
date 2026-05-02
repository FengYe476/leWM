# Phase 1 Track A Closeout

## 1. What Phase 1 Was

Phase 1 was the Track A subset of the original four-track Phase 1 proposal: stratified PushT evidence before any intervention work. Track A was chosen first because it was the only track runnable without RTX 5090 hardware and because Tracks B, C, and D all assume the bottleneck has already been localized well enough to know what a replacement encoder, calibration ladder, or retraining pass should test. Tracks B, C, and D were not started in Phase 1. The originating plan is [docs/phase1/phase1_proposal.md](phase1_proposal.md).

## 2. Why Phase 1 Stopped Here

Phase 1 Track A is complete enough to close because:

- The offset-50 `(D, R)` failure surface is mapped across latent, V3, V1, and V2 cost variants.
- The two-bottleneck interpretation is supported by oracle ablations rather than by a single correlation axis.
- The Phase 0 conclusions that motivated immediate Track B and Track D work have been weakened; Track A was the prerequisite evidence they needed.
- All five per-trajectory anchor pairs yielded rollouts consistent with the existing aggregate JSONs after the cleanup pass.
- The cleanup pass closed the known consolidation issues: broken smoke links, the D2xR0 mechanism-map label, atlas page 09 arithmetic, D3xR2 atlas coverage, and the F6 anchor pair.

Phase 1 does not claim to explain every cell. R3 residual failures under V1 in some cells, the D0xR1/D1xR0 latent residual edge after V1, and the sign-reversal cluster are partially characterized rather than fully explained.

## 3. Evidence Chain: Recommended Reading Order

| # | Document | Why Here |
|---:|---|---|
| 1 | [README.md](../../README.md) | Repository overview, setup, and current status. |
| 2 | [docs/research_plan.md](../research_plan.md) | Phase 0 framing and frozen exclusions before Track A existed. |
| 3 | [results/phase0_report.md](../../results/phase0_report.md) | The original Phase 0 interpretation as it stood. |
| 4 | [docs/phase1/phase1_proposal.md](phase1_proposal.md) | Why Phase 1 existed and why Track A came first. |
| 5 | [docs/phase1/track_a_stratification_proposal.md](track_a_stratification_proposal.md) | The locked 4x4 grid design. |
| 6 | [docs/phase1/track_a_summary.md](track_a_summary.md) | Canonical Phase 1 Track A findings; read this before details. |
| 7 | [docs/phase1/phase0_revisions.md](phase0_revisions.md) | Which Phase 0 claims hold, changed, or failed. |
| 8 | [docs/phase1/oracle_full_variant_comparison.md](oracle_full_variant_comparison.md) | The 16-cell latent/V3/V1/V2 matrix and ablation evidence. |
| 9 | [results/failure_atlas/05_predictor_sharpness_bottleneck.md](../../results/failure_atlas/05_predictor_sharpness_bottleneck.md) through [results/failure_atlas/09_d3_row_alignment_paradox.md](../../results/failure_atlas/09_d3_row_alignment_paradox.md) | Mechanism pages added by Phase 1. |
| 10 | [docs/phase1/per_trajectory_catalog.md](per_trajectory_catalog.md) | Visual anchors for the headline mechanisms. |
| 11 | [docs/phase1/track_a_supplementary_findings.md](track_a_supplementary_findings.md) | Failure quadrant and D-row cost-diagnosis details. |
| 12 | [docs/progress_log.md](../progress_log.md) | Chronological development trail and acceptance checkpoints. |

## 4. What Phase 1 Chose Not To Do

Phase 0 frozen exclusions, re-read after Track A:

| Phase 0 Frozen Exclusion | Closeout Status | Context |
|---|---|---|
| No model retraining | Partially defensible — context | Phase 1 did not establish a retraining target; as a permanent exclusion, it remains unresolved. |
| No architecture changes | Partially defensible — context | Phase 1 did not test architecture changes, but a blanket architecture exclusion is too broad for future planning. |
| No planner redesign beyond fixed evaluation settings | No longer defensible after Track A | Phase 1 shows that planner objective and cost shape are part of the bottleneck surface. |
| No broad hyperparameter search | Still defensible after Track A | Phase 1 narrowed mechanisms enough that broad untargeted sweeps would still be poor evidence. |
| No task expansion beyond the planned environments | Partially defensible — context | Phase 1 is PushT-only; generality beyond the planned environments remains untested. |
| No claim of final causal proof from a single metric | Still defensible after Track A | Track A reinforces this guardrail by relying on stratification, ablations, and trajectory checks together. |

Phase 1 also kept its own scope boundaries:

- no encoder replacement
- no calibration ladder
- no retraining
- no offset-75 or offset-100 evaluation under any cost variant
- no Cube evaluation under any cost variant
- no per-trajectory visualization beyond the five anchor pairs
- no exploration of the sign-reversal list beyond cataloging it
- no investigation of why some D0xR1 pairs are trivially solvable while others are F6 cases

## 5. Open Questions, By Confidence Level

### 5a. Well-Supported By Phase 1 Evidence

- What property distinguishes a D0xR1 trivial pair from a D0xR1 F6 pair? Pair 6 and pair 8 make this question concrete, but Phase 1 did not characterize the physical difference.
- Is the predictor-sharpness bottleneck specific to this LeWM predictor stack or a more generic JEPA-rollout limitation? Phase 1 localizes the issue in LeWM only.
- When a cell improves under V1 but not V3, which part of the conjunctive success criterion is doing the work? Phase 1 identifies the distinction but does not decompose it further.

### 5b. Suggested By Phase 1 Evidence

- The sign-reversal list may contain multiple subtypes rather than one mechanism.
- R3 residual failures may mix cost-shape, rotation, and trajectory-contact issues.
- The spread between V1 and V2 suggests that success alignment alone is not the whole story; outside-region cost shape matters in at least some cells.

### 5c. Speculative — Not Addressed By Phase 1

- Whether any Track A mechanism transfers to OGBench-Cube.
- Whether offset 75 or offset 100 would preserve the same cell labels.
- Whether a different encoder, predictor, planner objective, or retraining run would remove the observed bottlenecks.

## 6. What Phase 2 Starts With

These are entry points for planning, not a plan:

- Read first: [docs/phase1/track_a_summary.md](track_a_summary.md).
- Binding evidence to keep in view: the 16-cell latent/V3/V1/V2 matrix in [docs/phase1/oracle_full_variant_comparison.md](oracle_full_variant_comparison.md).
- Phase 0 conclusion still alive: predictor-to-real-latent fidelity remains supported, as recorded in [docs/phase1/phase0_revisions.md](phase0_revisions.md).

## 7. Verification Checklist For Future Readers

- Verify the 100-pair grid: `python -c "import json,collections; ps=json.load(open('results/phase1/track_a_pairs.json'))['pairs']; print(len(ps), collections.Counter(p['cell'] for p in ps))"`
- Verify DP1 from source JSON: `rg -n '"verdict"|"ci_low_std"|"ci_high_std"|"threshold"' results/phase1/track_a_analysis/track_a_analysis.json`
- Verify the oracle variant matrix: inspect [docs/phase1/oracle_full_variant_comparison.md](oracle_full_variant_comparison.md#3-full-16-cell-variant-matrix).
- Verify the D2xR0 cleanup: `rg -n "D2xR0|D2 \\|" docs/phase1/track_a_summary.md docs/phase1/oracle_full_variant_comparison.md`
- Verify the F6 visual anchor: inspect [results/phase1/figures/per_trajectory/8_D0xR1_cost_panel.png](../../results/phase1/figures/per_trajectory/8_D0xR1_cost_panel.png).
- Verify the D3 alignment paradox entry: inspect [results/failure_atlas/09_d3_row_alignment_paradox.md](../../results/failure_atlas/09_d3_row_alignment_paradox.md).
- Verify per-trajectory provenance: `ls results/phase1/per_trajectory/*_D0xR1_*`
