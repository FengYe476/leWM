# Track A Consolidated Summary

## 1. Track A Goal Recap

Track A set out to turn the Phase 0 PushT offset-50 diagnosis from a 30-pair random sample into a stratified, publication-grade characterization of when LeWM's latent planning objective agrees or disagrees with task success. Before running, the design was locked to a 100-pair displacement x rotation grid, offset `50`, `80` actions per pair with a `20/20/20/20` split across data, smooth random, CEM_early, and CEM_late, and fixed CEM hyperparameters of `300/30/30`, horizon `5`, receding horizon `5`, and action_block `5`. The original motivation is in [docs/phase1/phase1_proposal.md](phase1_proposal.md), and the stratified sampling plan is in [docs/phase1/track_a_stratification_proposal.md](track_a_stratification_proposal.md).

## 2. Evidence Inventory

### Sampling

1. [results/phase1/track_a_pairs.json](../../results/phase1/track_a_pairs.json): sampled 100 offset-50 PushT initial-goal pairs over the 4x4 displacement/rotation grid.
2. [docs/phase1/track_a_sampling_report.md](track_a_sampling_report.md): realized cell counts, eligible pool sizes, and confirmation that no cell was capped or empty.

### Latent CEM Main Run

3. [results/phase1/track_a_three_cost.json](../../results/phase1/track_a_three_cost.json): full Track A latent-CEM three-cost run, 100 pairs x 80 actions.
4. [docs/phase1/track_a_three_cost_run_report.md](track_a_three_cost_run_report.md): run provenance, per-cell success/correlation table, and global source success rates.
5. [results/phase1/track_a_three_cost_smoke.json](../../results/phase1/track_a_three_cost_smoke.json): initial smoke run.
6. [results/phase1/track_a_three_cost_smoke_hard.json](../../results/phase1/track_a_three_cost_smoke_hard.json): hard-cell smoke run.
7. [results/phase1/track_a_analysis/track_a_analysis.json](../../results/phase1/track_a_analysis/track_a_analysis.json): DP1 statistics, per-cell tables, and figure paths.
8. [docs/phase1/track_a_analysis_report.md](track_a_analysis_report.md): DP1 result, heatmaps, and sign-reversal summary.

### Aggregate Analyses

9. [results/phase1/track_a_analysis/track_a_sign_reversal_pairs.json](../../results/phase1/track_a_analysis/track_a_sign_reversal_pairs.json): pairs with negative per-pair Spearman rho.
10. [results/phase1/track_a_analysis/failure_mode_decomposition.json](../../results/phase1/track_a_analysis/failure_mode_decomposition.json): all_fail/some_succ by encoder-rho quadrant.
11. [results/phase1/track_a_analysis/d_row_cost_diagnosis.json](../../results/phase1/track_a_analysis/d_row_cost_diagnosis.json): D-row cost magnitudes and row-wise Pearson correlations.
12. [docs/phase1/track_a_supplementary_findings.md](track_a_supplementary_findings.md): failure-mode decomposition and D-row cost-magnitude fact sheet.
13. [results/phase1/figures/track_a/success_rate_heatmap.png](../../results/phase1/figures/track_a/success_rate_heatmap.png): success-rate heatmap.
14. [results/phase1/figures/track_a/mean_pair_spearman_heatmap.png](../../results/phase1/figures/track_a/mean_pair_spearman_heatmap.png): mean per-pair Spearman heatmap.
15. [results/phase1/figures/track_a/std_pair_spearman_heatmap.png](../../results/phase1/figures/track_a/std_pair_spearman_heatmap.png): per-cell Spearman standard-deviation heatmap.
16. [results/phase1/figures/track_a/cell_model_realz_pearson_heatmap.png](../../results/phase1/figures/track_a/cell_model_realz_pearson_heatmap.png): model-vs-real-latent Pearson heatmap.
17. [results/phase1/figures/track_a/sign_reversal_distribution.png](../../results/phase1/figures/track_a/sign_reversal_distribution.png): sign-reversal distribution plot.
18. [results/phase1/figures/track_a/per_cell_rho_distribution.png](../../results/phase1/figures/track_a/per_cell_rho_distribution.png): per-cell rho distribution.
19. [results/phase1/figures/track_a/failure_mode_quadrants.png](../../results/phase1/figures/track_a/failure_mode_quadrants.png): failure quadrant plot.
20. [results/phase1/figures/track_a/failure_mode_counts_grid.png](../../results/phase1/figures/track_a/failure_mode_counts_grid.png): failure mode counts over the 4x4 grid.
21. [results/phase1/figures/track_a/cost_magnitude_by_row.png](../../results/phase1/figures/track_a/cost_magnitude_by_row.png): row-wise cost magnitude distributions.
22. [results/phase1/figures/track_a/best_cost_per_pair_by_row.png](../../results/phase1/figures/track_a/best_cost_per_pair_by_row.png): best observed physical-state cost per pair by D row.
23. [results/phase1/figures/track_a/row_pearson_correlations.png](../../results/phase1/figures/track_a/row_pearson_correlations.png): row-wise Pearson correlations among costs.

### Oracle CEM Ablation

24. [results/phase1/d0_oracle_ablation/d0_oracle_V3.json](../../results/phase1/d0_oracle_ablation/d0_oracle_V3.json): D0 oracle CEM with V3 cost.
25. [results/phase1/d1_oracle_ablation/d1_oracle_V3.json](../../results/phase1/d1_oracle_ablation/d1_oracle_V3.json): D1 oracle CEM with V3 cost.
26. [results/phase1/d2_oracle_ablation/d2_oracle_V3.json](../../results/phase1/d2_oracle_ablation/d2_oracle_V3.json): D2 oracle CEM with V3 cost.
27. [results/phase1/d3_oracle_ablation/d3_oracle_V3.json](../../results/phase1/d3_oracle_ablation/d3_oracle_V3.json): D3 oracle CEM with V3 cost.
28. [docs/phase1/oracle_v3_row_comparison.md](oracle_v3_row_comparison.md): latent vs V3 oracle row comparison and heatmaps.
29. [results/phase1/v1_oracle_ablation/v1_d0.json](../../results/phase1/v1_oracle_ablation/v1_d0.json) through [results/phase1/v1_oracle_ablation/v1_d3.json](../../results/phase1/v1_oracle_ablation/v1_d3.json): V1 hinge oracle CEM, CEM-only outputs for all rows.
30. [results/phase1/v2_oracle_ablation/v2_d0.json](../../results/phase1/v2_oracle_ablation/v2_d0.json) through [results/phase1/v2_oracle_ablation/v2_d3.json](../../results/phase1/v2_oracle_ablation/v2_d3.json): V2 indicator oracle CEM, CEM-only outputs for all rows.
31. [docs/phase1/oracle_full_variant_comparison.md](oracle_full_variant_comparison.md): latent/V3/V1/V2 matrix, sanity checks, and cost-shape sensitivity.
32. [results/phase1/figures/track_a/oracle_vs_latent_delta_heatmap.png](../../results/phase1/figures/track_a/oracle_vs_latent_delta_heatmap.png): V3-minus-latent CEM_late delta heatmap.
33. [results/phase1/figures/track_a/v3_vs_v1_delta_heatmap.png](../../results/phase1/figures/track_a/v3_vs_v1_delta_heatmap.png): V1-minus-V3 CEM_late delta heatmap.
34. [results/phase1/figures/track_a/v3_vs_v2_delta_heatmap.png](../../results/phase1/figures/track_a/v3_vs_v2_delta_heatmap.png): V2-minus-V3 CEM_late delta heatmap.
35. [results/phase1/figures/track_a/v1_vs_v2_delta_heatmap.png](../../results/phase1/figures/track_a/v1_vs_v2_delta_heatmap.png): V2-minus-V1 CEM_late delta heatmap.
36. [results/phase1/figures/track_a/full_variant_success_grid.png](../../results/phase1/figures/track_a/full_variant_success_grid.png): four-variant success grid.

## 3. Headline Findings

**F1. Stratified sampling produced a balanced grid with no empty cells.**  
The final sample is `100/100` pairs across the 4x4 D/R grid, every cell has 6 or 7 pairs, and no cell was capped. Evidence: [results/phase1/track_a_pairs.json](../../results/phase1/track_a_pairs.json), [docs/phase1/track_a_sampling_report.md](track_a_sampling_report.md). Applies to all 16 cells.

**F2. DP1 passes, so Phase 0's high per-pair correlation variance was not random-sampling noise.**  
Track A measured per-pair Spearman std `0.477` with 95% bootstrap CI `[0.412, 0.529]`, above the `0.300` threshold; Phase 0 was mean `0.353`, std `0.486`. Evidence: [docs/phase1/track_a_analysis_report.md](track_a_analysis_report.md), [results/phase1/track_a_analysis/track_a_analysis.json](../../results/phase1/track_a_analysis/track_a_analysis.json). Applies globally across the 100-pair grid.

**F3. Failure decomposition reveals an invisible quadrant: all_fail with strong encoder rho.**  
There are `16` all_fail + strong_rho pairs, with mean displacement `121.51` px and mean rotation `1.325` rad; this quadrant touches D1xR0, D1xR3, D2xR2, D2xR3, and every D3 cell. Evidence: [docs/phase1/track_a_supplementary_findings.md](track_a_supplementary_findings.md), [results/phase1/track_a_analysis/failure_mode_decomposition.json](../../results/phase1/track_a_analysis/failure_mode_decomposition.json). Applies mainly to D2-D3 plus selected D1 cells.

**F4. Encoder-physics alignment is non-monotonic and inverted by D row.**  
Row Pearson(C_real_z, C_real_state) is D0=`0.323`, D1=`0.393`, D2=`0.451`, D3=`0.630`; D3 has the best row-wise encoder-physics alignment, not the worst. Evidence: [docs/phase1/track_a_supplementary_findings.md](track_a_supplementary_findings.md), [results/phase1/track_a_analysis/d_row_cost_diagnosis.json](../../results/phase1/track_a_analysis/d_row_cost_diagnosis.json). Applies to the D-row displacement axis.

**F5. V3 oracle CEM separates a predictor-sharpness regime from a cost-misalignment regime.**  
V3 beats latent CEM_late by `+29.81` pp on the D3 row and is positive on D2 overall, but underperforms latent in D0xR0 (`-17.50` pp), D0xR1 (`-50.00` pp), and D1xR0 (`-34.17` pp). Evidence: [docs/phase1/oracle_v3_row_comparison.md](oracle_v3_row_comparison.md), V3 row JSONs under [results/phase1/](../../results/phase1/). Applies to D2-D3 for predictor sharpness and D0/D1 low-R for cost misalignment.

**F6. V1 hinge oracle CEM beats V3 in 16/16 cells and largely closes the latent-favorable gaps.**  
D0xR0/D0xR1/D1xR0 move from V3-vs-latent gaps of `-17.50/-50.00/-34.17` pp to V1-vs-latent gaps of `0.00/-7.50/-17.50` pp. Evidence: [docs/phase1/oracle_full_variant_comparison.md](oracle_full_variant_comparison.md), V1 JSONs under [results/phase1/v1_oracle_ablation/](../../results/phase1/v1_oracle_ablation/). Applies to all 16 cells, especially D0/D1 low-R.

**F7. V2 indicator oracle CEM degenerates under this CEM setup.**  
The selected-elite degeneracy proxy flags `60` pairs with all stored early/late V2 elite costs equal to `1.0`; V2 falls below V3 by more than 10 pp in D1xR2, D2xR1, and all four D3 cells. Evidence: [docs/phase1/oracle_full_variant_comparison.md](oracle_full_variant_comparison.md), [results/phase1/v2_oracle_ablation/](../../results/phase1/v2_oracle_ablation/). Applies to cells where the indicator gives CEM little outside-region ordering signal.

**F8. Track A supports two independent bottleneck axes.**  
Predictor sharpness varies mainly along the displacement axis, while cost-criterion alignment varies in low-rotation and high-rotation cells and globally by cost shape; V1 and V2 differ by more than 5 pp in `14/16` cells. Evidence: [docs/phase1/oracle_v3_row_comparison.md](oracle_v3_row_comparison.md), [docs/phase1/oracle_full_variant_comparison.md](oracle_full_variant_comparison.md). Applies across the full 4x4 grid.

**F9. V1 oracle success shows the offset-50 task is often feasible under the right planning loop.**  
V1 reaches at least 95% CEM_late success in 6 cells: D0xR0, D0xR2, D1xR1, D2xR1, D3xR0, and D3xR3; the extreme-displacement cells D3xR0 and D3xR3 reach `99.2%` and `99.3%`. Evidence: [docs/phase1/oracle_full_variant_comparison.md](oracle_full_variant_comparison.md), [results/phase1/v1_oracle_ablation/](../../results/phase1/v1_oracle_ablation/). Applies to offset-50 PushT only.

## 4. Failure Mechanism Map

This map is an evidence-based inference, not a causal proof. Labels summarize the dominant bottleneck visible after Track A; they should be read with the supporting reports, not as a new taxonomy.

| D row \ R bin | R0 | R1 | R2 | R3 |
|---|---|---|---|---|
| D0 | easy / no bottleneck | cost misalignment | both | cost misalignment |
| D1 | cost misalignment | both | both | cost misalignment |
| D2 | both | predictor sharpness | predictor sharpness | cost misalignment |
| D3 | predictor sharpness | predictor sharpness | predictor sharpness | both |

The labels mean: predictor sharpness when switching from latent CEM to oracle state-cost CEM yields clear gains; cost misalignment when V3's scalar cost underperforms the conjunctive success criterion and V1 repairs much of the gap; both when both axes move the numbers materially; easy / no bottleneck when latent CEM already saturates. No final cell is labeled task infeasible because V1's high success in multiple R3 cells, including D3xR3, falsifies a blanket infeasibility reading for offset 50.

Falsifiers are direct. A predictor-sharpness label would weaken if repeated oracle-state CEM no longer beat latent in that cell. A cost-misalignment label would weaken if success-aligned hinge costs failed to close the V3 gap. A both label would weaken if either axis alone explained the observed pattern. An easy label would weaken if a broader candidate set or seed showed substantial failure. A task-infeasible label would require V1 and similarly aligned oracle planners to stay low; Track A did not establish that label.

## 5. Open Questions and Limitations

- Residual D0xR1 and D1xR0 latent-favorable gaps remain after V1: about `7.5` pp and `17.5` pp below latent, respectively.
- All Track A evidence is offset `50`; offsets `75` and `100` may differ.
- All Track A evidence is PushT only; it makes no claim about OGBench-Cube.
- Oracle CEM is an upper bound with privileged simulator state, not a deployable system.
- V1 uses a single alpha, `20/(pi/9)`; alpha sensitivity was not tested.
- V2 evidence applies only to this indicator formulation and this CEM setup; it does not show that all indicator costs are categorically bad.
- Phase 1 Track A did not test encoder replacement (Track B), calibration ladder (Track C), or retraining (Track D).

## 6. Cross-References

- Phase 0 claim audit: [docs/phase1/phase0_revisions.md](phase0_revisions.md)
- V3 oracle row comparison: [docs/phase1/oracle_v3_row_comparison.md](oracle_v3_row_comparison.md)
- Full V1/V2/V3 comparison: [docs/phase1/oracle_full_variant_comparison.md](oracle_full_variant_comparison.md)
- Track A analysis report: [docs/phase1/track_a_analysis_report.md](track_a_analysis_report.md)
- Track A supplementary findings: [docs/phase1/track_a_supplementary_findings.md](track_a_supplementary_findings.md)
- Failure Atlas additions: [results/failure_atlas/05_predictor_sharpness_bottleneck.md](../../results/failure_atlas/05_predictor_sharpness_bottleneck.md), [results/failure_atlas/06_cost_criterion_misalignment.md](../../results/failure_atlas/06_cost_criterion_misalignment.md), [results/failure_atlas/07_indicator_cost_degeneracy.md](../../results/failure_atlas/07_indicator_cost_degeneracy.md), [results/failure_atlas/08_r3_apparent_infeasibility_revision.md](../../results/failure_atlas/08_r3_apparent_infeasibility_revision.md), [results/failure_atlas/09_d3_row_alignment_paradox.md](../../results/failure_atlas/09_d3_row_alignment_paradox.md)
