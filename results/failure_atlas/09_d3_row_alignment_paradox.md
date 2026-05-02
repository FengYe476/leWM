# 09. D3 Row Alignment Paradox

## Definition

The D3 row alignment paradox is the measurement that the highest-displacement row has the strongest row-wise encoder-physics alignment, even though it is where latent CEM success is poor and oracle CEM gains are largest. This is not a failure mode by itself; it is an atlas entry because it directly falsifies the simple Phase 0 framing that encoder geometry monotonically fails under large displacement.

## Regimes Affected

- D-row comparison across D0, D1, D2, and D3.
- Most visible in D3xR0, D3xR1, D3xR2, D3xR3.
- Representative pair IDs: `74`, `78`, `86`, `93`, `99`.

## Key Numbers

- Row Pearson(C_real_z, C_real_state): D0=`0.323`, D1=`0.393`, D2=`0.451`, D3=`0.630`.
- D3 mean per-cell Spearman values are positive in all four cells: D3xR0=`0.667`, D3xR1=`0.726`, D3xR2=`0.581`, D3xR3=`0.401`.
- Despite this, D3 latent CEM_late is only `17.9%` row-wide.
- D3 V3 oracle CEM_late rises to `47.7%`, and V1 reaches D3xR0=`99.2%` and D3xR3=`99.3%`.

## Evidence

- [docs/phase1/track_a_three_cost_run_report.md](../../docs/phase1/track_a_three_cost_run_report.md)
- [docs/phase1/track_a_supplementary_findings.md](../../docs/phase1/track_a_supplementary_findings.md)
- [docs/phase1/oracle_v3_row_comparison.md](../../docs/phase1/oracle_v3_row_comparison.md)
- [docs/phase1/oracle_full_variant_comparison.md](../../docs/phase1/oracle_full_variant_comparison.md)

## See Also

- [docs/phase1/track_a_summary.md](../../docs/phase1/track_a_summary.md)
- [docs/phase1/phase0_revisions.md](../../docs/phase1/phase0_revisions.md)
