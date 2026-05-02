# 05. Predictor-Sharpness Bottleneck

## Definition

The predictor-sharpness bottleneck is the regime where latent CEM does not separate elite candidate actions well enough under the learned latent planning objective, while an oracle planner using ground-truth terminal state cost does separate them. This is not evidence that the encoder is fine in all regimes; it says that replacing the latent planning loop with oracle state-cost CEM sharply improves CEM_late success in specific high-displacement cells.

## Regimes Affected

- Primary cells: D3xR0, D3xR1, D3xR2, D3xR3.
- Secondary cells with positive V3-vs-latent signal: D2xR1, D2xR2, D1xR1, D1xR2, D0xR2.
- Representative pair IDs for D3 evidence: `74`, `78`, `87`, `93`, `99`.

## Key Numbers

- D3xR0 latent CEM_late=`16.7%`, V3 oracle CEM_late=`85.8%`, delta=`+69.17` pp.
- D3 row latent CEM_late=`17.9%`, V3 oracle CEM_late=`47.7%`, delta=`+29.81` pp.
- V1 hinge oracle reaches D3xR0=`99.2%` and D3xR3=`99.3%`.
- V3 oracle-favorable cells include the full D3 row and several D1/D2 R1/R2 cells.

## Evidence

- [docs/phase1/oracle_v3_row_comparison.md](../../docs/phase1/oracle_v3_row_comparison.md)
- [docs/phase1/oracle_full_variant_comparison.md](../../docs/phase1/oracle_full_variant_comparison.md)
- [results/phase1/d3_oracle_ablation/d3_oracle_V3.json](../phase1/d3_oracle_ablation/d3_oracle_V3.json)
- [results/phase1/v1_oracle_ablation/v1_d3.json](../phase1/v1_oracle_ablation/v1_d3.json)

## See Also

- [docs/phase1/track_a_summary.md](../../docs/phase1/track_a_summary.md)
- [docs/phase1/phase0_revisions.md](../../docs/phase1/phase0_revisions.md)
