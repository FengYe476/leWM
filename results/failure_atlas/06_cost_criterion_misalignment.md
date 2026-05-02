# 06. Cost-Criterion Misalignment

## Definition

Cost-criterion misalignment is the regime where the smooth V3 oracle cost, `block_pos_dist + angle_dist`, underperforms the latent planner because V3 is not equivalent to PushT's conjunctive success criterion, `block_pos_dist < 20` and `angle_dist < pi/9`. The V1 hinge cost is more success-aligned and closes much of the gap, showing that the scalar oracle cost itself can be the bottleneck.

## Regimes Affected

- Core latent-favorable cells: D0xR0, D0xR1, D1xR0.
- Additional cost-shape-sensitive cells: D0xR3, D1xR3, D2xR3, D3xR3.
- Representative pair IDs: `6`, `9`, `24`, `29`, `42`.

## Key Numbers

- D0xR1 latent CEM_late=`91.7%`, V3=`41.7%`, V1=`84.2%`.
- D0xR0 latent=`100.0%`, V3=`82.5%`, V1=`100.0%`.
- D1xR0 latent=`50.8%`, V3=`16.7%`, V1=`33.3%`.
- V1 beats V3 in `16/16` cells; V1 closes D0xR0/D0xR1/D1xR0 gaps from `-17.50/-50.00/-34.17` pp to `0.00/-7.50/-17.50` pp relative to latent.

## Evidence

- [docs/phase1/oracle_v3_row_comparison.md](../../docs/phase1/oracle_v3_row_comparison.md)
- [docs/phase1/oracle_full_variant_comparison.md](../../docs/phase1/oracle_full_variant_comparison.md)
- [results/phase1/v1_oracle_ablation/v1_d0.json](../phase1/v1_oracle_ablation/v1_d0.json)
- [results/phase1/v1_oracle_ablation/v1_d1.json](../phase1/v1_oracle_ablation/v1_d1.json)

## See Also

- [docs/phase1/track_a_summary.md](../../docs/phase1/track_a_summary.md)
- [docs/phase1/phase0_revisions.md](../../docs/phase1/phase0_revisions.md)
