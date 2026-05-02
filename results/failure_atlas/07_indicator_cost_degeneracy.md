# 07. Indicator Cost Degeneracy

## Definition

Indicator cost degeneracy occurs when the V2 oracle cost gives CEM too little ordering signal outside the success region: candidates score `1.0` until they already satisfy the binary criterion. In this PushT offset-50 setup, the stored elite-cost proxy shows many pairs where all selected early and late V2 elites still scored `1.0`, and V2 success drops sharply below V3 in several cells.

## Regimes Affected

- V2 below V3 by more than 10 pp: D1xR2, D2xR1, D3xR0, D3xR1, D3xR2, D3xR3.
- V2 selected-elite all-1.0 proxy touches 60 pairs across many cells, concentrated in harder D2/D3 and R3 regimes.
- Representative pair IDs: `36`, `54`, `87`, `93`, `99`.

## Key Numbers

- V2 selected-elite degeneracy proxy: `60` pairs with all stored early/late elite costs equal to `1.0`.
- D1xR2 V3=`79.2%`, V2=`33.3%`, delta=`-45.83` pp.
- D3xR1 V3=`44.3%`, V2=`14.3%`, delta=`-30.00` pp.
- D3xR2 V3=`27.5%`, V2=`0.0%`, delta=`-27.50` pp.
- V1 and V2 differ by more than 5 pp in `14/16` cells, always with V1 higher.

## Evidence

- [docs/phase1/oracle_full_variant_comparison.md](../../docs/phase1/oracle_full_variant_comparison.md)
- [results/phase1/v2_oracle_ablation/v2_d1.json](../phase1/v2_oracle_ablation/v2_d1.json)
- [results/phase1/v2_oracle_ablation/v2_d2.json](../phase1/v2_oracle_ablation/v2_d2.json)
- [results/phase1/v2_oracle_ablation/v2_d3.json](../phase1/v2_oracle_ablation/v2_d3.json)

## See Also

- [docs/phase1/track_a_summary.md](../../docs/phase1/track_a_summary.md)
- [docs/phase1/phase0_revisions.md](../../docs/phase1/phase0_revisions.md)
