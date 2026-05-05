# MPPI Comparison Memo

## Main Comparison

| Metric | CEM default | MPPI (tau=1.0) |
|---|---:|---:|
| Planning success (30 pairs) | 53.3% | 38.9% |
| Rpool(C_model) mean | 0.115 | 0.194 |
| Rpool(V1) mean | 0.537 | 0.707 |
| Pool C_real_state std mean | 7.947 | 29.478 |
| Pool success mass mean | 55.2% | 27.8% |

## Decision

**Endpoint-planning decoupling is partially optimizer-general and partially CEM-specific.**

CEM hard truncation compresses pool diversity by 3.7x relative to MPPI, reducing Rpool(C_model) from 0.194 to 0.115. However, even MPPI retains only 41% of the endpoint ranking signal (Rpool/Rendpoint = 0.194/0.47), confirming that most of the decoupling is representation-driven rather than optimizer-specific.

## Pool Diversity Comparison

CEM pool C_real_state std is 7.947, while MPPI pool C_real_state std is 29.478. This is a 3.7x difference, directly showing that CEM hard truncation compresses the candidate pool into a physically narrow region.

MPPI preserves more physical diversity because soft weighting does not discard any candidates from the sampled pool. That broader physical spread also raises Rpool(C_model), but it does not restore endpoint-level rank preservation.

## Decomposed Attribution

Using R_endpoint approximately 0.47 from the paper's PushT endpoint-ranking table, the matched 30-pair pool correlations are:

| Quantity | Value |
|---|---:|
| R_endpoint | 0.470 |
| Rpool(C_model), CEM | 0.115 |
| Rpool(C_model), MPPI | 0.194 |
| MPPI improvement over CEM | 0.079 |
| CEM-specific recovered endpoint-pool gap | 22.2% |
| MPPI retained endpoint signal | 41.3% |

Both CEM and MPPI remain far below R_endpoint, so endpoint-planning decoupling exists under both optimizers. The CEM to MPPI improvement of 0.079 recovers about `(0.194 - 0.115) / (0.47 - 0.115) = 22.2%` of the CEM endpoint-pool gap. The remaining gap is therefore primarily representation-driven, while CEM's hard truncation contributes an additional search-dynamics component.

Rpool(V1) is higher under MPPI than CEM (0.707 vs 0.537), consistent with the V1 oracle being a better local ranker in both optimizer pools.

## Planning Success Note

MPPI planning success (38.9%) is lower than CEM (53.3%), consistent with MPPI's slower convergence under 30 iterations and the sweep finding that tau=1.0 has ESS near 1.1. This does not invalidate the pool-level comparison because the diagnostic targets pool ranking geometry, not absolute planning performance.

Per subset, the invisible-quadrant pairs have 0% planning success under both matched CEM and MPPI, confirming that these pairs are fundamentally hard regardless of optimizer.

## Notes

- MPPI metrics are computed per seed, averaged across the 3 seeds for each pair, then averaged across the 30 pairs.
- CEM default metrics are the matched 30 pair rows from the Phase B Rpool(V1) artifact.
