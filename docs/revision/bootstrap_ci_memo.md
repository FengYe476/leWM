# Bootstrap Confidence Intervals

Generated: `2026-05-05T23:53:18.516831+00:00`
Bootstrap resamples: 10000

## Headline Metrics

| Metric | Estimate | 95% CI | n | Method |
|---|---:|---:|---:|---|
| Rpool(C_model), PushT same-pool | 0.084 | [0.037, 0.131] | 100 | percentile bootstrap over pairs |
| Rpool(C_V3), PushT same-pool | 0.258 | [0.146, 0.365] | 100 | percentile bootstrap over pairs |
| Rpool(C_V1), effective | 0.666 | [0.595, 0.735] | 100 | percentile bootstrap over pairs |
| Delta_CEM = Rendpoint - Rpool(C_model) | 0.422 | [0.374, 0.468] | 100 | percentile bootstrap over pairs |
| MPPI - CEM Rpool(C_model) | 0.079 | [-0.044, 0.204] | 30 | paired percentile bootstrap over matched MPPI/CEM pairs |
| CEM-specific attribution fraction | 22.2% | [-15.7%, 48.1%] | 30 | paired percentile bootstrap of (Rpool_MPPI - Rpool_CEM) / (Rendpoint - Rpool_CEM) |

Endpoint references:
- Delta_CEM reference: 0.506 (results/phase2/stage1/stage1a_full.json controls.C0.metrics.global_spearman)
- MPPI attribution reference: 0.470 (results/revision/mppi_pool_analysis.json decision.endpoint_reference_R)

## Repair Failures

| Repair | Estimate | 95% CI | n | Method |
|---|---:|---:|---:|---|
| Cost head hard-pair success | 0.0% | [0.0%, 20.6%] | 16 | Clopper-Pearson exact 95.0% binomial interval |
| Subspace-CEM regression among seed-pairs | 10.0% | [3.8%, 20.5%] | 60 | Clopper-Pearson exact 95.0% binomial interval |
| Subspace-CEM regression among discordant pairs | 85.7% | [42.1%, 99.6%] | 7 | Clopper-Pearson exact 95.0% binomial interval |

Subspace-CEM exact McNemar/binomial p-value: 0.125
Subspace-CEM success delta (subspace minus default): -8.3%
