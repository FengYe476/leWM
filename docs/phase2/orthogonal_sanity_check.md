# Orthogonal Projection Sanity Check

Generated: `2026-05-03T17:29:06.420904+00:00`
Git commit: `adfd5053584f73decea35500b726fe2a62202ae9`
Overall status: **PASS**

This is an offline Block 2.1 check. It reads saved PushT pool tensors and does not load a simulator, GPU, checkpoint, or policy.

## Orthogonal Identity Check

Pool dir: `results/phase2/protocol_match/pusht_pools`
Seed: `0`
Max `|Q^T Q - I|`: `1.332e-15`
Rank-1 match rate: `1.000`

| Pair | Saved Rank-1 | Default Argmin | Orthogonal Argmin | Max Identity-Default | Max Orthogonal-Default | Rank Match | Pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 147 | 147 | 147 | 5.084e-06 | 5.084e-06 | PASS | PASS |
| 1 | 106 | 106 | 106 | 1.175e-05 | 1.175e-05 | PASS | PASS |
| 2 | 232 | 232 | 232 | 2.341e-06 | 2.341e-06 | PASS | PASS |
| 3 | 12 | 12 | 12 | 5.482e-06 | 5.482e-06 | PASS | PASS |
| 4 | 135 | 135 | 135 | 1.358e-05 | 1.358e-05 | PASS | PASS |

## Gaussian m=192 Smoke

Source: `results/phase2/protocol_match/pusht_rerank_only.json`

| Protocol | Default Success | Gaussian m=192 Success | Gap pp | Tolerance pp | Pass |
| --- | --- | --- | --- | --- | --- |
| PushT re-rank-only | 35.00% | 34.00% | -1.00 | 2.00 | PASS |

## Cube Note

Cube orthogonal tensor sanity is skipped because no Cube pool `.pt` artifacts equivalent to PushT `pusht_pools/pair_*.pt` are currently available. The Cube m=192 Gaussian gap remains a JSON-level smoke comparison, not an orthogonal identity check.
