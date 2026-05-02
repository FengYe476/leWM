# Track A Sampling Report

Sampler output: `results/phase1/track_a_pairs.json`

Seed: `0`

Git commit at sample time: `3ac6afa93c98a533e904b32fede8b130a4f38fbb`

Dataset: `/Users/fengye/Desktop/Project/leWM/lewm-failure-audit/stablewm_cache/pusht_expert_train.h5`

Dataset rows / episodes: `2,336,736` rows / `18,685` episodes

Offset: `50`

## Realized Cell Counts

| Cell | Target | Eligible pool size | After episode dedup size | Actually sampled | Capped | Cap reason |
| --- | ---: | ---: | ---: | ---: | :---: | --- |
| D0xR0 | 6 | 102744 | 4520 | 6 | no |  |
| D0xR1 | 6 | 25563 | 2162 | 6 | no |  |
| D0xR2 | 6 | 14982 | 1837 | 6 | no |  |
| D0xR3 | 6 | 25237 | 1764 | 6 | no |  |
| D1xR0 | 6 | 148695 | 10329 | 6 | no |  |
| D1xR1 | 6 | 111627 | 9579 | 6 | no |  |
| D1xR2 | 6 | 46047 | 5439 | 6 | no |  |
| D1xR3 | 6 | 84455 | 4792 | 6 | no |  |
| D2xR0 | 6 | 152911 | 9715 | 6 | no |  |
| D2xR1 | 6 | 176179 | 12320 | 6 | no |  |
| D2xR2 | 7 | 118944 | 9314 | 7 | no |  |
| D2xR3 | 7 | 83412 | 6319 | 7 | no |  |
| D3xR0 | 6 | 38092 | 2623 | 6 | no |  |
| D3xR1 | 7 | 59305 | 4408 | 7 | no |  |
| D3xR2 | 6 | 81440 | 5653 | 6 | no |  |
| D3xR3 | 7 | 132954 | 5061 | 7 | no |  |
| **Total** | **100** | **1402587** | **95835** | **100** | **0 capped** |  |

## Capped Cells

No cells were capped. There are no `empty_pool` cells.

## Interpretation

Stratification succeeded: every cell in the accepted `4 x 4` displacement/rotation grid had a large eligible pool, and the per-episode per-cell deduped pool was still far above the target in all cells. The realized sample is exactly `100 / 100`, with no coverage gaps and no literal white-square cells expected in the Track A heatmap. This is close enough to proceed to the separate three-cost evaluation pass.
