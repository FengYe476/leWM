# Oracle V3 Row Comparison (D0 + D3)

## 1. Provenance

- Track A reference data: `results/phase1/track_a_three_cost.json`
- D3 oracle V3 data: `results/phase1/d3_oracle_ablation/d3_oracle_V3.json`
- D0 oracle V3 data: `results/phase1/d0_oracle_ablation/d0_oracle_V3.json`
- D3 V3 result commit: `78900c3b439a0f0d5d98f36b2f9aab4449e5cb61`
- D0 V3 run commit: `9cb64aa2c9812c6a9a0b97ecf1897d8bf4ac301a`
- D0 run wall-clock: `807.94` seconds (`13.47` minutes); external `/usr/bin/time` wall-clock was `816.95` seconds.
- Locked parameters: V3 baseline cost only; oracle real-state planner cost; seed 0; CEM 300/30/30, horizon 5, action_block 5; 80 actions per pair with 20/20/20/20 split.
- Note: the D0 cell counts are 6+6+6+6 = 24 pairs in `track_a_pairs.json`, so this run evaluated 24 D0 pairs.

## 2. D0 V3 Sanity Table

| Cell | Track A latent CEM_late | Track A latent overall | V3 oracle CEM_late | V3 oracle overall | Delta CEM_late pp | Delta overall pp | Relative failure reduction CEM_late |
|---|---:|---:|---:|---:|---:|---:|---:|
| D0xR0 | 100.0% | 76.7% | 82.5% | 74.2% | -17.50 | -2.50 | inf |
| D0xR1 | 91.7% | 44.6% | 41.7% | 31.7% | -50.00 | -12.92 | 7.000 |
| D0xR2 | 66.7% | 32.1% | 75.8% | 21.9% | 9.17 | -10.21 | 0.725 |
| D0xR3 | 16.7% | 4.6% | 16.7% | 4.4% | 0.00 | -0.21 | 1.000 |
| D0_overall | 68.8% | 39.5% | 54.2% | 33.0% | -14.58 | -6.46 | 1.467 |

## 3. D3 V3 Reference Table

| Cell | Track A latent CEM_late | Track A latent overall | V3 oracle CEM_late | V3 oracle overall | Delta CEM_late pp | Delta overall pp | Relative failure reduction CEM_late |
|---|---:|---:|---:|---:|---:|---:|---:|
| D3xR0 | 16.7% | 7.9% | 85.8% | 32.9% | 69.17 | 25.00 | 0.170 |
| D3xR1 | 37.1% | 11.4% | 44.3% | 17.5% | 7.14 | 6.07 | 0.886 |
| D3xR2 | 9.2% | 2.7% | 27.5% | 9.0% | 18.33 | 6.25 | 0.798 |
| D3xR3 | 7.1% | 1.8% | 35.7% | 9.6% | 28.57 | 7.86 | 0.692 |
| D3_overall | 17.9% | 6.0% | 47.7% | 17.0% | 29.81 | 10.96 | 0.637 |

## 4. Row Comparison Summary

| Cell | latent_CEM_late_success | oracle_CEM_late_success | delta_pp | relative_failure_reduction |
|---|---:|---:|---:|---:|
| D3xR0 | 16.7% | 85.8% | 69.17 | 0.170 |
| D3xR3 | 7.1% | 35.7% | 28.57 | 0.692 |
| D0xR2 | 66.7% | 75.8% | 9.17 | 0.725 |
| D3xR2 | 9.2% | 27.5% | 18.33 | 0.798 |
| D3xR1 | 37.1% | 44.3% | 7.14 | 0.886 |
| D0xR3 | 16.7% | 16.7% | 0.00 | 1.000 |
| D0xR1 | 91.7% | 41.7% | -50.00 | 7.000 |
| D0xR0 | 100.0% | 82.5% | -17.50 | inf |

## 5. Headline Finding

- Cells with CEM_late delta_pp >= 10: `3` / 8.
- Cells with relative_failure_reduction <= 0.5: `1` / 8.
- D0 mean CEM_late delta: `-14.58` pp; D3 mean CEM_late delta: `30.80` pp.
- D0 mean relative failure reduction: `inf`; D3 mean relative failure reduction: `0.637`.
- Pattern: D0 cells show a smaller CEM_late latent-vs-oracle gap than D3 cells on average.
- FLAG: D0 CEM_late oracle is worse than latent in: `D0xR0, D0xR1`.
- Additional note: D0 oracle overall success is lower than Track A latent overall in: `D0xR0, D0xR1, D0xR2, D0xR3`.

## 6. Limitations

- This only tests two D-rows; D1 and D2 may differ.
- Ceiling effects: D0 latent success is already high in some cells, so absolute pp gap can understate relative improvement.
- Oracle CEM uses real-env state; V3 is an upper bound on what any latent planner can do with this scalar cost.
- The D0 run evaluated 24 pairs because the accepted Track A sampler produced 6 pairs in each D0 cell.
- The D0 run reused the D3 ablation script with a D0 cell filter and D0 output prefix; no oracle-CEM implementation changes were made.

## 7. Open Question For Next Pass

Data reading, not causal claim: across these 8 cells, the latent-vs-oracle CEM_late gap is not D3-specific. It is largest in D3xR0, but D0xR2 also has a large positive gap and D0xR3 has a positive gap despite low absolute success. D0xR0 and D0xR1 do not show positive CEM_late lift, which makes the row comparison ceiling- and cell-dependent rather than a clean D3-only pattern. Recommended next step: run V3 on D1/D2 before V1/V2, so the row-level scope of the latent-vs-oracle gap is measured before changing the cost criterion.
