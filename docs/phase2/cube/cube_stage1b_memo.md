# Cube Stage 1B Memo

## 1. Verdict

Verdict: **CUBE-STAGE1B-FLAT-ROBUST**.

Cube projected-cost re-ranking is much flatter than PushT. Success is already high at `m=1` (`42.7%`) and peaks at `m=64` (`49.7%`), slightly above the default LeWM rank-1 baseline (`49.0%`) and above the projected `m=192` result (`47.7%`). There is no PushT-style sharp elbow from low dimensions to `m=32`; Cube instead shows broad robustness to random low-dimensional projections.

Important protocol note: this Cube run used the planned **re-rank-only** Stage 1B design. It runs default LeWM CEM once per pair, simulator-scores the final 300-candidate pool, and selects projected rank-1 candidates from that labelled pool. The PushT reference JSON used full projected CEM. Cross-environment comparisons should therefore be read as projected selection robustness, not as identical optimizer-dynamics experiments.

## 2. Provenance

| Item | Value |
|---|---|
| Script | `scripts/phase2/cube/cube_stage1b.py` |
| Full results | `results/phase2/cube/cube_stage1b.json` |
| Smoke results | `results/phase2/cube/cube_stage1b_smoke.json` |
| Full stdout log | `results/phase2/cube/cube_stage1b_full.log` |
| Smoke stdout log | `results/phase2/cube/cube_stage1b_smoke.log` |
| Planning mode | `rerank_default_final_pool` |
| Pairs | `100` |
| Final candidates scored | `300` per pair |
| Simulator rollouts | `30,000` |
| Projected records | `2,700` |
| Wall-clock | `4747.490s` (`1h 19m 7.5s`) |
| Smoke wall-clock | `234.394s` (`3m 54.4s`) |

## 3. Headline Results

| Dim | Cube success | PushT success | Cube planning Spearman | Cube false elite | Cube action L2 |
|---:|---:|---:|---:|---:|---:|
| `m=1` | `42.7% +/- 3.2%` | `11.1%` | `0.002` | `0.616` | `9.902` |
| `m=2` | `41.3% +/- 6.5%` | `13.8%` | `0.008` | `0.615` | `9.879` |
| `m=4` | `44.7% +/- 1.2%` | `15.3%` | `0.004` | `0.608` | `8.777` |
| `m=8` | `43.3% +/- 1.5%` | `20.6%` | `0.005` | `0.608` | `7.487` |
| `m=16` | `43.3% +/- 0.6%` | `23.8%` | `0.010` | `0.603` | `5.566` |
| `m=32` | `46.3% +/- 2.5%` | `28.0%` | `0.007` | `0.609` | `4.238` |
| `m=64` | `49.7% +/- 2.5%` | `29.6%` | `0.008` | `0.607` | `3.647` |
| `m=128` | `47.7% +/- 2.9%` | `30.7%` | `0.009` | `0.605` | `2.470` |
| `m=192` | `47.7% +/- 1.2%` | `34.4%` | `0.006` | `0.606` | `2.301` |

Default LeWM rank-1 success on the same Cube pools is `49.0%`. The `m=192` projected rank-1 success is `47.7%`, a `1.3pp` gap. This is expected: a same-dimensional Gaussian projection is approximately but not exactly distance-preserving, so it slightly perturbs the original Euclidean LeWM ranking. This gap is a useful noise floor for the projection intervention itself, and it is consistent with the PushT Stage 1A observation that C2 `m=192` closely but not perfectly matches C0.

The `m=64` success (`49.7%`) exceeding `m=192` (`47.7%`) should not be interpreted as evidence that more dimensions hurt. The `2.0pp` gap is within the seed-level standard deviations (`2.5%` at `m=64`, `1.2%` at `m=192`), and there are only `3` projection seeds. The safe conclusion is that Cube re-rank success is approximately flat from `m=32` to `m=192`.

## 4. Planning Elbow

Cube does **not** reproduce the PushT elbow at `m=32-64`. PushT rises sharply from `11.1%` at `m=1` to `28.0%` at `m=32`, then improves more slowly. Cube starts at `42.7%` at `m=1`, stays in a narrow `41-45%` band through `m=16`, and peaks at `49.7%` at `m=64`.

The Cube elbow is therefore weak or nearly absent. If forced to name one, `m=32-64` is still the local improvement region, but the gain from `m=1` to `m=64` is only `+7.0pp`, versus PushT's `+18.5pp` over the same range.

The final-pool candidate success rate provides important context for this flat curve. Across Cube pairs, `38.6%` of the 300 candidates in the default final pool are already successful, so a random draw from that pool would succeed about `38.6%` of the time. The `m=1` projected success of `42.7%` is only about `4pp` above this random-pool baseline. Cube therefore appears easy at low dimensions partly because the full-dimensional default CEM pool already contains many successful actions, not because a one-dimensional random projection is especially informative. This likely differs from PushT, where the final pool appears harder and low-dimensional selection is more costly.

## 5. Endpoint-Planning Decoupling

Cube shows even stronger endpoint-planning decoupling than PushT. Cube Stage 1A endpoint Spearman climbs from `0.238` at `m=1` to `0.603` at `m=192`, but Cube Stage 1B planning-pool Spearman remains essentially zero.

| Dim | Cube Stage 1A C2 Spearman | Cube Stage 1B Spearman | Gap |
|---:|---:|---:|---:|
| `m=1` | `0.238` | `0.002` | `0.236` |
| `m=2` | `0.323` | `0.008` | `0.315` |
| `m=4` | `0.418` | `0.004` | `0.413` |
| `m=8` | `0.477` | `0.005` | `0.472` |
| `m=16` | `0.520` | `0.010` | `0.509` |
| `m=32` | `0.570` | `0.007` | `0.563` |
| `m=64` | `0.575` | `0.008` | `0.567` |
| `m=128` | `0.594` | `0.009` | `0.585` |
| `m=192` | `0.603` | `0.006` | `0.597` |

This means global endpoint ranking quality is not what explains Cube planning-pool selection. The final CEM pool is locally clustered, and within that cluster projected costs have almost no monotonic relationship with true cube position distance while still selecting successful actions at high rates.

## 6. False Elite Stability

False elite rate is stable across dimensions, ranging only from `0.603` to `0.616`. This mirrors the PushT conclusion that projection does not collapse the landscape, but Cube's level is lower than PushT's roughly `0.689` because Cube's final candidate pools contain more successful candidates.

Top-30 overlap with LeWM increases smoothly with dimension: `0.203` at `m=1`, `0.609` at `m=8`, `0.831` at `m=64`, and `0.886` at `m=192`. Action L2 to the default rank-1 action falls from about `9.9` at `m=1-2` to `2.3` at `m=192`, confirming that higher dimensions converge toward the default LeWM ranking even though success is already strong at low dimension.

## 7. Cross-Environment Comparison

PushT and Cube now differ in both C6 and Stage 1B:

| Pattern | PushT | Cube |
|---|---|---|
| Random-init C6 | inversion (`C6-REAL`) | no inversion (`C6-NO-INVERSION`) |
| Stage 1A endpoint C0 | `0.506` | `0.604` |
| Stage 1B low-dim success | weak at `m=1-8` under full projected CEM | strong at `m=1-8` under default-pool re-ranking |
| Stage 1B elbow | clear `m=32-64` elbow | flat, weak elbow |
| Final-pool Spearman | near zero | even closer to zero |
| False elite | flat around `0.689` | flat around `0.61` |

The protocol difference is central. Cube re-ranks candidates from a pool that was already optimized by full-dimensional LeWM CEM, so the pool's baseline quality is high before projection is applied. PushT Stage 1B used full projected CEM, where the CEM optimization loop itself ran with projected costs. Cube's high low-dimensional success therefore partly reflects the quality of the default CEM pool, not just the ranking ability of the projection. Direct numerical comparison of absolute success rates across environments is confounded by this difference.

The valid cross-environment comparison is the **shape** of the dimension curve, not the absolute success level. Under that lens, PushT shows a clear `m=32-64` elbow, while Cube's re-rank curve is much flatter. The key result is therefore not simply "Cube needs fewer random dimensions"; it is that default-pool re-ranking on Cube remains flat across dimensions because the optimized pool already contains many successful candidates.

## 8. Implications for the Paper

Cube strengthens the endpoint-planning decoupling claim. Endpoint Spearman is excellent on the fixed Cube artifact, yet final-pool Spearman is nearly zero at every dimension. Planning success is therefore not reducible to global endpoint rank preservation.

Cube also complicates a single universal "planning subspace elbow" story. PushT supports a `32-64` dimensional effective planning subspace. Cube re-rank results suggest a broader or easier final-pool geometry where even one to four random dimensions often preserve successful selection. The paper should frame low-dimensional planning robustness as environment- and pool-dependent rather than a fixed intrinsic dimensionality.

Recommended paper finding:

> Random projection preserves CEM final-pool selection surprisingly well across environments, but the dimensional curve is environment-dependent: PushT has a clear `32-64` elbow, while Cube is already robust at very low dimensions and peaks near `m=64`.

## 9. Artifacts

| Type | Path |
|---|---|
| Script | `scripts/phase2/cube/cube_stage1b.py` |
| Smoke JSON | `results/phase2/cube/cube_stage1b_smoke.json` |
| Smoke stdout | `results/phase2/cube/cube_stage1b_smoke.log` |
| Full JSON | `results/phase2/cube/cube_stage1b.json` |
| Full stdout | `results/phase2/cube/cube_stage1b_full.log` |
| Memo | `docs/phase2/cube/cube_stage1b_memo.md` |
| Cube Stage 1A reference | `results/phase2/cube/cube_stage1a.json` |
| PushT Stage 1B reference | `results/phase2/stage1/stage1b_full.json` |
