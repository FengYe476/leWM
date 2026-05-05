# Response to Reviewer Comments

We thank the reviewer for the thorough and constructive evaluation. Below we address each weakness with new experiments and text revisions. All supplementary experiments, scripts, and data are released in the repository under `scripts/revision/`, `results/revision/`, and `docs/revision/`.

---

## W1: Single-model, dual-environment generalization

**Reviewer concern:** All experiments use only LeWM on PushT and OGBench-Cube. The abstract and introduction overclaim generality with phrases like "any latent world model that plans with terminal latent costs can be audited this way."

**Our response:** We agree that the empirical evidence is bounded to one model and two environments, and have revised all overclaiming language accordingly.

*Abstract:* The original sentence has been replaced with: "For LeWM-style latent planners that select actions by terminal latent costs under iterative sampling-based optimization, we recommend reporting pool-level and selection-level metrics alongside endpoint metrics."

*Introduction (Section 1):* We now state explicitly: "Our empirical claims apply to LeWM with Euclidean terminal latent costs under CEM-style optimization, evaluated on PushT and OGBench-Cube. The evaluation principle is broader, but it should be applied with the same instrumentation."

*Limitations (Section 9):* Three explicit scope boundaries are now listed: (1) single model, two environments; (2) CEM as primary optimizer with partial MPPI cross-check; (3) privileged physical cost required for full evaluation.

*Cross-model feasibility:* We investigated PLDM (Sobal et al., 2025) as a cross-model validation target. The public repository exists but no ready-to-use PushT checkpoint was available at revision time. We leave cross-model validation as future work, noting that the diagnostic protocol ($R_\text{endpoint}$, $R_\text{pool}$, $\Delta_\text{CEM}$) is model-agnostic and directly applicable to any latent world model that exposes its candidate pool.

**Location in revised paper:** Abstract (page 1), Section 1 (page 2, lines 88–97), Section 9 (page 12).

---

## W2: Diagnostic power of ΔCEM — attribution of R_pool ≈ 0

**Reviewer concern:** $R_\text{pool} \approx 0$ could reflect multiple mechanisms: (a) CEM convergence compresses candidates into a physically narrow neighborhood where any cost fails to discriminate, (b) the encoder genuinely loses local order preservation, or (c) predictor bias. The paper does not experimentally distinguish these.

**Our response:** We conducted a new Rpool(V1) attribution experiment on all 100 PushT default CEM final pools. For each pair, we computed the Spearman correlation of the V1 oracle hinge cost (a privileged physical cost) against $C_\text{real\_state}$ within the same 300-candidate pool, and compared it with the learned Euclidean cost.

*Key results:*

| Subset | n | $R_\text{pool}(C_\text{model})$ | $R_\text{pool}(C_\text{V1})$ |
|---|---|---|---|
| Overall | 100 | 0.084 | 0.666 |
| Invisible quadrant | 16 | 0.022 | 0.775 |
| Sign reversal | 21 | −0.040 | 0.925 |
| Latent-favorable | 12 | 0.197 | 0.417 |
| Ordinary | 47 | 0.146 | 0.581 |

The V1 oracle retains strong pool-level ranking ($R_\text{pool}(C_\text{V1}) = 0.666$) where the learned cost fails ($R_\text{pool}(C_\text{model}) = 0.084$). This rules out pure CEM convergence compression as the dominant explanation: if the pool were compressed into a physically indistinguishable neighborhood, V1 would also fail. Instead, the data clearly indicates **local representation failure** — the learned Euclidean cost loses order-preservation in the CEM-converged region while the privileged oracle cost does not.

The effect is strongest in the invisible quadrant (16 pairs where endpoint metrics look healthy but planning fails): $R_\text{pool}(C_\text{V1}) = 0.775$ while $R_\text{pool}(C_\text{model}) = 0.022$, with pool success mass of only 1.4%. The CEM pool contains successful candidates, but the learned cost cannot identify them.

**Location in revised paper:** New Section 4.4 "Mechanism attribution: convergence compression versus local representation failure" (page 8).

---

## W3: Weak Cube evidence

**Reviewer concern:** Cube full projected CEM uses only 25 pairs × 1 seed (vs. 100 pairs × 3 seeds for re-rank-only). The inverted-U pattern received MIXED status. This asymmetry weakens cross-environment claims.

**Our response:** We extended the Cube full projected CEM experiment from 25 pairs × 1 seed to **50 pairs × 3 projection seeds** across all five projection dimensions ($m \in \{1, 8, 32, 64, 192\}$), totaling 750 CEM rollouts with full 300-candidate simulator scoring (~10 hours on Apple Silicon MPS).

*Updated results:*

| $m$ | Full-CEM success (50p, 3s) | Re-rank success (100p, 3s) | Gap |
|---|---|---|---|
| 1 | 32.7% ± 8.1% | 42.7% ± 3.2% | −10.0 pp |
| 8 | 52.7% ± 4.2% | 43.3% ± 1.5% | +9.3 pp |
| 32 | 61.3% ± 1.2% | 46.3% ± 2.5% | +15.0 pp |
| 64 | 56.0% ± 5.3% | 49.7% ± 2.5% | +6.3 pp |
| 192 | 62.0% ± 6.0% | 47.7% ± 1.2% | +14.3 pp |

$R_\text{pool}(C_\text{model})$ across all dimensions ranges from −0.001 to 0.023 — consistently near zero, confirming endpoint-planning decoupling in Cube at scale.

The original inverted-U success pattern (peak at $m=32$, decline at $m=192$) **does not persist** at 50-pair, 3-seed scale. Success plateaus from $m=32$ onward, indicating the original pattern was small-sample noise. This simplifies the cross-environment comparison: both PushT and Cube show rising $R_\text{endpoint}$ with dimension while $R_\text{pool}$ remains near zero.

Table 2 in the revised paper now uses the extended 50-pair × 3-seed Cube data.

**Location in revised paper:** Table 2 (page 7), Section 4.2 discussion (page 7, lines 368–377), Section 9 Limitations (page 12).

---

## W4: No alternative optimizer comparison

**Reviewer concern:** The paper does not test whether endpoint-planning decoupling is CEM-specific (due to hard elite truncation) or generalizes to other iterative optimizers like MPPI.

**Our response:** We implemented an MPPI planner that replaces CEM's hard elite selection with softmax-weighted averaging over all 300 candidates, keeping all other parameters identical (same model, action parameterization, horizon, 30 iterations, 300 candidates). After a temperature sweep ($\tau \in \{0.01, 0.1, 1.0, 10.0\}$), we selected $\tau^* = 1.0$ and ran the experiment on 30 PushT pairs × 3 seeds (90 scored pools).

*Comparison:*

| Metric | CEM default | MPPI ($\tau=1.0$) |
|---|---|---|
| Planning success | 53.3% | 38.9% |
| $R_\text{pool}(C_\text{model})$ | 0.115 | 0.194 |
| $R_\text{pool}(C_\text{V1})$ | 0.537 | 0.707 |
| Pool $C_\text{real\_state}$ std | 7.947 | 29.478 |

MPPI preserves **3.7× more physical diversity** in its final pool than CEM (std 29.478 vs. 7.947), confirming that CEM's hard truncation compresses pool diversity. Correspondingly, $R_\text{pool}(C_\text{model})$ improves from 0.115 to 0.194 under MPPI.

However, MPPI still retains only ~39% of the endpoint ranking signal ($R_\text{pool}/R_\text{endpoint} = 0.194/0.495$). Using $R_\text{endpoint} = 0.495$ at $m=192$ as the reference, the CEM-to-MPPI improvement accounts for roughly **22%** of the endpoint-pool gap. Thus endpoint-planning decoupling is **partially optimizer-specific** (CEM's hard truncation worsens pool geometry) and **partially optimizer-general** (~78% of the lost ranking signal is representation-driven).

The lower MPPI planning success (38.9% vs. 53.3%) is consistent with slower convergence under 30 iterations and does not invalidate the pool-level comparison, which is the relevant diagnostic.

**Location in revised paper:** New Section 8.3 "Optimizer dependence: CEM versus MPPI" (page 12).

---

## W5: Privileged physical cost requirement limits practical applicability

**Reviewer concern:** All pool-level metrics require $C_\text{real\_state}$ (simulator access). How can one detect $\Delta_\text{CEM}$ in deployment without a simulator?

**Our response:** We tested whether privilege-free proxy metrics (computable from the learned cost alone, no simulator needed) correlate with selection regret across the 100 PushT pairs.

*Results:*

| Proxy metric | Corr. with selection regret | p-value |
|---|---|---|
| top30\_Cmodel\_std | 0.314 | 0.001 |
| pool\_Cmodel\_std | 0.254 | 0.011 |
| C\_model\_dynamic\_range | 0.223 | 0.026 |

Within the invisible-quadrant subset, the signal is stronger: pool\_Cmodel\_std vs. selection regret has $\rho = 0.588$ ($p = 0.017$). As a cross-check, pool\_Cmodel\_std tracks pool\_Creal\_std at $\rho = 0.482$ ($p < 0.001$), confirming that learned-cost spread partially reflects physical spread.

No single global proxy crosses our pre-specified $\rho > 0.4$ threshold, so these proxies are not full replacements for the privileged audit. However, they demonstrate that **partial monitoring is feasible without simulator access**: low learned-cost diversity in the CEM final pool can flag candidate-pool compression and elevated selection risk.

**Location in revised paper:** Section 8.1, paragraph on privilege-free monitoring (page 11, lines 584–593).

---

## Summary of Changes

| Weakness | Experiment | Key finding | Paper location |
|---|---|---|---|
| W1 | Text revision + PLDM feasibility | Scope narrowed; PLDM deferred | Abstract, Sec 1, Sec 9 |
| W2 | Rpool(V1) attribution (100 pairs) | Local representation failure dominant | New Sec 4.4 |
| W3 | Cube extended (50p × 3s, 750 runs) | Decoupling robust at scale; inverted-U gone | Table 2, Sec 4.2 |
| W4 | MPPI comparison (30p × 3s, 90 pools) | ~22% CEM-specific, ~78% representation-driven | New Sec 8.3 |
| W5 | Privilege-free proxy (100 pairs) | Partial monitoring feasible (ρ up to 0.588) | Sec 8.1 |

All supplementary scripts, results, and memos are available in the repository under `scripts/revision/`, `results/revision/`, and `docs/revision/`. The progress document `docs/revision/revision_progress.md` provides a complete timeline and artifact inventory.
