# P2-0 Result Memo: Learned Costs, Planning Gap, and Oracle-Budget CEM

## 1. Experiment Overview

P2-0 tested whether a trainable latent cost head `C_psi(z, z_g)` could replace LeWM's default squared Euclidean terminal latent cost in CEM planning for PushT. The world model encoder and predictor stayed frozen. The learned heads used features `concat([z, z_g, z - z_g, abs(z - z_g)])` with 192-d latents, yielding a 768-d input.

The main MLP variants were:

- Small: `768 -> 128 -> 128 -> 1`, ReLU hidden layers.
- Large: `768 -> 512 -> 512 -> 512 -> 1`, diagnostic only.

Training used pairwise ranking hinge loss from V1 oracle labels. Later diagnostic variants included mixed encoder/predictor latents, Mahalanobis metrics, and CEM-aware online distillation with scalar V1 labels on CEM-generated candidates.

The pre-registered planning-level bar was not just metric improvement: the cost must improve PushT CEM success, especially on Split 3, the 16 `all_fail + strong_rho` held-out pairs. A practical go signal was at least `5/16` rescues on Split 3 without obvious regressions on easier Split 1 cases.

## 2. Metric-Level Results

The table below recomputes all methods on the full Split 3 test set using `z_predicted` from the LeWM predictor, because these are the latents seen by CEM. This is the CEM-relevant metric table. Earlier diagnostic numbers such as mixed MLP PA `0.830` / Spearman `0.839` came from a 10-pair representative diagnostic subset, not the full held-out Split 3 test set.

| Method | Pairwise acc | Global Spearman | Per-pair Spearman mean +/- std |
|---|---:|---:|---:|
| Euclidean | 0.578 | 0.233 | 0.230 +/- 0.381 |
| MLP offline small | 0.614 | 0.270 | 0.313 +/- 0.317 |
| MLP offline large | 0.611 | 0.352 | 0.306 +/- 0.325 |
| MLP mixed small | 0.594 | 0.285 | 0.259 +/- 0.282 |
| MLP mixed + temperature | 0.597 | 0.378 | 0.272 +/- 0.292 |
| Mahalanobis low-rank | 0.614 | 0.301 | 0.313 +/- 0.307 |
| CEM-aware | 0.579 | 0.227 | 0.225 +/- 0.352 |

Artifact: `results/phase2/p2_0/split3_predicted_metric_table.json`.

## 3. Planning-Level Results

Pure learned-cost replacement did not transfer to planning success.

| Method | Split 3 planning | Split 1 planning | Notes |
|---|---:|---:|---|
| Euclidean latent CEM | 0/16 | 4/15 | Matched baseline |
| MLP offline small | 0/16 | 3/15 | Metric PA improved, planning regressed on Split 1 |
| MLP offline large | 0/16 | not run | Split 3 only |
| MLP mixed small | 0/16 | not run | Mixed latents did not rescue planning |
| MLP mixed + temperature | 0/16 | not run | Temperature did not rescue planning |
| Mahalanobis low-rank | 0/16 | not run | Best Mahalanobis metric variant |
| CEM-aware online distillation | 1/16 | 2/15 | One Split 3 rescue, but Split 1 regressed |

The pure cost-head route therefore fails the planning-level criterion.

## 4. Diagnosis

The planning gap was not a patching bug. The CEM cost hook changed selected actions, but metric-level ranking quality did not produce a useful CEM search landscape.

Key diagnostics:

- Encoder-predictor distribution shift: on a 10-pair diagnostic subset, MLP small Spearman dropped from `0.881` on real encoder terminal latents to `0.768` on predictor-imagined latents. Pairwise accuracy dropped from `0.856` to `0.785`.
- Latent gap magnitude: mean `L2(z_terminal, z_predicted) = 9.019`; mean `L2(z_terminal, z_goal) = 16.632`; ratio `0.542`.
- Cost compression in CEM: final CEM trace C_psi range was only `[-0.775, -0.707]`; top-30 elite std was `0.0015`, versus Euclidean top-30 elite std `0.3536`.
- Local landscape compression: around a converged Euclidean trajectory, Euclidean cost std was `7.8588`, while C_psi std was `0.0238`.
- CEM dynamics diverged immediately: in deep diagnosis pairs, Euclidean and C_psi final mean action sequences diverged from iteration 1, with final mean-action L2 gaps around `19.5`.
- Split 1 regressions were concrete: MLP C_psi regressed pair 29 (`D1xR0`) and pair 41 (`D1xR2`), both solved by Euclidean.

Interpretation: C_psi learned useful fixed-candidate ranking signals, but CEM needs a globally useful, high-spread landscape throughout iterative sampling. Ranking improvement alone was not sufficient.

## 5. Hybrid CEM Oracle-Budget Results

Hybrid CEM changes the problem: use a cheap learned or latent cost only as a coarse filter, then spend simulator oracle budget on a Top-K subset and refit CEM from the 30 lowest V1 candidates inside that subset.

Implementation note: while adding the C_psi prefilter, we found the first oracle-budget evaluator let prefilter order affect simulator rollout seeding and V1 tie-breaking. The corrected evaluator scores and tie-breaks oracle candidates by original candidate index, matching the Phase 1 oracle helper semantics. This leaves the Euclidean low-budget curve unchanged through K=150, but the corrected full-oracle receding-horizon K=300 count is `13/16` rather than the earlier order-dependent `15/16`.

Corrected Split 3 sweep:

| K | Euclidean prefilter | C_psi prefilter | C_psi advantage |
|---:|---:|---:|---:|
| 0 | 0/16 | 0/16 | +0 |
| 30 | 0/16 | 0/16 | +0 |
| 60 | 7/16 | 6/16 | -1 |
| 90 | 10/16 | 9/16 | -1 |
| 150 | 12/16 | 14/16 | +2 |
| 300 | 13/16 | 13/16 | +0 |

Oracle budget counts:

| K | Oracle rollouts | Oracle env steps |
|---:|---:|---:|
| 0 | 0 | 0 |
| 30 | 57,600 | 1,440,000 |
| 60 | 115,200 | 2,880,000 |
| 90 | 172,800 | 4,320,000 |
| 150 | 288,000 | 7,200,000 |
| 300 | 576,000 | 14,400,000 |

Split 3 per-pair comparison (`S` = success):

| Pair | Cell | Euc K60 | Euc K90 | Euc K150 | C_psi K60 | C_psi K90 | C_psi K150 |
|---:|---|---|---|---|---|---|---|
| 25 | D1xR0 | - | - | - | - | - | - |
| 46 | D1xR3 | S | S | S | - | S | S |
| 60 | D2xR2 | S | S | S | - | - | S |
| 61 | D2xR2 | - | S | S | S | S | S |
| 67 | D2xR3 | - | - | - | S | - | - |
| 70 | D2xR3 | - | - | S | - | - | S |
| 71 | D2xR3 | S | S | S | - | S | S |
| 73 | D2xR3 | - | - | - | - | - | S |
| 78 | D3xR0 | S | S | S | - | S | S |
| 86 | D3xR1 | - | S | S | S | S | S |
| 87 | D3xR2 | S | S | S | S | S | S |
| 93 | D3xR3 | - | - | - | - | - | S |
| 94 | D3xR3 | S | S | S | - | S | S |
| 96 | D3xR3 | S | S | S | - | - | S |
| 97 | D3xR3 | - | S | S | S | S | S |
| 99 | D3xR3 | - | - | S | S | S | S |

Split 1 C_psi prefilter at the advantageous K:

- C_psi prefilter K=150: `11/15`.
- Pure Euclidean baseline: `4/15`.
- Historical learned-cost regressions: pair 41 is solved; pair 29 still fails.

Split 1 C_psi K=150 failures were pairs `29`, `33`, `48`, and `63`.

Artifacts:

- Corrected Euclidean sweep: `results/phase2/p2_0/oracle_budget_cem_corrected/split3_euclidean_prefilter.json`
- Corrected C_psi sweep: `results/phase2/p2_0/oracle_budget_cem_corrected/split3_cpsi_prefilter.json`
- Split 1 C_psi K=150: `results/phase2/p2_0/oracle_budget_cem_corrected/split1_cpsi_prefilter_k150.json`

## 6. Key Findings

F1. Metric-level cost ranking improvement does not reliably transfer to CEM planning success. The learned cost can rank fixed candidates better but still produce a poor iterative search landscape.

F2. The planning bottleneck is cost-landscape flatness and CEM dynamics in the sampled search region, not encoder collapse.

F3. Top-30 coarse filtering is insufficient. Both Euclidean K=30 and C_psi K=30 remain `0/16`, confirming critical success candidates are outside the top-30 candidate set.

F4. Expanding Euclidean's coarse filter to K=60, only 20% of the 300 candidates, rescues `7/16` Split 3 pairs.

F5. There is a phase transition between K=30 and K=60, with continued gains through K=90 and K=150.

F6. C_psi does not reduce oracle budget at low K. It is worse than Euclidean at K=60 and K=90. It does help at K=150, improving Split 3 from `12/16` to `14/16`, suggesting the learned filter is useful only once the oracle set is already broad.

## 7. Revised Go/No-Go Assessment

Pure cost-head replacement is a no-go for P2-0. It fails the Split 3 rescue criterion and can degrade Split 1.

Hybrid CEM with minimal oracle intervention is a strong positive direction. Even Euclidean Top-K plus V1 re-ranking reaches:

- `7/16` at K=60.
- `10/16` at K=90.
- `12/16` at K=150.

C_psi prefiltering is not a low-budget replacement for Euclidean, but it becomes useful at K=150. The revised direction should not be "learn a replacement cost"; it should be "learn or choose a better coarse candidate selector plus minimal oracle or oracle-like intervention."

## 8. Implications for Phase 2 Continuation

The next Phase 2 work should focus on candidate selection and oracle-budget efficiency rather than standalone scalar replacement costs.

Concrete next steps:

- Treat K=60 Euclidean-prefilter hybrid as the minimal positive baseline.
- Study why C_psi helps only at K=150: compare candidate-set overlap and success-candidate inclusion for Euclidean vs C_psi at each K.
- Train an active selector to predict which candidates deserve simulator scoring, optimized for success-candidate recall rather than global ranking.
- Distill oracle re-ranking on CEM-generated candidates, but evaluate by Top-K recall and planning success, not just Spearman.
- Test offsets 25/75/100 to see whether the K threshold shifts with horizon difficulty.
- Revisit Tracks B/C/D around hybrid candidate filtering instead of pure terminal cost replacement.

## 9. Artifact Index

Core latent artifacts:

- `results/phase2/p2_0/track_a_latents.pt`
- `results/phase2/p2_0/track_a_predicted_latents.pt`

Metric summaries:

- `results/phase2/p2_0/all_splits_summary.json`
- `results/phase2/p2_0/split3_predicted_metric_table.json`
- `results/phase2/p2_0/planning_gap_diagnosis.json`
- `results/phase2/p2_0/planning_gap_diagnosis_mixed.json`
- `results/phase2/p2_0/deep_diagnosis.json`

Planning outputs:

- `results/phase2/p2_0/split1_planning.json`
- `results/phase2/p2_0/split3_planning.json`
- `results/phase2/p2_0/split3_planning_mixed.json`
- `results/phase2/p2_0/mahalanobis_planning.json`
- `results/phase2/p2_0/cem_aware_planning.json`
- `results/phase2/p2_0/cem_aware_split1_planning.json`

Hybrid oracle-budget outputs:

- `results/phase2/p2_0/oracle_budget_cem/oracle_budget_cem_summary.json`
- `results/phase2/p2_0/oracle_budget_cem_corrected/split3_euclidean_prefilter.json`
- `results/phase2/p2_0/oracle_budget_cem_corrected/split3_cpsi_prefilter.json`
- `results/phase2/p2_0/oracle_budget_cem_corrected/split1_cpsi_prefilter_k150.json`

Scripts:

- `scripts/phase2/cost_head_model.py`
- `scripts/phase2/dataloader.py`
- `scripts/phase2/train_cost_head.py`
- `scripts/phase2/eval_planning.py`
- `scripts/phase2/diagnose_planning_gap.py`
- `scripts/phase2/deep_diagnosis.py`
- `scripts/phase2/mahalanobis_baseline.py`
- `scripts/phase2/train_cem_aware.py`
- `scripts/phase2/eval_oracle_budget_cem.py`

## 10. Open Questions

- Can a learned prefilter reduce the oracle budget below K=60 while preserving success-candidate recall?
- Can we train a selector directly for Top-K success-candidate inclusion rather than scalar cost ranking?
- Can an oracle-like model approximate simulator V1 re-ranking well enough to replace real rollouts inside the hybrid loop?
- How stable is the K threshold across offsets 25, 75, and 100?
- Does the hybrid result transfer to other environments or is it PushT-specific?
- Are pairs 25, 46, and 67 exposing true dynamic limitations, tie sensitivity, or remaining horizon-budget constraints?
