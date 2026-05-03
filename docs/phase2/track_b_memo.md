# Phase 2 Track B Memo: DINOv2 Encoder Control

## 1. Experiment Design and Motivation

Phase 2 P2-0 showed that learned cost heads can improve metric-level ranking but still fail inside CEM planning, while hybrid CEM succeeds only when simulator/V1 re-ranking is applied continuously. Track B asks whether the endpoint cost-ranking mismatch is specific to LeWM's compact SIGReg encoder, or whether a stronger generic visual encoder gives a better coarse-ranking signal.

The originally proposed "DINOv2 as hybrid CEM prefilter" is not a cheap prefilter in this setup: LeWM CEM rollouts produce LeWM latents, not pixels or DINOv2 features. Evaluating DINOv2 for a candidate would require executing the candidate in the simulator to obtain terminal pixels, which is already oracle-like. Therefore Track B was run as endpoint re-scoring of the same Track A records:

- 100 Track A pairs x 80 action candidates = 8000 records.
- Ranking target: V1 hinge cost.
- Compared endpoint Euclidean costs:
  - LeWM SIGReg latent distance, 192-d.
  - DINOv2 ViT-B/14 CLS-token distance, 768-d.
  - DINOv2 ViT-B/14 mean-pooled patch-token distance, 768-d.
  - Seed-0 Gaussian random projection of LeWM latents, 192-d, as an additional control.

## 2. Feature Extraction Details

DINOv2 loaded successfully through `torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")`.

- Parameters: 86,580,480.
- Device: MPS.
- xFormers was unavailable, but DINOv2 still ran successfully.
- Seed: 0.
- Input preprocessing: PushT pixels converted to RGB tensors, resized to 224 x 224 when needed, scaled to [0, 1], then ImageNet-normalized.
- Features saved to `results/phase2/track_b/dinov2_features.pt`.
- Ranking comparison saved to `results/phase2/track_b/ranking_comparison.json`.

The P2-0 latent artifact did not contain terminal pixels, only LeWM latents and scalar costs. I therefore replayed the exact Track A action-selection pipeline and simulator rollouts, validating source/source-index/V1 cost against `results/phase2/p2_0/track_a_latents.pt` for every record before encoding terminal pixels with DINOv2. Pixels were encoded in batches and not stored in the main DINOv2 artifact.

## 3. Ranking Comparison Results

Main ranking metrics over 8000 records:

| Encoder | Dim | Global Spearman | Pairwise Acc | Per-Pair Rho Mean | Per-Pair Rho Std |
|---|---:|---:|---:|---:|---:|
| LeWM (SIGReg) | 192 | 0.5033 | 0.6470 | 0.3549 | 0.5021 |
| DINOv2 CLS | 768 | 0.2387 | 0.5943 | 0.2482 | 0.3651 |
| DINOv2 mean-pool | 768 | 0.2609 | 0.6099 | 0.2848 | 0.3686 |
| Random projection | 192 | 0.5010 | 0.6429 | 0.3493 | 0.4887 |

Per-cell pairwise accuracy, using the better DINOv2 variant, mean-pooled patch features:

| Cell | LeWM PA | DINOv2 mean PA | Delta |
|---|---:|---:|---:|
| D0xR0 | 0.8116 | 0.6410 | -0.1706 |
| D0xR1 | 0.8299 | 0.7259 | -0.1040 |
| D0xR2 | 0.7458 | 0.5906 | -0.1552 |
| D0xR3 | 0.4629 | 0.5058 | 0.0430 |
| D1xR0 | 0.7285 | 0.6247 | -0.1038 |
| D1xR1 | 0.5755 | 0.5563 | -0.0192 |
| D1xR2 | 0.6428 | 0.6650 | 0.0222 |
| D1xR3 | 0.3344 | 0.4087 | 0.0743 |
| D2xR0 | 0.6214 | 0.6703 | 0.0489 |
| D2xR1 | 0.7142 | 0.6242 | -0.0900 |
| D2xR2 | 0.4659 | 0.5773 | 0.1114 |
| D2xR3 | 0.6296 | 0.6260 | -0.0036 |
| D3xR0 | 0.7607 | 0.6507 | -0.1100 |
| D3xR1 | 0.7743 | 0.6769 | -0.0974 |
| D3xR2 | 0.7643 | 0.6278 | -0.1365 |
| D3xR3 | 0.6611 | 0.6175 | -0.0436 |

DINOv2 mean-pool beats LeWM in 5/16 cells, mostly in harder high-rotation or mid-displacement cells, but loses clearly in aggregate.

## 4. Interpretation

Outcome: DINOv2 << LeWM on the aggregate endpoint ranking metrics.

LeWM's task-trained SIGReg encoder is better for this PushT coarse-ranking objective than generic DINOv2 features, despite LeWM's planning limitations and much smaller latent size. The random projection result is also close to LeWM, which suggests that much of the LeWM endpoint signal is still present under random linear mixing; the failure mode is not simply "192 dimensions are too few."

This does not support the claim that a richer generic visual encoder would reduce oracle intervention needs. It also does not establish that the mismatch is generic across all visual encoders, because DINOv2 is not approximately tied with LeWM here. The strongest conclusion is narrower: LeWM's compact encoder is unlikely to be the primary bottleneck for coarse endpoint ranking, and replacing it with off-the-shelf DINOv2 Euclidean distance would not fix the cost landscape.

## 5. Implications for Paper Framing

The paper should not frame the P2-0 failure as mainly an encoder-capacity issue. A stronger generic visual encoder did not outperform LeWM on the same endpoint ranking task; LeWM remained the better coarse ranker overall.

The cleaner framing is:

- LeWM's learned representation contains useful endpoint-ranking information.
- The planning failure is more likely tied to the mismatch between endpoint Euclidean ranking and CEM's search dynamics, plus the need for continuous simulator-grounded correction.
- Hybrid CEM's success should be presented as evidence that Euclidean latent cost is a useful prefilter, not a sufficient planning objective.
- Track B weakens the "just use a stronger visual encoder" explanation and motivates paper language around objective/search mismatch rather than compact-encoder deficiency.
