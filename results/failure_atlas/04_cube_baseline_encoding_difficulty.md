# Cube Baseline Encoding Difficulty

## 1. Failure Type Name

Baseline 3D encoding difficulty, horizon-independent.

This failure pattern appears in OGBench-Cube as a lower baseline success rate that does not worsen systematically as the goal offset increases. It is therefore not the same mechanism as the PushT large-displacement long-horizon failure.

## 2. Evidence

| Offset | Success rate | Episode count | Budget |
|---:|---:|---:|---:|
| 25 | 68% | 34/50 | 50 |
| 50 | 50% | 25/50 | 100 |
| 75 | 58% | 29/50 | 150 |
| 100 | 50% | 25/50 | 200 |

The Cube sweep stays between `50%` and `68%` across all evaluated offsets. Increasing the offset from `25` to `100` does not produce a monotonic decline, and the offset-100 result is not meaningfully worse than offset-50.

The reproduced Cube baseline at offset `25` was `66%` (`33/50`), compared with the paper's `74%` reference point. The sweep offset-25 run was `68%` (`34/50`), consistent with the baseline reproduction.

## 3. Contrast With PushT

PushT degrades steeply under the same offset stress test:

| Environment | Offset 25 | Offset 50 | Offset 75 | Offset 100 | Pattern |
|---|---:|---:|---:|---:|---|
| PushT | 96% | 58% | 16% | 10% | Strong horizon-dependent degradation |
| OGBench-Cube | 68% | 50% | 58% | 50% | Flat, horizon-independent baseline difficulty |

This contrast is the key evidence that Cube is not simply another example of the PushT Case B/E failure. PushT starts near ceiling and collapses as the required physical displacement grows. Cube starts lower and stays roughly flat.

## 4. Interpretation

Cube failures stem from the inherent challenge of encoding visually complex 3D manipulation scenes with a small ViT-Tiny encoder and SIGReg-style representation learning, not from long-horizon rollout error. The planner does not become progressively worse as offset increases, and the sweep does not show a long-horizon signature.

The paper's own probing results support this interpretation: block quaternion, block yaw, and end-effector yaw probing accuracy are poor on Cube in Table 4. Those probes suggest that the Cube encoder has difficulty representing task-relevant 3D orientation variables even before long-horizon planning pressure is applied.

## 5. Decision-Tree Assignment

**Separate baseline representation limitation, not PushT Case B/E.** Cube is best treated as a horizon-independent encoder-capacity limitation. It strengthens the PushT conclusion by showing that the PushT finding is not a generic LeWM weakness: PushT specifically exposes a displacement-dependent latent geometry failure, while Cube exposes a harder baseline visual encoding problem.
