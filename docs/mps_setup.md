# MPS Setup Notes

LeWM's local `third_party/le-wm/jepa.py` needs one MPS-specific dtype fix for PushT evaluation on Apple Silicon.

## Why the fix is needed

The LeWM policy path forwards some floating tensors as `float64`. PyTorch's MPS backend on this machine rejects `float64`, so the tensors must be downcast to `float32` before being moved onto the `mps` device.

## Where the fix lives

The live local change is in `third_party/le-wm/jepa.py`.

Because `third_party/` is gitignored in this repo, the reusable patch is tracked here instead:

`patches/lewm_mps_fix.patch`

## Re-applying after a fresh clone

From the project root:

```bash
cd /Users/fengye/Desktop/Project/leWM/lewm-failure-audit/third_party/le-wm
git apply ../../patches/lewm_mps_fix.patch
```

## Running MPS evals

MPS works when the evaluation is launched from a normal terminal on the host machine. Inside the Codex sandbox, Metal devices may not be visible, so MPS availability checks can return false even when the host supports MPS.
