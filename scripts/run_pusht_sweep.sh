#!/bin/bash
# Full PushT goal-offset sweep on MPS.
# Run this manually in terminal (not in Codex sandbox).

cd /Users/fengye/Desktop/Project/leWM/lewm-failure-audit
conda run -n lewm-audit python scripts/eval_pusht_sweep.py \
    --cache-dir stablewm_cache \
    --offsets 25,50,75,100 \
    --num-eval 50 \
    --device mps
