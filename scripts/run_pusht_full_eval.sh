#!/bin/bash
# Full 50-episode PushT baseline evaluation on MPS
# Run this manually in terminal (not in Codex sandbox)

cd /Users/fengye/Desktop/Project/leWM/lewm-failure-audit
conda activate lewm-audit
python scripts/eval_pusht_baseline.py \
    --cache-dir stablewm_cache \
    --results-path results/pusht_baseline_eval_mps.json \
    --num-eval 50 \
    --device mps
