#!/bin/bash
cd /Users/fengye/Desktop/Project/leWM/lewm-failure-audit
conda run -n lewm-audit python scripts/eval_cube_sweep.py \
    --cache-dir stablewm_cache \
    --offsets 25,50,75,100 \
    --num-eval 50 \
    --device mps
