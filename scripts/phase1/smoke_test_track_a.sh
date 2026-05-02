#!/usr/bin/env bash
set -euo pipefail

conda run -n lewm-audit python scripts/phase1/eval_track_a_three_cost.py \
  --max-pairs 5 \
  --output results/phase1/track_a_three_cost_smoke.json
