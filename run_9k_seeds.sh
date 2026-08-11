#!/usr/bin/env bash
# 9k seed0 held-out CP F1 0.659 vs 6k's 0.636 (same seed): +6 CPs found, recall .598 -> .671 --
# the resolution hypothesis in the right direction, but +0.023 is INSIDE seed noise. Confirm.
set -u
export PYTHONPATH=_diffusion_net_repo/src
PY=.venv/Scripts/python.exe
for s in 1 2; do
  echo "=== 9k seed $s $(date) ==="
  $PY train_seg_extra.py --no-extra --train-dir _corpus9k_train --val-dir _corpus9k_val \
      --partial-dir _label_targets_9k _label_targets_2_9k _label_targets_3_9k \
      --partial-target connection --seed $s \
      --checkpoint-out results/seg_extra/human77c_9k_s$s.pt 2>&1 | grep -E "DONE best"
done
echo "=== DONE $(date) ==="
