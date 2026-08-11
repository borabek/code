#!/usr/bin/env bash
# The adjudication fix (87 human-confirmed false negatives -> 26907 vertices relabelled) improved
# ALL THREE measures on seed 0: human held-out .636 -> .719, PXC .780 -> .782, WEI .233 -> .252.
# Seed-confirm before it becomes the product -- single seeds misled us four times today.
set -u
export PYTHONPATH=_diffusion_net_repo/src
PY=.venv/Scripts/python.exe
for s in 1 2; do
  echo "=== adj seed $s $(date) ==="
  $PY train_seg_extra.py --no-extra --partial-dir _label_targets _label_targets_2 _label_targets_3 \
      --partial-target connection --seed $s \
      --checkpoint-out results/seg_extra/adj_s$s.pt 2>&1 | grep -E "DONE best"
done
echo "=== DONE $(date) ==="
