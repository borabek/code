#!/usr/bin/env bash
# 77 human labels (20 batch-1 + 28 batch-2 + 29 batch-3), connection-channel, 3 seeds.
# Baseline product human48c: arbiter CP F1 0.605 / mean 0.633; 71-only val 0.622.
set -u
export PYTHONPATH=_diffusion_net_repo/src
PY=.venv/Scripts/python.exe
for s in 0 1 2; do
  echo "=== human77c seed $s $(date) ==="
  $PY train_seg_extra.py --no-extra --partial-dir _label_targets _label_targets_2 _label_targets_3 \
      --partial-target connection --seed $s --checkpoint-out results/seg_extra/human77c_s$s.pt 2>&1 | grep -E "DONE best|partial-human"
done
echo "=== ALL DONE $(date) ==="
