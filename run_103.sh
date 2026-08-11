#!/usr/bin/env bash
# 103 human labels (20+28+29+26), connection-channel, 3 seeds. Baseline human77c CP F1 0.698.
set -u
export PYTHONPATH=_diffusion_net_repo/src
PY=.venv/Scripts/python.exe
for s in 0 1 2; do
  echo "=== human103c seed $s $(date) ==="
  $PY train_seg_extra.py --no-extra --partial-dir _label_targets _label_targets_2 _label_targets_3 _label_targets_4 \
      --partial-target connection --seed $s --checkpoint-out results/seg_extra/human103c_s$s.pt 2>&1 | grep -E "DONE best|partial-human"
done
echo "=== ALL DONE $(date) ==="
