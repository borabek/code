#!/usr/bin/env bash
set -u
export PYTHONPATH=_diffusion_net_repo/src
PY=.venv/Scripts/python.exe
for s in 1 2; do
  echo "=== human48c seed $s $(date) ==="
  $PY train_seg_extra.py --no-extra --partial-dir _label_targets _label_targets_2 \
      --partial-target connection --seed $s --checkpoint-out results/seg_extra/human48c_s$s.pt 2>&1 | grep -E "DONE best"
done
echo "=== ALL DONE $(date) ==="
