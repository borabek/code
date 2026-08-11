#!/usr/bin/env bash
# pos_weight 50 was a breakthrough on the 82-CP held-out (F1 .539 -> .684, recall .463 -> .646).
# Is there more headroom? Try 65 and 80. Then 3 seeds of the winner -> ensemble.
set -u
export PYTHONPATH=_diffusion_net_repo/src
PY=.venv/Scripts/python.exe
for w in 65 80; do
  echo "=== pos_weight $w $(date) ==="
  $PY train_seg_extra.py --no-extra --partial-dir _label_targets _label_targets_2 _label_targets_3 \
      --partial-target connection --partial-pos-weight $w --seed 0 \
      --checkpoint-out results/seg_extra/human77c_pw$w.pt 2>&1 | grep -E "DONE best"
done
echo "=== DONE $(date) ==="
