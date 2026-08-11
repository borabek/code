#!/usr/bin/env bash
# TODO A-2: partial_pos_weight sweep. RECALL is now the only lever (post-proc at ceiling, labels flat).
# pos_weight raises the connection-channel BCE positive weight -> more aggressive CP prediction ->
# higher recall, lower precision. Current 20. Try 35 and 50 (seed 0 for a quick read).
set -u
export PYTHONPATH=_diffusion_net_repo/src
PY=.venv/Scripts/python.exe
for w in 35 50; do
  echo "=== pos_weight $w $(date) ==="
  $PY train_seg_extra.py --no-extra --partial-dir _label_targets _label_targets_2 _label_targets_3 \
      --partial-target connection --partial-pos-weight $w --seed 0 \
      --checkpoint-out results/seg_extra/human77c_pw$w.pt 2>&1 | grep -E "DONE best"
done
echo "=== DONE $(date) ==="
