#!/usr/bin/env bash
# pos_weight 50 has the best RECALL (.646, 53/82) and ensembling adds mostly PRECISION
# (baseline ensemble: P +.124, R +.061) -> pw50 is the right ensemble base. Train seeds 1,2.
set -u
export PYTHONPATH=_diffusion_net_repo/src
PY=.venv/Scripts/python.exe
for s in 1 2; do
  echo "=== pw50 seed $s $(date) ==="
  $PY train_seg_extra.py --no-extra --partial-dir _label_targets _label_targets_2 _label_targets_3 \
      --partial-target connection --partial-pos-weight 50 --seed $s \
      --checkpoint-out results/seg_extra/human77c_pw50_s$s.pt 2>&1 | grep -E "DONE best"
done
echo "=== DONE $(date) ==="
