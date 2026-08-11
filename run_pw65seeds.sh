#!/usr/bin/env bash
# pw65 seed0 scored 0.690 held-out -- but pw50 seed0 (0.684) turned out to be a LUCKY seed
# (pw50 mean 0.651). Seed-confirm pw65 before believing it. Honest seed-mean comparison:
#   pw20 mean 0.588 | pw50 mean 0.651 | pw65 mean = ?
set -u
export PYTHONPATH=_diffusion_net_repo/src
PY=.venv/Scripts/python.exe
for s in 1 2; do
  echo "=== pw65 seed $s $(date) ==="
  $PY train_seg_extra.py --no-extra --partial-dir _label_targets _label_targets_2 _label_targets_3 \
      --partial-target connection --partial-pos-weight 65 --seed $s \
      --checkpoint-out results/seg_extra/human77c_pw65_s$s.pt 2>&1 | grep -E "DONE best"
done
echo "=== DONE $(date) ==="
