#!/usr/bin/env bash
# TODO A-3 (last untried lever): MILD augmentation on the human parts. Aggressive augment (maxang 1.05)
# hurt the corpus before; mild (0.25 rad) + on the human-label training has never been tried.
# Baseline to beat: pw20 seed-mean 0.588 held-out (same seeds, no augment).
set -u
export PYTHONPATH=_diffusion_net_repo/src
PY=.venv/Scripts/python.exe
for s in 0 1 2; do
  echo "=== augment(mild) seed $s $(date) ==="
  $PY train_seg_extra.py --no-extra --partial-dir _label_targets _label_targets_2 _label_targets_3 \
      --partial-target connection --augment --augment-maxang 0.25 --seed $s \
      --checkpoint-out results/seg_extra/human77c_aug_s$s.pt 2>&1 | grep -E "DONE best"
done
echo "=== DONE $(date) ==="
