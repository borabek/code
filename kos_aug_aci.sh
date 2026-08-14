#!/usr/bin/env bash
# Y1b — AUGMENTASYON ACISI TARAMASI
#
# OLCULEN: 0.3 rad kazandiriyor (seg IoU +0.0197, uctan uca robot ISARETLI
# +0.0696). 1.05 rad ZATEN denenmis ve DUSURMUS. Optimum 0.3 ile 1.05
# arasinda bir yerde ve 0.3'un ALTINDA da olabilir -- taranmadi.
# Iki uc denenir: 0.15 (cok hafif) ve 0.5 (orta).
# Tek degisken: augment-maxang. Diger her sey A recetesi.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=_diffusion_net_repo/src PYTHONWARNINGS=ignore
INSAN="_label_targets _label_targets_2 _label_targets_3 _label_targets_recall _label_targets_recall_hard"
for A in 0.15 0.5; do
  echo "############ augment-maxang $A ############"
  .venv/Scripts/python.exe train_seg_extra.py --no-extra --partial-dir $INSAN \
    --partial-target connection --seed 0 --k-eig 96 --epochs 200 \
    --augment --augment-maxang $A \
    --checkpoint-out "results/seg_extra/y1b_aug${A}_s0.pt"
done
