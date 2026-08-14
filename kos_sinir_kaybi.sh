#!/usr/bin/env bash
# Y13 — SINIR-FARKINDALI KAYIP
# CP fiziksel olarak bir SINIRDIR (mouth cemberi) ama mevcut loss
# (NLL + Tversky) BOLGEYI hedefler. Ek terim, komsusu farkli sinifta olan
# tepelere odaklanir. Taban: hafif augmentasyon 0.15 (0.6528).
set -u
cd "$(dirname "$0")"
export PYTHONPATH=_diffusion_net_repo/src PYTHONWARNINGS=ignore
INSAN="_label_targets _label_targets_2 _label_targets_3 _label_targets_recall _label_targets_recall_hard"
for W in 0.3 1.0; do
  echo "########## sinir_weight $W ##########"
  .venv/Scripts/python.exe train_seg_extra.py --no-extra --partial-dir $INSAN \
    --partial-target connection --seed 0 --k-eig 96 --epochs 200 \
    --augment --augment-maxang 0.15 --sinir-weight $W \
    --checkpoint-out "results/seg_extra/y13_sinir${W}_s0.pt"
done
