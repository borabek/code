#!/usr/bin/env bash
# Y14 — FOCAL-TVERSKY (gamma ussu)
#
# Tversky ZATEN acik (agirlik 0.25, alpha 0.3, beta 0.7). Focal varyanti
# (1-TI)^gamma with kolay orneklerin katkisini bastirir. gamma=1 klasik
# Tversky'dir, i.e. default davranis DEGISMEZ -- arm tek degiskenli.
#
# Taban: hafif augmentasyon 0.15 (bugunun kazanani, 0.6528).
set -u
cd "$(dirname "$0")"
export PYTHONPATH=_diffusion_net_repo/src PYTHONWARNINGS=ignore
INSAN="_label_targets _label_targets_2 _label_targets_3 _label_targets_recall _label_targets_recall_hard"
.venv/Scripts/python.exe train_seg_extra.py --no-extra --partial-dir $INSAN \
  --partial-target connection --seed 0 --k-eig 96 --epochs 200 \
  --augment --augment-maxang 0.15 --tversky-gamma 1.33 \
  --checkpoint-out results/seg_extra/y14_ft133_s0.pt
