#!/usr/bin/env bash
# Y1 — HAFIF AUGMENTASYON (agresif which ZATEN denenmis ve dusurmus)
#
# `--augment` bayraginin kendi yardim metni: "1.05 = aggressive (the
# heuristic-tuned one that hurt quality)". Yani AGRESIF donme (~60 derece)
# denenmis ve kaliteyi dusurmus. Egitim logu hala `augment=False` diyor,
# i.e. su an HIC augmentasyon none. Denenmemis which HAFIF ayar.
#
# TEK DEGISKEN: augment (0.3 rad ~ 17 derece). Etiket dizinleri, seed,
# k-eig, epoch, girdi ozniteligi AYNI.
# Taban = A arm: 189 part, 200 epoch, val Conn_IoU 0.6232.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=_diffusion_net_repo/src PYTHONWARNINGS=ignore
INSAN="_label_targets _label_targets_2 _label_targets_3 _label_targets_recall _label_targets_recall_hard"
.venv/Scripts/python.exe train_seg_extra.py --no-extra --partial-dir $INSAN \
  --partial-target connection --seed 0 --k-eig 96 --epochs 200 \
  --augment --augment-maxang 0.3 \
  --checkpoint-out results/seg_extra/y1_aug03_s0.pt
