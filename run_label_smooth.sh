#!/usr/bin/env bash
# Y6 — LABEL SMOOTHING (yumusak etiket)
#
# Kismi etiketli maskeli BCE'de hedef 1.0 yerine (1-eps) kullanilir.
# Gerekce: kismi etiketler ELLE isaretlendi ve kenar tepelerinde
# belirsizdir; conclusive 1.0 hedefi modeli asiri kendine guvenli yapar.
# Augmentasyon kazandi (+0.0197..+0.0649) -- same aileden a duzenleyici.
#
# TEK DEGISKEN: label smoothing. Diger each sey A recetesi + hafif augment
# (bugunun kazanani) uzerine biner.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=_diffusion_net_repo/src PYTHONWARNINGS=ignore
INSAN="_label_targets _label_targets_2 _label_targets_3 _label_targets_recall _label_targets_recall_hard"
.venv/Scripts/python.exe train_seg_extra.py --no-extra --partial-dir $INSAN \
  --partial-target connection --seed 0 --k-eig 96 --epochs 200 \
  --augment --augment-maxang 0.15 --label-smooth 0.05 \
  --checkpoint-out results/seg_extra/y6_ls_s0.pt
