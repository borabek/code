#!/usr/bin/env bash
# Y5 — YARDIMCI GOREV (aux-wire): TEL/ALET etiketiyle cok-gorevli ogrenme
#
# `--aux-wire` bayragi kodda VARDI ama bu kampanyada HIC acilmadi.
# 53 parcada TEL/ALET yardimci etiketi mevcut. Cok-gorevli ogrenme
# temsili guclendirebilir: ayni govdeden iki farkli sinyal.
#
# TEK DEGISKEN: aux-wire. Taban = hafif augmentasyon 0.15 (0.6528).
set -u
cd "$(dirname "$0")"
export PYTHONPATH=_diffusion_net_repo/src PYTHONWARNINGS=ignore
INSAN="_label_targets _label_targets_2 _label_targets_3 _label_targets_recall _label_targets_recall_hard"
.venv/Scripts/python.exe train_seg_extra.py --no-extra --partial-dir $INSAN \
  --partial-target connection --seed 0 --k-eig 96 --epochs 200 \
  --augment --augment-maxang 0.15 --aux-wire \
  --checkpoint-out results/seg_extra/y5_aux_s0.pt
