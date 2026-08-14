#!/usr/bin/env bash
# Y1-TOPLULUK — augmentasyonlu EK TOHUMLAR
#
# MEASURED (VAL 100 part, TAM zincir, tek-vs-tek adil kiyas, `measured_path` yolu):
#   A kontrol (tek ckpt) : detection 0.5430 | robot unsigned 0.4792 | ISARETLI 0.4135
#   Y1 augment (tek ckpt): detection 0.5808 | robot unsigned 0.5293 | ISARETLI 0.4831
#   diff                 : +0.0378       | +0.0501               | **+0.0696**
#
# Ve TEK augmentasyonlu ckpt, DAGITILMIS DORT ckpt'lik toplulugun sahasina
# (0.4839) denk. Oyleyse augmentasyonlu a TOPLULUK ikisini de gecmeli.
# Bu betik seed 1 ve 2'yi ekler (seed 0 = y1_aug03_s0.pt already present).
set -u
cd "$(dirname "$0")"
export PYTHONPATH=_diffusion_net_repo/src PYTHONWARNINGS=ignore
INSAN="_label_targets _label_targets_2 _label_targets_3 _label_targets_recall _label_targets_recall_hard"
for S in 1 2; do
  echo "############ augment seed $S ############"
  .venv/Scripts/python.exe train_seg_extra.py --no-extra --partial-dir $INSAN \
    --partial-target connection --seed $S --k-eig 96 --epochs 200 \
    --augment --augment-maxang 0.3 \
    --checkpoint-out "results/seg_extra/y1_aug03_s$S.pt"
done
