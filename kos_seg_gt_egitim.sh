#!/usr/bin/env bash
# SEG-2 — SEGMENTASYONU GT KORPUSUYLA YENIDEN EGIT (A/B)
#
# NEDEN. Canli seg kontrol noktalari 27 Temmuz tarihli. Kontrol kolu yalnizca
# **189 parca** (118 kismi insan etiketi) goruyor. Kampanya boyunca SECICI
# optimize edildi, segmentasyon DONDURULMUS kaldi ve NIT'te zincirin EN BASI
# bilgi uretmiyor (GT 0.5244 / rastgele 0.4394 = 1.19x; SUPU'da 83x).
#
# TEK DEGISKEN: `_label_targets_gt` (1556 parca, URETICI GT'sinden boyanmis,
# d6/d7 bekciyle DISARIDA). B kolu 189 -> ~1745 parca, yani 9 KAT veri.
#
# EPOCH SECIMI -- DURUSTCE ASIMETRIK. B'yi de 200 epoch kosmak ~16 saat
# surerdi. B 40 epoch kosuyor: 40 x 1745 = 69.800 ornek, A'nin
# 200 x 189 = 37.800 ornegiyle kiyasla ~1.85 kat. Yani B kendi recetesine
# gore AZ egitilmis; kiyas B'nin ALEYHINE egimli. B yine de kazanirsa kanit
# guclu, kaybederse BELIRSIZ -- rapora oyle yazilir.
#
# k-eig 96: canli kontrol noktalari `recall_hard_keig96_*`.
# URUN MODELI EZILMEZ: `--checkpoint-out` yeni ada yazar.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=_diffusion_net_repo/src PYTHONWARNINGS=ignore
PY=".venv/Scripts/python.exe"
INSAN="_label_targets _label_targets_2 _label_targets_3 _label_targets_recall _label_targets_recall_hard"

echo "############ A (KONTROL): GT korpusu YOK, 200 epoch ############"
$PY train_seg_extra.py --no-extra --partial-dir $INSAN \
  --partial-target connection --seed 0 --k-eig 96 --epochs 200 \
  --checkpoint-out results/seg_extra/gt_A_kontrol_s0.pt

echo "############ B (DENEY): + _label_targets_gt (1556 parca), 40 epoch ############"
$PY train_seg_extra.py --no-extra --partial-dir $INSAN _label_targets_gt \
  --partial-target connection --seed 0 --k-eig 96 --epochs 40 \
  --checkpoint-out results/seg_extra/gt_B_korpus_s0.pt
