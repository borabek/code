#!/usr/bin/env bash
# SEG-2C — GT KORPUSU KOLU, SILINDIRIK KABUK ETIKETIYLE
#
# B kolu KURE boyamayla dustu (0.5540 vs A 0.6232). Kablo girisi kure
# degil KANAL YUZEYIDIR; etiket sekli yanlisti. Yeni etiketler GT EKSENI
# etrafinda silindirik kabuk: isaretli tepe orani %0.94 -> %1.63 (elle
# etiketli korpus ~%1.5). Tek degisken ETIKET SEKLI.
#
# ILK DENEME OLDU: 1556 parcalik `_label_targets_gt` ile operatör hazirligi
# sirasinda traceback BIRAKMADAN oldu -- yani sert kill (bellek).
# A kolu 189 parcayla sorunsuz kosmustu (val Conn_IoU 0.6232).
#
# Bu kol 500 parcalik ALT KUME kullanir ve alt kume RASTGELE DEGIL:
# isaretli tepe sayisina (yogunluga) gore secildi, cunku duvar YOGUN
# parcalarda. Toplam ~689 parca = A'nin 3.6 kati.
#
# 60 epoch: 60 x 689 = 41.340 ornek, A'nin 200 x 189 = 37.800'une YAKIN.
# Yani bu kez kiyas hesap bakimindan DENGELI (ilk denemede B'nin aleyhineydi).
set -u
cd "$(dirname "$0")"
export PYTHONPATH=_diffusion_net_repo/src PYTHONWARNINGS=ignore
INSAN="_label_targets _label_targets_2 _label_targets_3 _label_targets_recall _label_targets_recall_hard"
.venv/Scripts/python.exe train_seg_extra.py --no-extra \
  --partial-dir $INSAN _label_targets_gt2_500 \
  --partial-target connection --seed 0 --k-eig 96 --epochs 60 \
  --checkpoint-out results/seg_extra/gt_C500_s0.pt
