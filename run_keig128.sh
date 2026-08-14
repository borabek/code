#!/usr/bin/env bash
# S1: SPEKTRAL BASELINE 96 -> 128
#
# RATIONALE (kendi olcumumuz): k_eig 64 -> 96 bizim TEK plato kiran hamlemizdi
# (F1 0.383 -> 0.477, uc tohumda ucu de kazandi). Orada durduk.
#
# TEZ NE DIYOR (durustce): 128 tezin kendi arama uzayinda (Tablo 3: 32/64/128/256/512/1024)
# AMA tez §1945 o aramayi kosmus ve "64'ten excess ozvektor asiri ogrenmeye path acmistir"
# demis; en iyi modelini 64'te birakmis. Yani this arm tezin YONTEMINE sadik, SONUCUNA not.
# Mesruiyeti tezden not, bizim 3/3 tohumlu 96 olcumumuzden geliyor -- orada tezin hukmu
# bizim veride tutmadi (muhtemel reason: corpus 1913 parcaya cikti, more very data more very
# kapasite kaldirir).
#
# TEK DEGISKEN: seed 0, because karsilastirilacagi uye recall_hard_keig96_s0 -- same seed,
# same etiket dizinleri, same secim metrigi. Tek diff k_eig.
#
# KILL (onceden yazili): VAL kumesinde uctan uca TESPIT F1 +0.01 gelmezse uye ALINMAZ.
# EK IZLEME: tezin uyarisi gercek a sinyal -- only final F1'e not, EGITIM/DOGRULAMA
# ACIKLIGINA da bakilacak. Aciklik 96'ya per buyuyorsa tez hakli demektir ve orada duruyoruz.
set -u
export PYTHONPATH=_diffusion_net_repo/src
PY=.venv/Scripts/python.exe
DIRS="_label_targets _label_targets_2 _label_targets_3 _label_targets_recall _label_targets_recall_hard"
OUT=results/seg_extra/recall_hard_keig128_s0.pt

echo "=== k_eig 128, seed 0 (comparison: recall_hard_keig96_s0) $(date) ==="
$PY train_seg_extra.py --no-extra --k-eig 128 \
    --partial-dir $DIRS --partial-target connection --seed 0 \
    --select-metric connection_iou \
    --checkpoint-out "$OUT"
echo "=== BITTI $(date) -> $OUT ==="
