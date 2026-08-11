#!/usr/bin/env bash
# S1: SPEKTRAL TABAN 96 -> 128
#
# GEREKCE (kendi olcumumuz): k_eig 64 -> 96 bizim TEK plato kiran hamlemizdi
# (F1 0.383 -> 0.477, uc tohumda ucu de kazandi). Orada durduk.
#
# TEZ NE DIYOR (durustce): 128 tezin kendi arama uzayinda (Tablo 3: 32/64/128/256/512/1024)
# AMA tez §1945 o aramayi kosmus ve "64'ten fazla ozvektor asiri ogrenmeye yol acmistir"
# demis; en iyi modelini 64'te birakmis. Yani bu kol tezin YONTEMINE sadik, SONUCUNA degil.
# Mesruiyeti tezden degil, bizim 3/3 tohumlu 96 olcumumuzden geliyor -- orada tezin hukmu
# bizim veride tutmadi (muhtemel sebep: korpus 1913 parcaya cikti, daha cok veri daha cok
# kapasite kaldirir).
#
# TEK DEGISKEN: tohum 0, cunku karsilastirilacagi uye recall_hard_keig96_s0 -- ayni tohum,
# ayni etiket dizinleri, ayni secim metrigi. Tek fark k_eig.
#
# KILL (onceden yazili): VAL kumesinde uctan uca TESPIT F1 +0.01 gelmezse uye ALINMAZ.
# EK IZLEME: tezin uyarisi gercek bir sinyal -- yalniz final F1'e degil, EGITIM/DOGRULAMA
# ACIKLIGINA da bakilacak. Aciklik 96'ya gore buyuyorsa tez hakli demektir ve orada duruyoruz.
set -u
export PYTHONPATH=_diffusion_net_repo/src
PY=.venv/Scripts/python.exe
DIRS="_label_targets _label_targets_2 _label_targets_3 _label_targets_recall _label_targets_recall_hard"
OUT=results/seg_extra/recall_hard_keig128_s0.pt

echo "=== k_eig 128, tohum 0 (karsilastirma: recall_hard_keig96_s0) $(date) ==="
$PY train_seg_extra.py --no-extra --k-eig 128 \
    --partial-dir $DIRS --partial-target connection --seed 0 \
    --select-metric connection_iou \
    --checkpoint-out "$OUT"
echo "=== BITTI $(date) -> $OUT ==="
