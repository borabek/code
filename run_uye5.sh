#!/usr/bin/env bash
# L1: BAGIMSIZ 5. TOPLULUK UYESI (k_eig 96, seed 3).
#
# RATIONALE (this night measured, results/k_log.txt): gate'in 13 ozelliginden yalnizca `votes` durust
# geometri bolmesinde transfer ediyor (AUC dususu 0.018; digerleri 0.12-0.18 dusuyor). `votes`
# = kac independent uyenin same acikligi gordugu. Yani tek gercekten genelleyen sinyalin
# COZUNURLUGU uye sayisiyla sinirli: 4 uye with oy only {1,2,3,4} degerlerini alabiliyor.
#
# L (turetilmis 5. uye = probability ortalamasi) MEASURED ve KAYBETTI (detection 0.6308 -> 0.6034):
# turetilmis uye BAGIMSIZ not, oy cozunurlugu katmiyor. Bu kosu gercekten independent a
# tohumla same tarifi tekrarlar -- tek farki rastgele baslangic ve data sirasi.
#
# KILL: VAL kumesinde detection F1 +0.01 gelmezse uye alinmaz (urun 4 uyeyle kalir).
set -u
export PYTHONPATH=_diffusion_net_repo/src
PY=.venv/Scripts/python.exe
DIRS="_label_targets _label_targets_2 _label_targets_3 _label_targets_recall _label_targets_recall_hard"
OUT=results/seg_extra/recall_hard_keig96_s3.pt

echo "=== 5. uye: k_eig 96, seed 3, $(date) ==="
echo "    etiket dizinleri: $DIRS"
$PY train_seg_extra.py --no-extra --k-eig 96 \
    --partial-dir $DIRS --partial-target connection --seed 3 \
    --select-metric connection_iou \
    --checkpoint-out "$OUT"
echo "=== BITTI $(date) -> $OUT ==="
