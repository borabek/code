#!/usr/bin/env bash
# E2: HIC KOSULMAMIS DENEY -- dagitilan tarif + URETICI ETIKETLERI.
#
# DENETIM BULGUSU: dagitilan 4 checkpoint yalnizca 189 mesh gordu. Diskte 257 manufacturer-etiketli
# part (_mfg_labels 142 + _mfg_labels_highcp 115) duruyor ve HIC kullanilmadi.
#
# "Daha very data" IKI KEZ denenmis, IKISI DE KAYBETMIS -- but IKISI DE KARISTIRILMIS deneydi:
#   H3: same anda (a) +257 manufacturer etiketi, (b) +132 EEC, (c) 4 INSAN DIZINI DUSURULDU
#       (i.e. urunun insan etiketlerinin %77'si atildi). Kaybin kaynagi olculmedi.
#   G : bambaska a ag (ikili detection basi, width128/blocks4/dropout0 -- tezin "asiri ogrenme"
#       diye isaretledigi yapilandirma), sifirdan, 26/40 epoch'ta kesildi.
#
# BU KOSU TEK DEGISKEN OYNATIR: dagitilan tarifin AYNISI + manufacturer etiketleri EK as.
#
# LEAKAGE KORUMASI (measured, this yuzden TEMIZ kopyalar kullaniliyor):
#   _mfg_labels/train        142 parcanin 12'si LOCKED GRUBUNDA -> atildi (130 kaldi)
#   _mfg_labels_highcp/train 115 parcanin 19'u LOCKED GRUBUNDA -> atildi ( 96 kaldi)
#   Filtresiz training 95 LOCKED grubunun 31'ini yakardi = exam biterdi.
#
# KILL: VAL Conn-IoU dagitilanlarin araligini (0.6378-0.6629) gecmezse arm kapanir.
set -u
export PYTHONPATH=_diffusion_net_repo/src
PY=.venv/Scripts/python.exe
INSAN="_label_targets _label_targets_2 _label_targets_3 _label_targets_recall _label_targets_recall_hard"
MFG="_mfg_labels_TEMIZ _mfg_labels_highcp_TEMIZ"
OUT=results/seg_extra/e2_mfg_s0.pt
echo "=== E2: dagitilan tarif + manufacturer etiketleri, k_eig 96, seed 0, $(date) ==="
echo "    insan dizinleri : $INSAN"
echo "    manufacturer dizinleri: $MFG"
$PY train_seg_extra.py --no-extra --k-eig 96 \
    --partial-dir $INSAN $MFG --partial-target connection --seed 0 \
    --select-metric connection_iou \
    --checkpoint-out "$OUT"
echo "=== BITTI $(date) -> $OUT ==="
