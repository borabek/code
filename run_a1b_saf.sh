#!/usr/bin/env bash
# A1b: DAGITILABILIR version -- training korpusu buyur AMA measurement set BOZULMAZ.
#
# A1a (415 mesh) "more very data ise yariyor mu" sorusunu SABIT val kumesinde cevaplar,
# but etiketlerinin 52 measurement grubuyla cakismasi yuzunden UCTAN UCA olculemez
# (measurement set 171 -> 119 gruba duserdi = mevcut sayilarla karsilastirilamaz).
#
# A1b etiketleri hem LOCKED hem OLCUM gruplarindan arindirir:
#   _mfg_labels        142 -> 97 part
#   _mfg_labels_highcp 115 ->  7 part  (96'sinin 89'u measurement grubunda -- neredeyse tamami
#                                        olcumu yiyor, karsiliginda hicbir sey vermiyor)
# Egitim 189 -> 293 mesh (1.55x); measurement set 171 grup DEGISMEZ.
#
# KILL: val Conn-IoU dagitilanlarin araligini (0.6378-0.6629) gecmezse arm kapanir.
set -u
export PYTHONPATH=_diffusion_net_repo/src
PY=.venv/Scripts/python.exe
INSAN="_label_targets _label_targets_2 _label_targets_3 _label_targets_recall _label_targets_recall_hard"
MFG="_mfg_labels_SAF _mfg_labels_highcp_SAF"
SEED=${1:-0}
OUT=results/seg_extra/a1b_saf_s${SEED}.pt
echo "=== A1b: saf manufacturer etiketleri, k_eig 96, seed ${SEED}, $(date) ==="
$PY train_seg_extra.py --no-extra --k-eig 96 \
    --partial-dir $INSAN $MFG --partial-target connection --seed ${SEED} \
    --select-metric connection_iou \
    --checkpoint-out "$OUT"
echo "=== BITTI $(date) -> $OUT ==="
