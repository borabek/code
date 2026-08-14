#!/usr/bin/env bash
# FB-2: TEL/ALET yardimci supervizyonuyla 4 uyeli ensemble.
#
# DIAGNOSIS: tezin Contact sinifi "Kontaktierung bzw. Werkzeugeinschub" -- tel girisi ve alet
# mouth TEK SINIF. Backbone, bizim ayirmak istedigimiz iki seyi BIRLESTIRMEK about to egitildi.
# Adli tip raporu: "DENENMEMIS tek yapisal ML duzeltmesi".
# TEZ IHLALI YOK: ana 5-sinif kafasi ve kaybi diff with dogrulandi, BIT DUZEYINDE same.
#
# IZLEME UYARISI (2026-08-03'te saatler kaybettirdi): Git Bash `ps` komut satirini
# GOSTERMEZ -> `ps aux | grep train_seg_extra` HER ZAMAN 0 doner ve `pkill -f` hicbir sey
# oldurmez. Bu yuzden kosan a egitimi 3 kez yeniden baslattim ve 5 surec same anda CPU
# for kavga etti. DURUM KONTROLU ICIN: PowerShell Get-CimInstance Win32_Process.
# Bu betik ilerlemeyi DISKE yazar (results/_fb2_ilerleme.txt) ki kontrol surece bagli olmasin.
set -u
export PYTHONWARNINGS=ignore PYTHONPATH=_diffusion_net_repo/src PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
DIRS="_label_targets _label_targets_2 _label_targets_3 _label_targets_recall _label_targets_recall_hard"
IZ=results/_fb2_ilerleme.txt
: > "$IZ"
for SEED in 0 1 2 3; do
  OUT=results/seg_extra/fb2_aux_s${SEED}.pt
  echo "seed ${SEED} BASLADI $(date +%H:%M:%S)" >> "$IZ"
  $PY train_seg_extra.py --no-extra --k-eig 96 \
      --partial-dir $DIRS --partial-target connection --seed ${SEED} \
      --select-metric connection_iou \
      --aux-wire --aux-w 0.5 --aux-pos-weight 0.54 \
      --checkpoint-out "$OUT"
  echo "seed ${SEED} BITTI   $(date +%H:%M:%S) cikis=$?" >> "$IZ"
done
echo "TUMU BITTI $(date +%H:%M:%S)" >> "$IZ"
