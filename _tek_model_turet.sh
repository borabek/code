#!/usr/bin/env bash
# VAL Conn-IoU uctan uca ongoruyor mu? Tek-model turetmeleri (ensemble etkisi YOK).
set -u
export PYTHONWARNINGS=ignore PYTHONPATH=_diffusion_net_repo/src PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
for ck in keig128_s1 keig96_s0 keig128_s0 keig96_s3; do
  out="results/_der_tek_$ck.pkl"
  [ -f "$out" ] && { echo "$ck zaten var"; continue; }
  echo "=== $ck $(date +%H:%M) ==="
  $PY -u turet.py --cikti "$out" --ckpt "results/seg_extra/recall_hard_$ck.pt" 2>/dev/null \
    | grep -E "kayit ->|GT toplam|candidate toplam"
done
echo "=== TUMU BITTI $(date +%H:%M) ==="
