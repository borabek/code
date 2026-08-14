#!/usr/bin/env bash
# Y16 — REMESH-VARYANT TOPLULUGU (training GEREKTIRMEZ)
#
# Zincir each parcayi 6000 tepeye remesh eder. Remesh KENDISI a noise
# kaynagidir: different hedef cozunurluk different ucgenleme, different prediction.
# Ayni model, FARKLI remesh hedefleriyle kosulup ciktilar birlestirilirse
# this noise stabilize olabilir.
#
# Yeni training YOK; only inference. `EZ_REMESH` cevre degiskeni with hedef
# tepe sayisi verilir.
set -u
cd "$(dirname "$0")"
for T in 5000 6000 7200; do
  echo "########## remesh hedefi $T ##########"
  EZ_LISTE=results/split3.json EZ_BOLME=val EZ_N=100 \
  EZ_REMESH=$T EZ_DOKUM="results/_dokum_remesh$T.json" \
  .venv/Scripts/python.exe probe_chain_paired_compare.py \
    > "logs/val_remesh$T.log" 2>&1
  grep -aA5 "UC METRIK" "logs/val_remesh$T.log" | tail -4
done
