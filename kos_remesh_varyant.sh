#!/usr/bin/env bash
# Y16 — REMESH-VARYANT TOPLULUGU (egitim GEREKTIRMEZ)
#
# Zincir her parcayi 6000 tepeye remesh eder. Remesh KENDISI bir gurultu
# kaynagidir: farkli hedef cozunurluk farkli ucgenleme, farkli tahmin.
# Ayni model, FARKLI remesh hedefleriyle kosulup ciktilar birlestirilirse
# bu gurultu stabilize olabilir.
#
# Yeni egitim YOK; yalniz cikarim. `EZ_REMESH` cevre degiskeni ile hedef
# tepe sayisi verilir.
set -u
cd "$(dirname "$0")"
for T in 5000 6000 7200; do
  echo "########## remesh hedefi $T ##########"
  EZ_LISTE=results/split3.json EZ_BOLME=val EZ_N=100 \
  EZ_REMESH=$T EZ_DOKUM="results/_dokum_remesh$T.json" \
  .venv/Scripts/python.exe sonda_zincir_esli_kiyas.py \
    > "logs/val_remesh$T.log" 2>&1
  grep -aA5 "UC METRIK" "logs/val_remesh$T.log" | tail -4
done
