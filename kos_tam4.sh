#!/usr/bin/env bash
# CEILING-24 KORPUSU (`_p6_oz_tam4`). TEK DEGISKEN: YB_MAX_SEC 12 -> 24.
#
# WHY (measured 2026-08-12, probe_max_select.py):
#   UPUN/SUPU 30 part : yonlu recall 0.8462 -> 0.9077  (+0.0615, maliyet 1.20x)
#   NIT       18 part : yonlu recall 0.5913 -> 0.8755  (+0.2842, maliyet 1.33x)
#   ceiling 48'de IKISINDE DE ek kazanc YOK -> diz noktasi 24.
#
# NIT, D6 GT'sinin %45.7'si ve pool tavanini TEK BASINA asagi ceken brand
# (yonlu recall 0.5254; kaybi tam as YON kaybi -- konum recall'u 0.7807).
# Kaba hesapla NIT 0.8755'e cikarsa D6 total yonlu recall 0.7264 -> ~0.886,
# F1 tavani 0.8415 -> ~0.94: GATE A (>=0.85) GECMEMEKTEN GECMEYE doner.
#
# Diger butun ayarlar `_p6_oz_tam3` with BIREBIR same tutulur; otherwise iki corpus
# kiyaslanamaz ve kazanc tavana mi baska a seye mi ait, ayirt edilemez.
#
# 5 pay: makinede 16 cekirdek present ve same anda gece programinin B fazi with EK
# kuyrugu kosuyor; hepsini doyurmamak for 6 not 5.
set -u
cd "$(dirname "$0")"
G=results/_gece
mkdir -p "$G"
KILIT="$G/.kilit_tam4"
if [ -e "$KILIT" ] && kill -0 "$(cat "$KILIT" 2>/dev/null)" 2>/dev/null; then
  echo "ZATEN KOSUYOR (pid $(cat "$KILIT"))"; exit 0
fi
echo $$ > "$KILIT"
trap 'rm -f "$KILIT"' EXIT

ANA="$G/TAM4.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$ANA"; }

say "=== CEILING-24 KORPUSU BASLADI (5 pay) ==="
for i in 0 1 2 3 4; do
  ( BH_MESH_ESIK=0.05 YB_FAN=256 P6_KAYNAK=012 YB_MAX_SEC=24 \
    P6_MESH_R=2.5 P6_MESH_MAX=350 P6_MESH_KAT=5 \
    P6_CIK=results/_p6_oz_tam4 P6_SHARD="$i/5" \
    python run_p6_feature.py d6 > "$G/tam4_d6_$i.log" 2>&1
    BH_MESH_ESIK=0.05 YB_FAN=256 P6_KAYNAK=012 YB_MAX_SEC=24 \
    P6_MESH_R=2.5 P6_MESH_MAX=350 P6_MESH_KAT=5 \
    P6_CIK=results/_p6_oz_tam4 P6_SHARD="$i/5" \
    python run_p6_feature.py tam > "$G/tam4_tam_$i.log" 2>&1 ) &
done
wait
say "=== CEILING-24 CIKARIMI BITTI -- $(ls results/_p6_oz_tam4 2>/dev/null | wc -l) dosya ==="

# GATE A'yi HEM d6 HEM tam kumesinde olc; receipt adlari kumeyi tasir.
for k in d6 tam; do
  say "GATE A olcumu ($k)"
  HT_ONLER=$k P6_DIZIN=results/_p6_oz_tam4 python probe_pool_ceiling.py \
    > "$G/TAM4_kapiA_$k.log" 2>&1
  tail -4 "$G/TAM4_kapiA_$k.log" | sed 's/^/    /' | tee -a "$ANA"
done
say "=== TAM4 BITTI ==="
