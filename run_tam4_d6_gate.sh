#!/usr/bin/env bash
# CEILING-24 GATE A'sini d6 TAMAMLANIR TAMAMLANMAZ olc -- `tam` korpusunu bekleme.
#
# WHY: GATE A'nin kritik sorusu "tavani 24 yapmak YONLU recall'u aciyor mu?"
# ve bunun cevabi d6'da yatiyor -- pool tavanini asagi ceken NIT orada
# (D6 GT'sinin %45.7'si, yonlu recall 0.5254). d6 yalnizca 468 part; 2583
# parcalik `tam` korpusunu beklemek cevabi saatlerce geciktirir.
#
# `run_tam4.sh` already sonunda hem d6 hem tam for GATE A olcuyor; this betik
# onun yerine gecmez, ERKEN CEVAP verir. Ayni receipt adina yazar (cluster adi
# dosyada) -- sonraki measurement same sayiyi uretir.
set -u
cd "$(dirname "$0")"
G=results/_gece
LOG="$G/TAM4_d6_erken.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

BEK=0
while [ $BEK -lt $((8 * 3600)) ]; do
  n=$(ls results/_p6_oz_tam4 2>/dev/null | grep -c '^d6_')
  if [ "$n" -ge 460 ]; then
    say "d6 TAMAM ($n/468) -- GATE A olculuyor"
    HT_ONLER=d6 P6_DIZIN=results/_p6_oz_tam4 python probe_pool_ceiling.py \
      >> "$LOG" 2>&1
    say "--- RESULT ---"
    tail -18 "$LOG" | sed 's/^/    /'
    exit 0
  fi
  say "bekliyor: d6 $n/468"
  sleep 300
  BEK=$((BEK + 300))
done
say "ZAMAN ASIMI -- d6 tamamlanmadi"
