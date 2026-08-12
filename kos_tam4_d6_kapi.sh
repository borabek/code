#!/usr/bin/env bash
# TAVAN-24 KAPI A'sini d6 TAMAMLANIR TAMAMLANMAZ olc -- `tam` korpusunu bekleme.
#
# NEDEN: KAPI A'nin kritik sorusu "tavani 24 yapmak YONLU recall'u aciyor mu?"
# ve bunun cevabi d6'da yatiyor -- havuz tavanini asagi ceken NIT orada
# (D6 GT'sinin %45.7'si, yonlu recall 0.5254). d6 yalnizca 468 parca; 2583
# parcalik `tam` korpusunu beklemek cevabi saatlerce geciktirir.
#
# `kos_tam4.sh` zaten sonunda hem d6 hem tam icin KAPI A olcuyor; bu betik
# onun yerine gecmez, ERKEN CEVAP verir. Ayni makbuz adina yazar (kume adi
# dosyada) -- sonraki olcum ayni sayiyi uretir.
set -u
cd "$(dirname "$0")"
G=results/_gece
LOG="$G/TAM4_d6_erken.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

BEK=0
while [ $BEK -lt $((8 * 3600)) ]; do
  n=$(ls results/_p6_oz_tam4 2>/dev/null | grep -c '^d6_')
  if [ "$n" -ge 460 ]; then
    say "d6 TAMAM ($n/468) -- KAPI A olculuyor"
    HT_ONLER=d6 P6_DIZIN=results/_p6_oz_tam4 python sonda_havuz_tavani.py \
      >> "$LOG" 2>&1
    say "--- SONUC ---"
    tail -18 "$LOG" | sed 's/^/    /'
    exit 0
  fi
  say "bekliyor: d6 $n/468"
  sleep 300
  BEK=$((BEK + 300))
done
say "ZAMAN ASIMI -- d6 tamamlanmadi"
