#!/usr/bin/env bash
# YALNIZ MAKBUZU OLMAYAN EK BLOKLARINI KOS.
#
# NEDEN: `kos_gece.sh` faz duzeyinde DEVAM EDEMIYOR. Makine kapanip acilinca
# zincirin BASINDAN basliyor ve biten fazlari (B1 46 dk, B6 60 dk, ozkalib
# 27 dk) YENIDEN kosuyor -- ~2.2 saat mukerrer is. Bu betik her blogun
# makbuzuna bakar ve VAR OLANI ATLAR.
#
# `derinlik` icin EK_ISCI=4: paralel yol bit-ayni sonuc verdigi DOGRULANDI
# (tests/test_ek_paralel_esdeger.py). Tek cekirdekte ~8.5 saat, 4 iscide ~2.
set -u
cd "$(dirname "$0")"
G=results/_gece
mkdir -p "$G"
KILIT="$G/.kilit_kalan"
if [ -e "$KILIT" ] && kill -0 "$(cat "$KILIT" 2>/dev/null)" 2>/dev/null; then
  echo "ZATEN KOSUYOR"; exit 0
fi
echo $$ > "$KILIT"
trap 'rm -f "$KILIT"' EXIT

ANA="$G/EK_KALAN.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$ANA"; }

bos_ram() {
  powershell.exe -NoProfile -Command \
    "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)" \
    2>/dev/null | tr -d '\r'
}

say "=== KALAN EK BLOKLARI ==="
for blok in kafes_adet simetri derinlik; do
  if [ -f "results/ek_blok_$blok.json" ]; then
    say "ATLANDI: $blok (makbuzu var)"
    continue
  fi
  # bellek kapisi: iki agir EK isi ust uste binmesin
  bek=0
  while [ $bek -lt 7200 ]; do
    r=$(bos_ram); r=${r:-0}
    [ "${r%%.*}" -ge 10 ] && break
    say "  $blok bekliyor: ${r}GB bos"
    sleep 300; bek=$((bek + 300))
  done

  isci=1
  [ "$blok" = "derinlik" ] && isci=4      # dogrulanmis paralel yol
  say "BASLIYOR: $blok (isci $isci)"
  t0=$(date +%s)
  if EK_BLOK="$blok" EK_ISCI=$isci P6_DIZIN=results/_p6_oz_tam3 \
       P6_KUME=tam,d6 P6_KAT_MIN=200 P6_ARAMA_N=200 P6_ITER=200 \
       P6_NEG_KAT=6 python kos_ek_oznitelik.py > "$G/EK_$blok.log" 2>&1; then
    say "BITTI: $blok ($(( $(date +%s) - t0 ))s) -- $(tail -3 "$G/EK_$blok.log" | head -2 | tr '\n' ' ')"
  else
    say "DUSTU: $blok ($(( $(date +%s) - t0 ))s)"
    tail -4 "$G/EK_$blok.log" | sed 's/^/    /' | tee -a "$ANA"
  fi
done
say "=== KALAN BLOKLAR BITTI ==="
