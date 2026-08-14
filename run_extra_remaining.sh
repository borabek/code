#!/usr/bin/env bash
# YALNIZ MAKBUZU OLMAYAN EK BLOKLARINI KOS.
#
# WHY: `run_night.sh` faz duzeyinde DEVAM EDEMIYOR. Makine kapanip acilinca
# zincirin BASINDAN basliyor ve biten fazlari (B1 46 dk, B6 60 dk, ozkalib
# 27 dk) YENIDEN kosuyor -- ~2.2 saat mukerrer is. Bu betik each blogun
# makbuzuna bakar ve VAR OLANI ATLAR.
#
# `depth` for EK_ISCI=4: paralel path bit-same sonuc verdigi DOGRULANDI
# (tests/test_extra_paralel_esdeger.py). Tek cekirdekte ~8.5 saat, 4 iscide ~2.
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

# BASKA AGIR EK ISI VAR MI?
# 10:02 ve 10:05'te iki independent hat (this betik + run_extra_queue.sh) kapilarini
# same saniyede gecti, ucu birden ~6'sar GB yukledi ve bos RAM 0.9 GB'a dustu.
# RAM kapisi TEK BASINA yetmiyor -- isin KENDISI sorulmali.
#
# NOT: this fonksiyon a kez CAGRILDI but TANIMLANMAMISTI (python str.replace
# eslesmeyi bulamayinca SESSIZCE hicbir sey yapmamisti). Bash'te tanimsiz
# fonksiyon bos doner, `${a:-0}` onu 0 yapar ve gate ACILIR -- i.e. koruma
# varmis gibi gorunup no calismaz.
baska_ek() {
  powershell.exe -NoProfile -Command \
    "(Get-CimInstance Win32_Process | Where-Object { \$_.Name -match 'python' -and \$_.CommandLine -match 'run_extra_feature|kos_s4_kume' } | Measure-Object).Count" \
    2>/dev/null | tr -d '\r'
}

# KENDI KENDINI DOGRULA: fonksiyon number dondurmuyorsa koruma YOK demektir.
_t=$(baska_ek)
case "${_t:-x}" in
  ''|*[!0-9]*) say "!! KORUMA CALISMIYOR (baska_ek '$_t' dondu) -- cikiliyor"
               exit 1 ;;
esac
say "koruma dogrulandi (baska_ek -> $_t)"

say "=== KALAN EK BLOKLARI ==="
for blok in kafes_adet symmetry depth; do
  if [ -f "results/ek_blok_$blok.json" ]; then
    say "ATLANDI: $blok (makbuzu present)"
    continue
  fi
  # bellek kapisi: iki agir EK isi ust uste binmesin
  bek=0
  while [ $bek -lt 14400 ]; do
    r=$(bos_ram); r=${r:-0}
    a=$(baska_ek); a=${a:-0}
    if [ "${r%%.*}" -ge 10 ] && [ "$a" -eq 0 ]; then
      sleep 20                       # yaris kirici: kisa bekleyip TEKRAR bak
      a=$(baska_ek); a=${a:-0}
      [ "$a" -eq 0 ] && break
    fi
    say "  $blok bekliyor: ${r}GB bos, baska EK isi $a"
    sleep 300; bek=$((bek + 300))
  done

  isci=1
  [ "$blok" = "depth" ] && isci=4      # dogrulanmis paralel path
  say "BASLIYOR: $blok (isci $isci)"
  t0=$(date +%s)
  if EK_BLOK="$blok" EK_ISCI=$isci P6_DIZIN=results/_p6_oz_tam3 \
       P6_KUME=tam,d6 P6_KAT_MIN=200 P6_ARAMA_N=200 P6_ITER=200 \
       P6_NEG_KAT=6 python run_extra_feature.py > "$G/EK_$blok.log" 2>&1; then
    say "BITTI: $blok ($(( $(date +%s) - t0 ))s) -- $(tail -3 "$G/EK_$blok.log" | head -2 | tr '\n' ' ')"
  else
    say "DUSTU: $blok ($(( $(date +%s) - t0 ))s)"
    tail -4 "$G/EK_$blok.log" | sed 's/^/    /' | tee -a "$ANA"
  fi
done
say "=== KALAN BLOKLAR BITTI ==="
