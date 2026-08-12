#!/usr/bin/env bash
# S4 olcumunu BELLEK KAPISI arkasinda kos (~6 GB yukluyor).
set -u
cd "$(dirname "$0")"
G=results/_gece; mkdir -p "$G"
L="$G/S4.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$L"; }
bos() { powershell.exe -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)" 2>/dev/null | tr -d '\r'; }

# BASKA AGIR EK ISI VAR MI? RAM kapisi TEK BASINA yetmedi: 10:02 ve 10:05'te
# iki bagimsiz hat kapiyi ayni saniyede gecti, uc is birden yuklendi ve bos
# RAM 0.9 GB'a dustu.
#
# BU FONKSIYON BIR KEZ CAGRILDI AMA TANIMLANMAMISTI (python str.replace
# eslesmeyi bulamayinca sessizce hicbir sey yapmadi; ben de dogrulamadan
# "guncellendi" dedim). Bash'te tanimsiz fonksiyon BOS doner, `${a:-0}` onu
# 0 yapar ve kapi ACILIR -- koruma varmis gibi gorunup hic calismaz.
baska() {
  powershell.exe -NoProfile -Command \
    "(Get-CimInstance Win32_Process | Where-Object { \$_.Name -match 'python' -and \$_.CommandLine -match 'kos_ek_oznitelik' } | Measure-Object).Count" \
    2>/dev/null | tr -d '\r'
}

# KENDI KENDINI DOGRULA: sayi donmuyorsa koruma YOK demektir.
_t=$(baska)
case "${_t:-x}" in
  ''|*[!0-9]*) say "!! KORUMA CALISMIYOR (baska '$_t') -- cikiliyor"; exit 1 ;;
esac
say "koruma dogrulandi (baska -> $_t)"

bek=0
while [ $bek -lt 21600 ]; do
  r=$(bos); r=${r:-0}
  a=$(baska); a=${a:-0}
  if [ "${r%%.*}" -ge 11 ] && [ "$a" -eq 0 ]; then
    sleep 20; a=$(baska); a=${a:-0}
    if [ "$a" -eq 0 ]; then say "KAPI ACIK (${r}GB, baska is yok) -- S4 basliyor"; break; fi
  fi
  say "bekliyor: ${r}GB bos, baska EK isi $a"
  sleep 300; bek=$((bek+300))
done
t0=$(date +%s)
if P6_DIZIN=results/_p6_oz_tam3 P6_KUME=tam,d6 P6_KAT_MIN=200 \
   P6_ARAMA_N=200 P6_ITER=200 P6_NEG_KAT=6 S4_DEVIR=25 \
   python kos_s4_kume_modeli.py >> "$L" 2>&1; then
  say "BITTI ($(( $(date +%s) - t0 ))s)"
  tail -6 "$L" | sed 's/^/    /'
else
  say "DUSTU ($(( $(date +%s) - t0 ))s)"; tail -5 "$L" | sed 's/^/    /'
fi
