#!/usr/bin/env bash
# TAM4 CIKARIMINI, AGIR ISLER BITINCE OTOMATIK SURDUR.
#
# WHY DURDURULDU (2026-08-12 10:32): S4 (~6 GB) + EK blogu (~6 GB) + 6
# inference iscisi (~6 GB) 31 GB RAM'i zorladi; Windows `pagefile.sys`'i
# 23.9 GB'a buyuttu ve C: bos alani 7.28 -> 3.34 GB'a dustu. Cikarim
# durdurulunca disk 13.11 GB'a FIRLADI -- i.e. disk sorunu a veri sismesi
# not, BELLEK BASKISININ yan etkisiydi.
#
# Cikarim atla-present-olani calistigi for kaldigi yerden devam eder.
set -u
cd "$(dirname "$0")"
G=results/_gece; mkdir -p "$G"
L="$G/CIKARIM_GOZCU.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$L"; }

agir() {
  powershell.exe -NoProfile -Command \
    "(Get-CimInstance Win32_Process | Where-Object { \$_.Name -match 'python' -and \$_.CommandLine -match 'run_extra_feature|kos_s4_kume|kademe2' } | Measure-Object).Count" \
    2>/dev/null | tr -d '\r'
}
bos() {
  powershell.exe -NoProfile -Command \
    "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)" \
    2>/dev/null | tr -d '\r'
}

# KENDI KENDINI DOGRULA (bugun uc kez "koruma present sanildi but yoktu")
_t=$(agir)
case "${_t:-x}" in
  ''|*[!0-9]*) say "!! KORUMA CALISMIYOR (agir '$_t') -- cikiliyor"; exit 1 ;;
esac
say "gozcu basladi (agir is -> $_t)"

bek=0
while [ $bek -lt 43200 ]; do
  n=$(ls results/_p6_oz_tam4 2>/dev/null | grep -c '^tam_')
  if [ "$n" -ge 2570 ]; then say "inference ZATEN TAM ($n/2583)"; exit 0; fi
  a=$(agir); a=${a:-0}
  r=$(bos);  r=${r:-0}
  if [ "$a" -eq 0 ] && [ "${r%%.*}" -ge 12 ]; then
    say "AGIR IS YOK (${r}GB bos) -- inference surduruluyor ($n/2583)"
    bash kos_tam4_devam.sh
    say "inference turu bitti"
    exit 0
  fi
  say "bekliyor: $n/2583 dosya, agir is $a, ${r}GB bos"
  sleep 600
  bek=$((bek + 600))
done
say "ZAMAN ASIMI"
