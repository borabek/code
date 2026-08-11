#!/bin/sh
# Egitim sirasinda disk kritige inerse SUREC TEMIZ DURDURULUR.
# Gerekce: bu makinede disk dolunca is yarim kalip bozuk durum birakiyor
# (bkz. disk-dolunca-gmsh-donuyor). Nobetci 30s araliklidir; 145s GEC KALIYOR.
ESIK_MB=700
while true; do
  BOS=$(df -m /c | tail -1 | awk '{print $4}')
  if [ "$BOS" -lt "$ESIK_MB" ]; then
    echo "$(date +%H:%M:%S) DISK ESIGI ($BOS MB) -- egitim durduruluyor" >> results/_nobetci.log
    powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'train_seg_extra' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
    exit 1
  fi
  if ! powershell.exe -NoProfile -Command "if ((Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'train_seg_extra' }).Count -gt 0) { exit 0 } else { exit 1 }" ; then
    echo "$(date +%H:%M:%S) egitim bitti, nobetci cikiyor (bos $BOS MB)" >> results/_nobetci.log
    exit 0
  fi
  sleep 30
done
