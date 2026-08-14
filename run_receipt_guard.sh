#!/usr/bin/env bash
# ORKESTRATOR HER KADEME2 KOSUSUNU AYNI DOSYAYA YAZIYOR
# (results/p6_kademe2_full.json). B1'in makbuzu B6 tarafindan, B6'nin makbuzu
# da sonraki kosu tarafindan EZILIR. Bu gozcu each faz bitiminde makbuzu ayri
# ada kopyalar -- otherwise gecenin sonunda elde yalnizca EN SON kosu kalir.
set -u
cd "$(dirname "$0")"
LOG=results/_gece/MAKBUZ_KORU.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "gozcu basladi"
for i in $(seq 1 480); do
  if grep -q "BITTI: B6_ensemble" results/_gece/ANA.log 2>/dev/null \
     && [ ! -f results/p6_kademe2_B6_ensemble.json ]; then
    cp -f results/p6_kademe2_full.json results/p6_kademe2_B6_ensemble.json
    say "B6 makbuzu korundu"
    exit 0
  fi
  sleep 120
done
say "zaman asimi"
