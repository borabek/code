#!/usr/bin/env bash
# 0.75/A: YELPAZE uctan uca. Tek variable -- same corpus, same katlar, same
# kurallar; diff only `P6_DIZIN` (yelpazesiz `_p6_oz_u25` vs yelpazeli
# `_p6_oz_fan`).
#
# Yelpaze havuza yeni YON secenekleri katar (option +%30) but SUTUN SAYISINI
# degistirmez, i.e. kiyas dogrudan yapilabilir.
#
# Dagitilan model paketi KORUNUR/GERI YUKLENIR.
set -u
cd "$(dirname "$0")"
KORU=results/_paket_koruma_fan_$(date +%s).pkl
cp results/p6_kademe2_model.pkl "$KORU"
echo "dagitilan paket korundu -> $KORU"
geri() { cp -f "$KORU" results/p6_kademe2_model.pkl
         echo "dagitilan paket GERI YUKLENDI"; }
trap geri EXIT

export P6_KUME=tam,d6
export P6_KAT_MIN=200
export P6_OLCUT=makro
export P6_KOLLAR=P6
export P6_NMSLER=5.0
export P6_KAHIN=0
export P6_ARAMA_N=250
export P6_NEG_KAT=6
export P6_ITER=200
export P6_SIRA=0

for d in _p6_oz_u25 _p6_oz_fan; do
  echo "=== P6_DIZIN=results/$d ==="
  P6_DIZIN="results/$d" python run_p6_kademe2.py 2>&1 | tail -8
  cp results/p6_kademe2_full.json "results/p6_yelpaze_$d.json"
done
echo
python - <<'PY'
import json
for d in ("_p6_oz_u25", "_p6_oz_fan"):
    t = json.load(open(f"results/p6_yelpaze_{d}.json"))["total"]
    ad = "YELPAZESIZ" if d.endswith("u25") else "YELPAZELI "
    print(f"  {ad}: " + " | ".join(f"{k} {v['robot']:.4f}" for k, v in t.items()))
PY
