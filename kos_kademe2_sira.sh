#!/usr/bin/env bash
# 0.75/B: ikinci kademeyi SIRA DAMGALAMA blogu ile yeniden egit ve olc.
#
# Onceki kosuda kademe2 (P6_KAFES) 0.3045, kademe1 (P6) 0.3091 verdi -- kaskad
# 0.0046 GERIDEYDI. Sira damgalama blogu (3 sutun) tam da kaskadin zayif oldugu
# yeri hedefler: yogun parcada sira uyeligi, aday-basina skorun goremedigi
# parca-duzeyi bilgidir.
#
# TEK DEGISKEN: `P6_SIRA`. Ayni korpus, ayni katlar, ayni kurallar.
# Cikti ayri dosyaya yazilir; mevcut model paketi EZILMEZ.
set -u
cd "$(dirname "$0")"
export P6_KUME=tam,d6
export P6_DIZIN=results/_p6_oz_u25
export P6_KAT_MIN=200
export P6_OLCUT=makro
export P6_KOLLAR=P6,P6_KAFES
export P6_NMSLER=5.0
export P6_KAHIN=0
export P6_ARAMA_N=250
export P6_NEG_KAT=6
export P6_ITER=200

# DAGITILAN PAKETI KORU. `kos_p6_kademe2.py` cikti yolunu sabit yaziyor ve
# `rejim` alanini URETMIYOR; korumasiz kosarsak D7'de olculen paketin rejim
# kapisi SESSIZCE kaybolurdu.
KORU=results/_paket_koruma_$(date +%s).pkl
cp results/p6_kademe2_model.pkl "$KORU"
echo "dagitilan paket korundu -> $KORU"
geri() { cp -f "$KORU" results/p6_kademe2_model.pkl
         echo "dagitilan paket GERI YUKLENDI"; }
trap geri EXIT

for s in 1 0; do
  echo "=== P6_SIRA=$s ==="
  P6_SIRA=$s python kos_p6_kademe2.py 2>&1 | tail -12
  cp results/p6_kademe2_tam.json "results/p6_kademe2_sira$s.json"
  cp results/p6_kademe2_model.pkl "results/p6_kademe2_model_sira$s.pkl"
done
echo
echo "KIYAS: results/p6_kademe2_sira1.json (sira ACIK) vs sira0.json (KAPALI)"
python - <<'PY'
import json
for s in (1, 0):
    d = json.load(open(f"results/p6_kademe2_sira{s}.json"))["toplam"]
    print(f"  SIRA={s}: " + " | ".join(
        f"{k} {v['robot']:.4f}" for k, v in d.items()))
PY
