#!/usr/bin/env bash
# TAVAN-24 UCTAN UCA: ceiling kazanci gercekten F1'e donuyor mu?
#
# KAPI A bir TAVAN olcumudur: "mukemmel selector olsa ne alirdik". Tavanin
# yukselmesi, selector o yonleri SECEBILIRSE F1'e doner -- secemezse ceiling
# bosuna yukselir. Bu betik uctan uca farki olcer.
#
# NEDEN d6 UZERINDE: `tam` bolumu her iki korpusta da saatler surecek, ama d6
# IKISINDE DE TAMAM. d6 gelistirme kumesidir (SINAV DEGIL), brand katlariyla
# LOMO kurulabilir. Tek degisken KORPUS (ceiling 12 vs 24); ayni parts, ayni
# katlar, ayni seed, ayni kurallar.
#
# UYARI: d6'nin TABANI zayiftir ve bu kampanyada bir kez ALDATTI (D6'da +0.1681
# vaat eden arm `tam` katlarinda +0.0230 verdi). Buradaki sayi YON GOSTERIR,
# headline DEGILDIR; `tam` korpusu bitince orada tekrarlanir.
set -u
cd "$(dirname "$0")"
G=results/_gece
mkdir -p "$G"
ANA="$G/UCTAN_UCA.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$ANA"; }

KAT_MIN=${UU_KAT_MIN:-50}

say "=== TAVAN UCTAN UCA (d6, kat_min=$KAT_MIN) ==="
for ad in tam3 tam4; do
  dz="results/_p6_oz_$ad"
  n=$(ls "$dz" 2>/dev/null | grep -c '^d6_')
  say "$ad: $n d6 parcasi"
  t0=$(date +%s)
  if P6_DIZIN="$dz" P6_KUME=d6 P6_KAT_MIN=$KAT_MIN P6_KOLLAR=P6 \
     P6_NMSLER=5.0 P6_ARAMA_N=120 P6_NEG_KAT=6 P6_ITER=200 \
     python run_p6_kademe2.py > "$G/UU_$ad.log" 2>&1; then
    cp -f results/p6_kademe2_tam.json "results/uctan_uca_$ad.json"
    r=$(python -c "
import json; d=json.load(open('results/uctan_uca_$ad.json'))
t=d['toplam']['P6']
print(f\"robot {t['robot']:.4f} recall {t['recall']:.4f} precision {t['precision']:.4f}\")")
    say "BITTI: $ad ($(( $(date +%s) - t0 ))s) -> $r"
  else
    say "DUSTU: $ad"
    tail -4 "$G/UU_$ad.log" | sed 's/^/    /' | tee -a "$ANA"
  fi
done

python - <<'PY' 2>&1 | tee -a "$ANA"
import json, os
r = {}
for ad in ("tam3", "tam4"):
    y = f"results/uctan_uca_{ad}.json"
    if os.path.exists(y):
        r[ad] = json.load(open(y))["toplam"]["P6"]
if len(r) == 2:
    a, b = r["tam3"], r["tam4"]
    print("\n=== TAVAN UCTAN UCA (d6 brand katlari) ===")
    print(f"{'':<14}{'ceiling 12':>12}{'ceiling 24':>12}{'fark':>10}")
    for k in ("robot", "recall", "precision"):
        print(f"{k:<14}{a[k]:>12.4f}{b[k]:>12.4f}{b[k]-a[k]:>+10.4f}")
    print("\nNOT: d6 GELISTIRME kumesidir, headline DEGIL. Yon gosterir.")
PY
say "=== BITTI ==="
