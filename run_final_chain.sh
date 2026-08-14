#!/usr/bin/env bash
# FINAL ZINCIR -- training bitiminden D7 mansetine up to, elle mudahalesiz.
#
#  1. egitimin model paketini yazmasini bekle
#  2. POZ KAFASI karari: D6 alt kumesinde URUN_P6=1 iken POZ acik/kapali olc,
#     KAZANANI sabitle  (rationale: `pick_direction_from_dictionary` direction bankasinin sectigi direction
#     ezebilir; this belirsizlikle SINAV okumasi harcanmaz)
#  3. hazirlik kontrolu -- urun yolu calisiyor mu, sutun sayisi tutuyor mu,
#     skorlar dejenere mi
#  4. D7 OKUMA #1: baseline + P6 same kosuda, 8 pay
#  5. headline: bootstrap GA + brand kirilimi + temiz-703 duyarliligi
#
# Karar kurallari `docs/KAMPANYA_050_report.md` bolum 5'te OKUMADAN ONCE ilan
# edildi. Bu betik onlari uygular, yeniden secmez.
set -u
cd "$(dirname "$0")"
MODEL=results/p6_kademe2_model.pkl
ISARET=results/_zincir_baslangic

date +%s > "$ISARET"
echo "=== 1/5 EGITIM BEKLENIYOR ==="
# Model dosyasinin ISARET'ten YENI olmasini bekle. `pgrep` Git Bash'te none;
# bunun yerine a SURE SINIRI konur -- zincir sonsuza up to beklemez.
BEKLE_MAKS=${BEKLE_MAKS:-7200}
t0=$(date +%s)
while true; do
  if [ -f "$MODEL" ] && [ "$MODEL" -nt "$ISARET" ]; then
    a=$(stat -c %s "$MODEL"); sleep 10; b=$(stat -c %s "$MODEL")
    [ "$a" = "$b" ] && break        # boyut sabitlendi = yazim bitti
  fi
  if [ $(( $(date +%s) - t0 )) -gt "$BEKLE_MAKS" ]; then
    echo "!! training ${BEKLE_MAKS}s icinde model yazmadi -- ZINCIR DURDU"
    exit 1
  fi
  sleep 20
done
echo "model yazildi: $(date)"
python - <<'PY'
import json, pickle
d = pickle.load(open("results/p6_kademe2_model.pkl", "rb"))
print("  arm", d.get("arm"), "| rule", d.get("rule"), "| nms", d.get("nms"),
      "| 2.kademe", "VAR" if d.get("kademe2") is not None else "YOK")
PY

echo
echo "=== 2/5 POZ KAFASI KARARI (D6 alt set, 150 part) ==="
for poz in 1 0; do
  for i in 0 1 2 3; do
    ( URUN_P6=1 DOG_N=150 DOG_POZ=$poz DOG_SHARD="$i/4" \
      DOG_CIKTI="results/_poz${poz}_$i.json" \
      python probe_d6_product.py > "results/_poz${poz}_$i.log" 2>&1 ) &
  done
  wait
  python merge_receipt.py "results/poz_$poz.json" results/_poz${poz}_*.json
done
POZ=$(python - <<'PY'
import json
a = json.load(open("results/poz_1.json"))["sonuc"]["robot"]
b = json.load(open("results/poz_0.json"))["sonuc"]["robot"]
print(f"  POZ OPEN {a:.4f} | POZ KAPALI {b:.4f} -> selected "
      f"{'OPEN' if a >= b else 'KAPALI'}", file=__import__('sys').stderr)
print(1 if a >= b else 0)
PY
)
echo "  DOG_POZ=$POZ sabitlendi"

echo
echo "=== 3/5 HAZIRLIK KONTROLU ==="
if ! URUN_P6=1 python probe_p6_hazir.py 3; then
  echo "!! HAZIR DEGIL -- D7 OKUMASI YAPILMADI"; exit 2
fi

echo
echo "=== 4/5 D7 OKUMA #1 ==="
N=8 DOG_POZ=$POZ bash run_d7_read.sh

echo
echo "=== 5/5 BITTI ==="
