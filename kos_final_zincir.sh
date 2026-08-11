#!/usr/bin/env bash
# FINAL ZINCIR -- egitim bitiminden D7 mansetine kadar, elle mudahalesiz.
#
#  1. egitimin model paketini yazmasini bekle
#  2. POZ KAFASI karari: D6 alt kumesinde URUN_P6=1 iken POZ acik/kapali olc,
#     KAZANANI sabitle  (gerekce: `yon_sozluk_sec` yon bankasinin sectigi yonu
#     ezebilir; bu belirsizlikle SINAV okumasi harcanmaz)
#  3. hazirlik kontrolu -- urun yolu calisiyor mu, sutun sayisi tutuyor mu,
#     skorlar dejenere mi
#  4. D7 OKUMA #1: taban + P6 ayni kosuda, 8 pay
#  5. manset: bootstrap GA + marka kirilimi + temiz-703 duyarliligi
#
# Karar kurallari `docs/KAMPANYA_050_RAPOR.md` bolum 5'te OKUMADAN ONCE ilan
# edildi. Bu betik onlari uygular, yeniden secmez.
set -u
cd "$(dirname "$0")"
MODEL=results/p6_kademe2_model.pkl
ISARET=results/_zincir_baslangic

date +%s > "$ISARET"
echo "=== 1/5 EGITIM BEKLENIYOR ==="
# Model dosyasinin ISARET'ten YENI olmasini bekle. `pgrep` Git Bash'te yok;
# bunun yerine bir SURE SINIRI konur -- zincir sonsuza kadar beklemez.
BEKLE_MAKS=${BEKLE_MAKS:-7200}
t0=$(date +%s)
while true; do
  if [ -f "$MODEL" ] && [ "$MODEL" -nt "$ISARET" ]; then
    a=$(stat -c %s "$MODEL"); sleep 10; b=$(stat -c %s "$MODEL")
    [ "$a" = "$b" ] && break        # boyut sabitlendi = yazim bitti
  fi
  if [ $(( $(date +%s) - t0 )) -gt "$BEKLE_MAKS" ]; then
    echo "!! egitim ${BEKLE_MAKS}s icinde model yazmadi -- ZINCIR DURDU"
    exit 1
  fi
  sleep 20
done
echo "model yazildi: $(date)"
python - <<'PY'
import json, pickle
d = pickle.load(open("results/p6_kademe2_model.pkl", "rb"))
print("  kol", d.get("kol"), "| kural", d.get("kural"), "| nms", d.get("nms"),
      "| 2.kademe", "VAR" if d.get("kademe2") is not None else "YOK")
PY

echo
echo "=== 2/5 POZ KAFASI KARARI (D6 alt kumesi, 150 parca) ==="
for poz in 1 0; do
  for i in 0 1 2 3; do
    ( URUN_P6=1 DOG_N=150 DOG_POZ=$poz DOG_SHARD="$i/4" \
      DOG_CIKTI="results/_poz${poz}_$i.json" \
      python sonda_d6_urun.py > "results/_poz${poz}_$i.log" 2>&1 ) &
  done
  wait
  python birlestir_makbuz.py "results/poz_$poz.json" results/_poz${poz}_*.json
done
POZ=$(python - <<'PY'
import json
a = json.load(open("results/poz_1.json"))["sonuc"]["robot"]
b = json.load(open("results/poz_0.json"))["sonuc"]["robot"]
print(f"  POZ ACIK {a:.4f} | POZ KAPALI {b:.4f} -> secilen "
      f"{'ACIK' if a >= b else 'KAPALI'}", file=__import__('sys').stderr)
print(1 if a >= b else 0)
PY
)
echo "  DOG_POZ=$POZ sabitlendi"

echo
echo "=== 3/5 HAZIRLIK KONTROLU ==="
if ! URUN_P6=1 python sonda_p6_hazir.py 3; then
  echo "!! HAZIR DEGIL -- D7 OKUMASI YAPILMADI"; exit 2
fi

echo
echo "=== 4/5 D7 OKUMA #1 ==="
N=8 DOG_POZ=$POZ bash kos_d7_okuma.sh

echo
echo "=== 5/5 BITTI ==="
