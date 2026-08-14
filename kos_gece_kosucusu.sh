#!/usr/bin/env bash
# GECE KOSUCUSU — karar gerektirmeyen adimlari kendi kendine zincirler.
#
# 1) `_p6_oz_tam4` cikarimi bitene up to bekler (present olani atlar, guvenli)
# 2) URETIM modelini TAM corpus + neg=12 with yeniden egitir
# 3) Uctan uca measures ve receipt birakir
#
# Her adim kendi logunu yazar; hicbiri onceki adimin ciktisini VARSAYMAZ,
# eksikse high sesle durur (this projede sessiz no-op defalarca yandi).
set -u
cd "$(dirname "$0")"
PY=".venv/Scripts/python.exe"
G="logs/gece"; mkdir -p "$G"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$G/ana.log"; }

say "=== GECE KOSUCUSU BASLADI ==="

# --- 1) cikarimin bitmesini bekle
say "1) inference bekleniyor..."
ONCE=-1
while true; do
  N=$(ls results/_p6_oz_tam4/tam_*.npz 2>/dev/null | wc -l)
  CANLI=$(ps -W 2>/dev/null | grep -c "python" || echo 0)
  if [ "$N" -ge 2560 ]; then say "   inference BITTI: $N dosya"; break; fi
  if [ "$N" = "$ONCE" ]; then
    DURGUN=$((DURGUN+1))
  else
    DURGUN=0; ONCE=$N
    say "   inference $N/2583"
  fi
  # 40 dakika no ilerlemezse inference olmustur; elimizdekiyle devam et
  if [ "${DURGUN:-0}" -ge 8 ]; then
    say "   inference DURDU ($N dosya) -- elimizdekiyle devam"; break
  fi
  sleep 300
done

N=$(ls results/_p6_oz_tam4/tam_*.npz 2>/dev/null | wc -l)
if [ "$N" -lt 1500 ]; then
  say "!! inference very missing ($N) -- training ATLANDI"; exit 1
fi

# --- 2) URETIM modelini yeniden egit (tam corpus + neg=12)
say "2) uretim modeli yeniden egitiliyor (tam+d6 corpus, neg=12)..."
cp -f results/p6_kademe2_model.pkl "results/p6_kademe2_model.pkl.gece_oncesi" 2>/dev/null || true
P6_KUME=tam,d6 P6_KAT_MIN=60 $PY run_p6_kademe2.py > "$G/uretim_egitim.log" 2>&1
if grep -q "receipt ->" "$G/uretim_egitim.log"; then
  say "   training BITTI"
  grep -A6 "arm             robot" "$G/uretim_egitim.log" | tail -6 | tee -a "$G/ana.log"
else
  say "!! training TAMAMLANMADI -- son satirlar:"
  tail -5 "$G/uretim_egitim.log" | tee -a "$G/ana.log"
fi

say "=== GECE KOSUCUSU BITTI ==="
