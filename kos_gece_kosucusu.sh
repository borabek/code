#!/usr/bin/env bash
# GECE KOSUCUSU — karar gerektirmeyen adimlari kendi kendine zincirler.
#
# 1) `_p6_oz_tam4` cikarimi bitene kadar bekler (var olani atlar, guvenli)
# 2) URETIM modelini TAM korpus + neg=12 ile yeniden egitir
# 3) Uctan uca olcer ve makbuz birakir
#
# Her adim kendi logunu yazar; hicbiri onceki adimin ciktisini VARSAYMAZ,
# eksikse yuksek sesle durur (bu projede sessiz no-op defalarca yandi).
set -u
cd "$(dirname "$0")"
PY=".venv/Scripts/python.exe"
G="logs/gece"; mkdir -p "$G"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$G/ana.log"; }

say "=== GECE KOSUCUSU BASLADI ==="

# --- 1) cikarimin bitmesini bekle
say "1) cikarim bekleniyor..."
ONCE=-1
while true; do
  N=$(ls results/_p6_oz_tam4/tam_*.npz 2>/dev/null | wc -l)
  CANLI=$(ps -W 2>/dev/null | grep -c "python" || echo 0)
  if [ "$N" -ge 2560 ]; then say "   cikarim BITTI: $N dosya"; break; fi
  if [ "$N" = "$ONCE" ]; then
    DURGUN=$((DURGUN+1))
  else
    DURGUN=0; ONCE=$N
    say "   cikarim $N/2583"
  fi
  # 40 dakika hic ilerlemezse cikarim olmustur; elimizdekiyle devam et
  if [ "${DURGUN:-0}" -ge 8 ]; then
    say "   cikarim DURDU ($N dosya) -- elimizdekiyle devam"; break
  fi
  sleep 300
done

N=$(ls results/_p6_oz_tam4/tam_*.npz 2>/dev/null | wc -l)
if [ "$N" -lt 1500 ]; then
  say "!! cikarim cok eksik ($N) -- egitim ATLANDI"; exit 1
fi

# --- 2) URETIM modelini yeniden egit (tam korpus + neg=12)
say "2) uretim modeli yeniden egitiliyor (tam+d6 korpus, neg=12)..."
cp -f results/p6_kademe2_model.pkl "results/p6_kademe2_model.pkl.gece_oncesi" 2>/dev/null || true
P6_KUME=tam,d6 P6_KAT_MIN=60 $PY kos_p6_kademe2.py > "$G/uretim_egitim.log" 2>&1
if grep -q "makbuz ->" "$G/uretim_egitim.log"; then
  say "   egitim BITTI"
  grep -A6 "kol             robot" "$G/uretim_egitim.log" | tail -6 | tee -a "$G/ana.log"
else
  say "!! egitim TAMAMLANMADI -- son satirlar:"
  tail -5 "$G/uretim_egitim.log" | tee -a "$G/ana.log"
fi

say "=== GECE KOSUCUSU BITTI ==="
