#!/usr/bin/env bash
# D7 SINAV OKUMASI -- butce 3, bu bir okumadir.
#
# Iki kol AYNI kosuda olculur:
#   TABAN  URUN_P6=0  -> dagitilan urun (genis havuz + v6 gate + poz kafasi)
#   P6     URUN_P6=1  -> yon bankasi + mesh havuzu + iki kademeli siralayici
# Ikisi de URUNUN TEK kanonik zincirinden (`kanonik_zincir.urun_cikti`) gecer.
#
# Paylara bolunur cunku P6 kolu parca basina saniyeler suruyor; birlestirme
# mikro F1 icin KAYIPSIZDIR (parca basina TP/FP/FN toplami).
#
# ONCE `python sonda_p6_hazir.py` GECMELIDIR.
set -u
cd "$(dirname "$0")"
N=${N:-6}
rm -f results/d7_p6_*.json results/d7_taban_*.json

echo "=== KOL 1/2: TABAN (URUN_P6=0) ==="
for i in $(seq 0 $((N-1))); do
  ( URUN_P6=0 URUN_GENIS=1 DOG_SHARD="$i/$N" \
    DOG_CIKTI="results/d7_taban_$i.json" \
    python sonda_dagitim_dogrula.py > "results/_d7t_$i.log" 2>&1 ) &
done
wait
python birlestir_makbuz.py results/d7_taban.json results/d7_taban_*.json

echo
echo "=== KOL 2/2: P6 (URUN_P6=1) ==="
for i in $(seq 0 $((N-1))); do
  ( URUN_P6=1 DOG_SHARD="$i/$N" \
    DOG_CIKTI="results/d7_p6_$i.json" \
    python sonda_dagitim_dogrula.py > "results/_d7p_$i.log" 2>&1 ) &
done
wait
python birlestir_makbuz.py results/d7_p6.json results/d7_p6_*.json

echo
echo "=== MANSET ==="
python manset_050.py results/d7_p6.json results/d7_taban.json
