#!/usr/bin/env bash
# D7 SINAV OKUMASI -- butce 3, bu bir okumadir.
#
# Iki arm AYNI kosuda olculur:
#   TABAN  URUN_P6=0  -> dagitilan urun (genis pool + v6 gate + poz kafasi)
#   P6     URUN_P6=1  -> direction bankasi + mesh havuzu + iki kademeli siralayici
# Ikisi de URUNUN TEK kanonik zincirinden (`canonical_chain.product_output`) gecer.
#
# Paylara bolunur cunku P6 kolu part basina saniyeler suruyor; birlestirme
# mikro F1 icin KAYIPSIZDIR (part basina TP/FP/FN toplami).
#
# ONCE `python probe_p6_hazir.py` GECMELIDIR.
set -u
cd "$(dirname "$0")"
N=${N:-8}
rm -f results/d7_p6_*.json results/d7_taban_*.json

# TABAN kolu HER ZAMAN poz kafasi ACIK kosar -- dagitilan urunun hali budur.
# `DOG_POZ` yalniz P6 kolu icindir; tabana uygulamak onu kendi dagitilan
# yapilandirmasindan zayiflatir ve kiyasi haksiz kilar.
echo "=== KOL 1/2: TABAN (URUN_P6=0, poz kafasi ACIK) ==="
for i in $(seq 0 $((N-1))); do
  ( URUN_P6=0 URUN_GENIS=1 DOG_POZ=1 DOG_SHARD="$i/$N" \
    DOG_CIKTI="results/d7_taban_$i.json" \
    python probe_dagitim_dogrula.py > "results/_d7t_$i.log" 2>&1 ) &
done
wait
python birlestir_makbuz.py results/d7_taban.json results/d7_taban_*.json

# P6 kolu ILAN EDILEN yapilandirmayla kosar: poz kafasi KAPALI.
# Gerekce: model ve karar kurali brand katlarinda POZ KAFASI OLMADAN secildi;
# uzerine dogrulanmamis bir son islem koymak, olculen seyden baska bir sey
# dagitmak olurdu.
echo
echo "=== KOL 2/3: P6 (URUN_P6=1, poz kafasi KAPALI -- ILAN EDILEN) ==="
for i in $(seq 0 $((N-1))); do
  ( URUN_P6=1 DOG_POZ=0 DOG_SHARD="$i/$N" \
    DOG_CIKTI="results/d7_p6_$i.json" \
    python probe_dagitim_dogrula.py > "results/_d7p_$i.log" 2>&1 ) &
done
wait
python birlestir_makbuz.py results/d7_p6.json results/d7_p6_*.json

# UCUNCU KOL yalnizca GOZLEMDIR: poz kafasi P6'nin sectigi yonu eziyor mu?
# Manset bu koldan SECILMEZ -- sinava bakip yapilandirma secmek, sinavdan ayar
# cekmektir.
echo
echo "=== KOL 3/3: P6 + poz kafasi (GOZLEM, headline DEGIL) ==="
rm -f results/d7_p6poz_*.json
for i in $(seq 0 $((N-1))); do
  ( URUN_P6=1 DOG_POZ=1 DOG_SHARD="$i/$N" \
    DOG_CIKTI="results/d7_p6poz_$i.json" \
    python probe_dagitim_dogrula.py > "results/_d7pp_$i.log" 2>&1 ) &
done
wait
python birlestir_makbuz.py results/d7_p6poz.json results/d7_p6poz_*.json

echo
echo "=== MANSET (ILAN EDILEN arm) ==="
python manset_050.py results/d7_p6.json results/d7_taban.json

echo
echo "=== GUVEN KAPISI (kalibrasyon D7'de DEGIL: egri raporlanir) ==="
python run_confidence_gate.py results/d7_p6.json || true
