#!/usr/bin/env bash
# GERCEK MODEL: 2583 parcalik `tam` korpusu, brand katlarinda rule secimi.
#
# Katlar (P6_KAT_MIN=200): TOGI 810 / PXC 713 / WEI 686 / SIE 262 = 2471 part.
# Kollar: BASELINE (dagitilan rule) | P6 | P6_KAFES (kaskad) | P6_GEO (segmentasyonsuz)
# Kural secim olcutu MAKRO -- tek markada cokmeyen kurali tercih eder.
# NMS each D6 kivriminda 5.0 secildi; tam kosuda sabitlenir (tarama maliyeti 3x).
# KAHIN kapali (teshis amacliydi, maliyeti ikiye katliyor).
#
# D6 ve D7'ye BAKILMAZ.
set -u
cd "$(dirname "$0")"
# EGITIM KORPUSU: `tam` + `d6`. D6 SINAV DEGIL -- gelistirme set; egitime
# katmak D7 for mesru ve brand cesitliligini 9 -> 17 yapar. D7'ye DOKUNULMAZ.
export P6_KUME=${P6_KUME:-tam,d6}
export P6_DIZIN=results/_p6_oz_u25
export P6_KAT_MIN=${P6_KAT_MIN:-200}
export P6_OLCUT=${P6_OLCUT:-makro}
export P6_KOLLAR=${P6_KOLLAR:-BASELINE,P6,P6_KAFES,P6_GEO}
export P6_NMSLER=${P6_NMSLER:-5.0}
export P6_KAHIN=0
export P6_ARAMA_N=${P6_ARAMA_N:-250}
export P6_NEG_KAT=${P6_NEG_KAT:-6}
export P6_ITER=${P6_ITER:-200}
# NOT: ITER 400 -> 200 ve ARAMA_N 600 -> 250 SURE for. 4 arm x 4 fold = 16 model
# + 4 OOF modeli; 2583 part x ~2000 option = ~5M satir. Ogrenme egrisi
# logaritmik oldugu for 200 iterasyon 400'un very altinda not, but kosu
# suresi yariya iniyor.
n=$(ls results/_p6_oz_u25/tam_*.npz 2>/dev/null | wc -l)
echo "training korpusu: $n / 2583 part hazir"
echo "katlar >= $P6_KAT_MIN | criterion $P6_OLCUT | kollar $P6_KOLLAR"
python run_p6_kademe2.py
