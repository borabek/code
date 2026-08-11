#!/usr/bin/env bash
# GERCEK MODEL: 2583 parcalik `tam` korpusu, marka katlarinda kural secimi.
#
# Katlar (P6_KAT_MIN=200): TOGI 810 / PXC 713 / WEI 686 / SIE 262 = 2471 parca.
# Kollar: TABAN (dagitilan kural) | P6 | P6_KAFES (kaskad) | P6_GEO (segmentasyonsuz)
# Kural secim olcutu MAKRO -- tek markada cokmeyen kurali tercih eder.
# NMS her D6 kivriminda 5.0 secildi; tam kosuda sabitlenir (tarama maliyeti 3x).
# KAHIN kapali (teshis amacliydi, maliyeti ikiye katliyor).
#
# D6 ve D7'ye BAKILMAZ.
set -u
cd "$(dirname "$0")"
export P6_KUME=tam
export P6_DIZIN=results/_p6_oz_u25
export P6_KAT_MIN=${P6_KAT_MIN:-200}
export P6_OLCUT=${P6_OLCUT:-makro}
export P6_KOLLAR=${P6_KOLLAR:-TABAN,P6,P6_KAFES,P6_GEO}
export P6_NMSLER=${P6_NMSLER:-5.0}
export P6_KAHIN=0
export P6_ARAMA_N=${P6_ARAMA_N:-300}
export P6_NEG_KAT=${P6_NEG_KAT:-6}
export P6_ITER=${P6_ITER:-250}
n=$(ls results/_p6_oz_u25/tam_*.npz 2>/dev/null | wc -l)
echo "egitim korpusu: $n / 2583 parca hazir"
echo "katlar >= $P6_KAT_MIN | olcut $P6_OLCUT | kollar $P6_KOLLAR"
python kos_p6_kademe2.py
