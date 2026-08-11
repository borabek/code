#!/usr/bin/env bash
# GECE ORKESTRATORU -- docs/GECE_PLANI_080.md'yi bastan sona kosar.
#
# Ilkeler:
#  * Her faz KENDI logunu yazar; bir faz duserse zincir DURMAZ, sonraki faza gecer.
#  * Dagitilan model paketi her fazdan once korunur, sonra geri yuklenir.
#  * D7'ye HIC BAKILMAZ. Butun olcumler `tam` marka katlarinda.
#  * Her fazin ciktisi results/gece_<faz>.json olarak birikir.
set -u
cd "$(dirname "$0")"
G=results/_gece
mkdir -p "$G"
ANA="$G/ANA.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$ANA"; }

KORU="$G/paket_baslangic.pkl"
cp -f results/p6_kademe2_model.pkl "$KORU" 2>/dev/null || true
paket_geri() { [ -f "$KORU" ] && cp -f "$KORU" results/p6_kademe2_model.pkl; }
trap paket_geri EXIT

faz() {                     # faz <ad> <komut...>
  local ad="$1"; shift
  log "BASLIYOR: $ad"
  local t0=$(date +%s)
  if "$@" > "$G/$ad.log" 2>&1; then
    log "BITTI: $ad ($(( $(date +%s) - t0 ))s)"
  else
    log "DUSTU: $ad ($(( $(date +%s) - t0 ))s) -- son satirlar:"
    tail -5 "$G/$ad.log" | sed 's/^/    /' | tee -a "$ANA"
  fi
  paket_geri
}

log "=========== GECE PROGRAMI BASLADI ==========="

# ---------------------------------------------------------------- FAZ A
# A1b: UC KALDIRAC ACIK korpus (yelpaze + dusuk mesh esigi + seyreltme)
# UST SINIR 600 -> 350: 600'de cikarim 6.3 dosya/dk (7.7 SAAT, gece yetmez).
# Olculen diz noktasi zaten 251 adaydaydi (konum recall 0.8713); 350 onun
# uzerinde kalir ama maliyeti yariya iner. ASIL KALDIRAC mesh ESIGI (0.05),
# aday SAYISI degil -- CWT'de konum recall'u acan oydu.
if [ "$(ls results/_p6_oz_tam3 2>/dev/null | wc -l)" -lt 3040 ]; then
  log "BASLIYOR: A1b tam-acik korpus cikarimi (6 pay)"
  for i in 0 1 2 3 4 5; do
    ( BH_MESH_ESIK=0.05 YB_FAN=256 P6_KAYNAK=012 \
      P6_MESH_R=2.5 P6_MESH_MAX=350 P6_MESH_KAT=5 \
      P6_CIK=results/_p6_oz_tam3 P6_SHARD="$i/6" \
      python kos_p6_oznitelik.py d6 > "$G/A1b_d6_$i.log" 2>&1
      BH_MESH_ESIK=0.05 YB_FAN=256 P6_KAYNAK=012 \
      P6_MESH_R=2.5 P6_MESH_MAX=350 P6_MESH_KAT=5 \
      P6_CIK=results/_p6_oz_tam3 P6_SHARD="$i/6" \
      python kos_p6_oznitelik.py tam > "$G/A1b_tam_$i.log" 2>&1 ) &
  done
  wait
  log "BITTI: A1b -- $(ls results/_p6_oz_tam3 | wc -l) dosya"
fi

# A2: yeni havuzun tavani (KAPI A)
faz A2_havuz_tavani env P6_DIZIN=results/_p6_oz_tam3 python sonda_havuz_tavani.py

# A3: yelpaze uctan uca (yelpazeli korpus hazirsa)
if [ "$(ls results/_p6_oz_fan 2>/dev/null | wc -l)" -ge 3040 ]; then
  faz A3_yelpaze_uctan_uca bash kos_yelpaze_olc.sh
fi

# ---------------------------------------------------------------- FAZ B
faz B1_zor_negatif env P6_DIZIN=results/_p6_oz_tam3 P6_ZORNEG=1 \
    P6_KUME=tam,d6 P6_KAT_MIN=200 P6_KOLLAR=P6 P6_NMSLER=5.0 \
    P6_ARAMA_N=250 P6_NEG_KAT=6 P6_ITER=200 python kos_p6_kademe2.py

faz B6_ensemble env P6_DIZIN=results/_p6_oz_tam3 P6_TOHUM_N=3 \
    P6_KUME=tam,d6 P6_KAT_MIN=200 P6_KOLLAR=P6 P6_NMSLER=5.0 \
    P6_ARAMA_N=250 P6_NEG_KAT=6 P6_ITER=200 python kos_p6_kademe2.py

# EK OZNITELIK BLOKLARI -- hepsi TEK cerceveden, tek degiskenli.
# Ucuzdan pahaliya sirali: dusen bir blok sonrakini engellemez.
for blok in ozkalib kafes_adet simetri derinlik; do
  faz "EK_$blok" env EK_BLOK="$blok" P6_DIZIN=results/_p6_oz_tam3 \
      P6_KUME=tam,d6 P6_KAT_MIN=200 P6_ARAMA_N=200 P6_ITER=200 \
      P6_NEG_KAT=6 python kos_ek_oznitelik.py
done

# ---------------------------------------------------------------- FAZ C
[ -f kos_c2_brep_topoloji.py ]    && faz C2_brep_topoloji    python kos_c2_brep_topoloji.py

# ---------------------------------------------------------------- FAZ D
[ -f kos_d1_tepe_ag.py ] && faz D1_tepe_basi_ag python kos_d1_tepe_ag.py

log "=========== GECE PROGRAMI BITTI ==========="
log "Ozet:"
grep -E "BITTI:|DUSTU:" "$ANA" | tail -30 | tee -a "$ANA"
