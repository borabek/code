#!/usr/bin/env bash
# MAKINE KAPANIP ACILDIKTAN SONRA HER SEYI KALDIGI YERDEN BASLAT.
#
# Kullanim:  bash kos_devam_et.sh
#
# Hepsi GUVENLE tekrar baslatilabilir:
#  * cikarim  -> var olan npz'ler ATLANIR (pid anahtarli)
#  * orkestrator -> A1b dosya sayisini gorur ve atlar, faz zincirine devam eder
#  * EK kuyrugu -> makbuzu olan blok yeniden olculur (zarar yok, sadece sure)
#  * hepsinin kendi kilidi var; iki ornek ayni anda kosmaz
set -u
cd "$(dirname "$0")"
mkdir -p results/_gece
echo "=== DEVAM: $(date +%H:%M:%S) ==="

bas() {  # bas <ad> <komut...>
  local ad="$1"; shift
  if pgrep -f "$ad" > /dev/null 2>&1; then
    echo "  $ad ZATEN KOSUYOR"
  else
    nohup "$@" > /dev/null 2>&1 &
    disown
    echo "  $ad baslatildi"
  fi
}

# 1) tavan-24 korpusunun `tam` bolumu (d6 TAMAM: 468/468)
bas kos_tam4_devam.sh bash kos_tam4_devam.sh

# 2) gece orkestratoru (B fazi / EK bloklari)
bas kos_gece.sh bash kos_gece.sh

# 3) EK blok kuyrugu (kanonik -> kume -> topoloji -> saha_kapisi -> tam3 taban)
bas kos_ek_kuyruk.sh bash kos_ek_kuyruk.sh

# 4) makbuz koruma gozcusu
bas kos_makbuz_koru.sh bash kos_makbuz_koru.sh

# 5) S7 cokus teshisi (makbuzu yoksa)
if [ ! -f results/cokus_teshisi_d6.json ]; then
  bas sonda_cokus.py env P6_DIZIN=results/_p6_oz_tam4 CK_KUME=d6 \
      CK_KAT_MIN=40 python sonda_cokus.py
fi

sleep 5
echo
echo "DURUM:"
echo "  tam4: d6=$(ls results/_p6_oz_tam4 2>/dev/null | grep -c '^d6_')/468"\
     "tam=$(ls results/_p6_oz_tam4 2>/dev/null | grep -c '^tam_')/2583"
echo "  rapor: python sabah_ozeti.py"
