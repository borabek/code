#!/usr/bin/env bash
# MAKINE KAPANIP ACILDIKTAN SONRA HER SEYI KALDIGI YERDEN BASLAT.
#
# Kullanim:  bash run_resume_do.sh
#
# Hepsi GUVENLE tekrar baslatilabilir:
#  * inference  -> present which npz'ler ATLANIR (pid anahtarli)
#  * orkestrator -> A1b file sayisini gorur ve atlar, faz zincirine devam eder
#  * EK kuyrugu -> makbuzu which blok yeniden olculur (zarar none, sadece sure)
#  * hepsinin kendi kilidi present; iki ornek same anda kosmaz
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

# 1) ceiling-24 korpusunun `tam` bolumu (d6 TAMAM: 468/468)
bas run_tam4_resume.sh bash run_tam4_resume.sh

# 2) night orkestratoru (B fazi / EK bloklari)
bas run_night.sh bash run_night.sh

# 3) EK blok kuyrugu (kanonik -> cluster -> topoloji -> saha_kapisi -> tam3 baseline)
bas run_extra_queue.sh bash run_extra_queue.sh

# 4) receipt koruma gozcusu
bas run_receipt_guard.sh bash run_receipt_guard.sh

# 5) S7 cokus teshisi (makbuzu otherwise)
if [ ! -f results/cokus_teshisi_d6.json ]; then
  bas probe_cokus.py env P6_DIZIN=results/_p6_oz_tam4 CK_KUME=d6 \
      CK_KAT_MIN=40 python probe_cokus.py
fi

sleep 5
echo
echo "DURUM:"
echo "  tam4: d6=$(ls results/_p6_oz_tam4 2>/dev/null | grep -c '^d6_')/468"\
     "tam=$(ls results/_p6_oz_tam4 2>/dev/null | grep -c '^tam_')/2583"
echo "  rapor: python sabah_ozeti.py"
