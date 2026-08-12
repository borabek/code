#!/usr/bin/env bash
# EK BLOK KUYRUGU -- orkestratorun DISINDA, bellek korumali.
#
# NEDEN AYRI: `kos_gece.sh` kosarken duzenlenemez (bash betigi bayt konumundan
# okur; ortasindan degistirmek yurutmeyi bozar). Gece basladiktan SONRA yazilan
# bloklar bu yuzden oraya eklenmedi, yanina zincirlendi.
#
# IKI KAPI (ikisi de saglanmadan hicbir blok baslamaz):
#   1) korpus TAM        -- eksik korpusta olculen sayi gecersizdir
#   2) bos RAM >= 10 GB  -- her EK kosusu ~6 GB yukluyor. Bu gece UC surec
#                           bellek tukendigi icin SESSIZCE oldu (cikis kodu 0,
#                           hicbir hata satiri yok) -- teshis edilmesi zor.
#
# SIRA: once KANONIK (en ucuz + en guclu tek degiskenli ayrim), sonra TOPOLOJI.
set -u
cd "$(dirname "$0")"
G=results/_gece
mkdir -p "$G"
BLOKLAR=${EK_KUYRUK:-"kanonik topoloji"}
ANA="$G/EK_KUYRUK.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$ANA"; }

GEREK_DOSYA=3040
GEREK_RAM=10

bos_ram() {
  # DIKKAT: FreePhysicalMemory KB cinsindendir; GB icin dogru bolen /1MB.
  powershell.exe -NoProfile -Command \
    "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)" \
    2>/dev/null | tr -d '\r'
}

# UCUNCU KAPI: BASKA AGIR IS KOSUYORSA BASLAMA.
# 04:26'da bos RAM 0.8 GB'a dustu: B fazi (kademe2) ve EK blogu AYNI ANDA
# korpusun tamamini belleğe aliyor (~6'sar GB) ve ustune 5 cikarim iscisi.
# RAM kapisi yalnizca BASLANGICTA bakiyordu; agir isin kendisini sormak gerek.
agir_is() {
  powershell.exe -NoProfile -Command \
    "(Get-CimInstance Win32_Process | Where-Object { \$_.Name -match 'python' -and \$_.CommandLine -match 'kademe2|kos_ek_oznitelik' } | Measure-Object).Count" \
    2>/dev/null | tr -d '\r'
}

kapilari_bekle() {
  local bek=0
  while [ $bek -lt $((10 * 3600)) ]; do
    local n r a
    n=$(ls results/_p6_oz_tam3 2>/dev/null | wc -l)
    r=$(bos_ram); r=${r:-0}
    a=$(agir_is); a=${a:-0}
    if [ "$n" -ge $GEREK_DOSYA ] && [ "${r%%.*}" -ge $GEREK_RAM ] \
       && [ "$a" -eq 0 ]; then
      say "KAPILAR ACIK: $n dosya, ${r}GB bos, agir is yok"
      return 0
    fi
    say "bekliyor: $n/$GEREK_DOSYA dosya, ${r}GB bos (gereken ${GEREK_RAM}GB), agir is $a"
    sleep 600
    bek=$((bek + 600))
  done
  say "ZAMAN ASIMI"
  return 1
}

say "=== EK KUYRUGU BASLADI: $BLOKLAR ==="
for b in $BLOKLAR; do
  kapilari_bekle || exit 1
  say "BASLIYOR: $b"
  t0=$(date +%s)
  if EK_BLOK="$b" P6_DIZIN=results/_p6_oz_tam3 P6_KUME=tam,d6 \
       P6_KAT_MIN=200 P6_ARAMA_N=200 P6_ITER=200 P6_NEG_KAT=6 \
       python kos_ek_oznitelik.py > "$G/EK_$b.log" 2>&1; then
    say "BITTI: $b ($(( $(date +%s) - t0 ))s) -- $(tail -3 "$G/EK_$b.log" | head -2 | tr '\n' ' ')"
  else
    say "DUSTU: $b ($(( $(date +%s) - t0 ))s)"
    tail -4 "$G/EK_$b.log" | sed 's/^/    /' | tee -a "$ANA"
  fi
done
# SAHA GUVEN KAPISI -- "robot hangi isaretlere kendi basina guvenebilir?"
# GLB'deki kirmizi/turuncu ayrimi bugun keyfi bir esikle (robot_conf_auto=0.5,
# 3 oy) yapiliyor. Bu olcum esigi OLCULMUS kesinlige baglar ve bunu GORULMEMIS
# marka katlarinda yapar -- yani D7 sinavini HARCAMADAN.
if kapilari_bekle; then
  say "BASLIYOR: saha_kapisi"
  t0=$(date +%s)
  if P6_DIZIN=results/_p6_oz_tam3 P6_KUME=tam,d6 P6_KAT_MIN=200 \
       P6_ITER=200 P6_NEG_KAT=6 \
       python kos_saha_kapisi.py > "$G/SAHA_KAPISI.log" 2>&1; then
    say "BITTI: saha_kapisi ($(( $(date +%s) - t0 ))s)"
    tail -8 "$G/SAHA_KAPISI.log" | sed 's/^/    /' | tee -a "$ANA"
  else
    say "DUSTU: saha_kapisi ($(( $(date +%s) - t0 ))s)"
    tail -4 "$G/SAHA_KAPISI.log" | sed 's/^/    /' | tee -a "$ANA"
  fi
fi

say "=== EK KUYRUGU BITTI ==="
