#!/usr/bin/env bash
# TAVAN-24 KORPUSUNUN `tam` BOLUMUNU SURDUR (d6 ayri iscilerle tamamlaniyor).
#
# ONCEKI KOSUDA IKI SORUN CIKTI, ikisi de burada kapatildi:
#  1) BELLEK: 04:26'da B fazi ile cakisip bes paydan UCU MemoryError ile oldu.
#     Artik her pay baslamadan once bos RAM kapisi var ve pay sayisi 5 -> 4.
#  2) ERKEN OLCUM: isciler durunca `wait` donuyor ve betik EKSIK korpusta
#     KAPI A olcup makbuz yaziyordu (355/468 parcayla 0.8952 gibi yaniltici
#     bir sayi). Artik olcum yalnizca korpus TAM ise yapilir.
#
# Var olan dosyalar atlandigi icin bu betik kaldigi yerden devam eder.
set -u
cd "$(dirname "$0")"
G=results/_gece
mkdir -p "$G"
KILIT="$G/.kilit_tam4d"
if [ -e "$KILIT" ] && kill -0 "$(cat "$KILIT" 2>/dev/null)" 2>/dev/null; then
  echo "ZATEN KOSUYOR"; exit 0
fi
echo $$ > "$KILIT"
trap 'rm -f "$KILIT"' EXIT

ANA="$G/TAM4.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$ANA"; }

bos_ram() {
  powershell.exe -NoProfile -Command \
    "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)" \
    2>/dev/null | tr -d '\r'
}

# Agir is (kademe2) bitene ve RAM acilana kadar bekle
bek=0
while [ $bek -lt $((6 * 3600)) ]; do
  r=$(bos_ram); r=${r:-0}
  if [ "${r%%.*}" -ge 8 ]; then break; fi
  say "tam bolumu bekliyor: ${r}GB bos (gereken 8GB)"
  sleep 300; bek=$((bek + 300))
done

say "=== TAVAN-24 `tam` BOLUMU SURUYOR (4 pay) ==="
for i in 0 1 2 3; do
  ( BH_MESH_ESIK=0.05 YB_FAN=256 P6_KAYNAK=012 YB_MAX_SEC=24 \
    P6_MESH_R=2.5 P6_MESH_MAX=350 P6_MESH_KAT=5 \
    P6_CIK=results/_p6_oz_tam4 P6_SHARD="$i/4" \
    python kos_p6_oznitelik.py tam > "$G/tam4d_tam_$i.log" 2>&1 ) &
done
wait

n_d6=$(ls results/_p6_oz_tam4 2>/dev/null | grep -c '^d6_')
n_tam=$(ls results/_p6_oz_tam4 2>/dev/null | grep -c '^tam_')
say "cikarim durdu -- d6 $n_d6/468, tam $n_tam/2583"

# KAPI A: YALNIZCA korpus TAM ise. Eksik korpusta olculen sayi yaniltici olur
# (biten parcalar rastgele degil, once biten yani kolay parcalardir).
for k in d6 tam; do
  if [ "$k" = "d6" ]; then n=$n_d6; hedef=460; else n=$n_tam; hedef=2570; fi
  if [ "$n" -ge "$hedef" ]; then
    say "KAPI A olcumu ($k, $n parca)"
    HT_ONLER=$k P6_DIZIN=results/_p6_oz_tam4 python sonda_havuz_tavani.py \
      > "$G/TAM4_kapiA_$k.log" 2>&1
    tail -4 "$G/TAM4_kapiA_$k.log" | sed 's/^/    /' | tee -a "$ANA"
  else
    say "KAPI A ATLANDI ($k): korpus EKSIK ($n < $hedef) -- eksik korpusta"
    say "  olculen sayi yaniltici olurdu. Esli kiyas icin: sonda_esli_tavan.py"
  fi
done
say "=== TAM4 DEVAM BITTI ==="
