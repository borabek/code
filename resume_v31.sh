#!/bin/bash
# v31'i kaldigi yerden devam ettirir (PC kapanmasi sonrasi tek komut).
#
#   bash resume_v31.sh
#
# Neden bu 3 env degiskeni: corpus v8 (3033 part) 12.556 training grafigine
# yamalaniyor ve varsayilan yol 31GB'a SIGMIYOR -- commit %99'a yapisir, Windows
# pagefile'i buyutur, disk biter. Ikisi de matematiksel olarak notr (ayni tohumla
# kayiplar birebir ayni olcüldü):
#   CP_LAZY_HIER=1        hiyerarsileri RAM yerine prep_cache'ten batch basina okur
#                         (--prep-cache-dir SART; onbelleksiz her batch sifirdan kurar)
#   CP_FREE_TRAIN_SOURCE  modelin bir daha okumadigi ham verts/faces/target'i birakir
#                         (augment 0 oldugu icin kodun kendi bosaltmasi hic calismiyor)
#   CP_PREP_WORKERS=6     KANITLANMIS deger. 12 yapma: ucustaki bellek ikiye katlanir
#                         ve prep commit'i patlatir.
#
# --resume: checkpoints/cp_hp_v31_ftc6_last.ckpt'ten devam eder ve DONMUS split
# manifest'ini geri oynatir (corpus'a part eklense bile egitilmis bir part val'e
# kayamaz).
#
# Beklenen: hazirlik ~40dk (hiyerarsiler onbellekli; kNN + oznitelik yeniden kurulur,
# onlar bilerek diske yazilmiyor -- 6.4GB tutardi), sonra ~14.5dk/epoch.
set -u
cd "c:/Users/DE00024082/Desktop/code"

if [ ! -d prep_cache ]; then
  echo "UYARI: prep_cache YOK -- hazirlik 40dk yerine ~55dk surer (hiyerarsiler yeniden kurulur)"
fi
if [ ! -f checkpoints/cp_hp_v31_ftc6_last.ckpt ]; then
  echo "checkpoint yok -> BASTAN baslatiliyor (--resume atlaniyor)"
  RESUME=""
else
  echo "checkpoint bulundu -> epoch $(grep -ao '"epoch": [0-9]*' checkpoints/cp_hp_v31_ftc6_history.json 2>/dev/null | tail -1 | grep -o '[0-9]*' || echo 5)'ten devam"
  RESUME="--resume"
fi

free_gb=$(df -BG /c | tail -1 | awk '{print $4}' | tr -d 'G')
if [ "${free_gb:-0}" -lt 4 ]; then
  echo "HATA: disk sadece ${free_gb}GB -- training pagefile'i buyutup diski doldurur. Once yer ac."
  exit 1
fi

echo "=== v31 devam ediyor ($(date '+%H:%M')) -- log: train_run_hp_v31_ftc6.log ==="
CP_LAZY_HIER=1 CP_FREE_TRAIN_SOURCE=1 CP_PREP_WORKERS=6 \
  .venv/Scripts/python.exe train_cp.py "C:\Users\DE00024082\Desktop\JSON" \
  --config run_hp_v31_ftc6.yaml --prep-cache-dir prep_cache $RESUME \
  >> train_run_hp_v31_ftc6.log 2>&1
echo "bitti (cikis $?)"
