#!/bin/bash
# Yeni STEP partisini uctan uca isler: DENETLE -> ETIKETLE -> BIRLESTIR -> v31'i BASLAT
#
# Kullanim:  bash ingest_new_batch.sh
# On kosul:  yeni .stp dosyalari all_wscad_stp/ icine atilmis olmali.
#
# Bu script yeni gelenleri MEVCUT corpus'defn (wscad_corpus_v5) ayirt eder ve yalnizca
# onlari etiketler -- 1741 parcayi yeniden etiketlemez.
set -u
cd "c:/Users/DE00024082/Desktop/code"

echo "=== 1) YENI DOSYALARI TESPIT ET ==="
ls all_wscad_stp/*.stp | sort > _pool_now.txt
# corpus'ta already etiketli olanlarin adlari
ls wscad_corpus_v5/*.json 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/\.json$//' | sort > _labelled.txt
# havuzdaki each STEP for part_nr (file adi) cikar, etiketli olmayanlari sec
: > _new_steps.txt
while read -r f; do
  pn=$(basename "$f" .stp)
  grep -qxF "$pn" _labelled.txt || echo "$f" >> _new_steps.txt
done < _pool_now.txt
n_new=$(grep -c . _new_steps.txt || echo 0)
echo "yeni/etiketsiz STEP: $n_new"
if [ "$n_new" -lt 50 ]; then
  echo "HATA: yeni file none/az ($n_new) -- dosyalari all_wscad_stp/ icine attin mi?"
  exit 1
fi

echo "=== 2) KAPSAM DENETIMI (dagitim bloklari / yasakli list) ==="
.venv/Scripts/python.exe audit_batch.py --files _new_steps.txt --guard pxc_out_of_scope.txt \
  --delete-out-of-scope 2>&1 | tee _new_batch_audit.log
# audit silmis olabilir -> listeyi tazele
awk '{print}' _new_steps.txt > _tmp && while read -r f; do [ -f "$f" ] && echo "$f"; done < _tmp > _new_steps.txt && rm -f _tmp
n_new=$(grep -c . _new_steps.txt || echo 0)
echo "audit sonrasi etiketlenecek: $n_new"

echo "=== 3) ETIKETLE (6 shard, kanitlanmis tarif) ==="
rm -rf _nshard_dir_* wscad_corpus_new_p* _nshard_0*
split -n l/6 -d _new_steps.txt _nshard_
for i in 0 1 2 3 4 5; do
  mkdir -p _nshard_dir_$i
  while read -r f; do ln -f "$f" "_nshard_dir_$i/$(basename "$f")" 2>/dev/null || cp "$f" "_nshard_dir_$i/"; done < _nshard_0$i
  ( .venv/Scripts/python.exe step_openings.py _nshard_dir_$i --label-corpus wscad_corpus_new_p$i \
      --auto --deflection 0.5 --check-dirs --merge-tol 4.0 > _new_shard_$i.log 2>&1 & )
done
while true; do
  d=0; for i in 0 1 2 3 4 5; do grep -qa "wrote .* labelled part" _new_shard_$i.log 2>/dev/null && d=$((d+1)); done
  [ "$d" -eq 6 ] && break; sleep 60
done
for i in 0 1 2 3 4 5; do
  powershell -NoProfile -Command "compact /c '/s:c:\Users\DE00024082\Desktop\code\wscad_corpus_new_p$i' /i /q" > /dev/null 2>&1
done

echo "=== 4) CORPUS v6 = v5 + yeni (global dedup) ==="
mkdir -p wscad_corpus_v6
powershell -NoProfile -Command "compact /c '/s:c:\Users\DE00024082\Desktop\code\wscad_corpus_v6' /i /q" > /dev/null 2>&1
.venv/Scripts/python.exe merge_shards.py --shards wscad_corpus_v5 wscad_corpus_new_p0 \
  wscad_corpus_new_p1 wscad_corpus_new_p2 wscad_corpus_new_p3 wscad_corpus_new_p4 \
  wscad_corpus_new_p5 --out wscad_corpus_v6 --move 2>&1 | tee _v6_merge.log
n6=$(ls wscad_corpus_v6 | wc -l)
echo "corpus v6: $n6 part"
if [ "$n6" -lt 2500 ]; then echo "HATA: corpus v6 missing ($n6) -- v31 BASLATILMADI"; exit 1; fi
rm -rf wscad_corpus_new_p* _nshard_dir_* _nshard_0* wscad_corpus_v5

echo "=== 5) ETIKET TAVANI ==="
.venv/Scripts/python.exe diag_encoding.py wscad_corpus_v6 2>&1 | grep -E "parts with|min_votes=1" | tee _v6_ceiling.log

echo "=== 6) v31 BASLIYOR ==="
sed -e 's/^run_name: .*/run_name: cp_hp_v31_ftc6/' -e 's/wscad_corpus_v5/wscad_corpus_v6/' \
    run_hp_v30_ftc5.yaml > run_hp_v31_ftc6.yaml
CP_PREP_WORKERS=6 .venv/Scripts/python.exe train_cp.py "C:\Users\DE00024082\Desktop\JSON" \
  --config run_hp_v31_ftc6.yaml --prep-cache-dir prep_cache > train_run_hp_v31_ftc6.log 2>&1
echo "v31 bitti (cikis $?)"
