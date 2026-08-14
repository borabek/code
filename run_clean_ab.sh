#!/usr/bin/env bash
# #11 TEMIZ A/B: 95 WEI etiketinin ETKISINI confound'suz olc. Ayni seed, same recete, TEK fark = _recall_v2.
# Her ikisi de best-corpus-val + last-epoch kaydeder (#8 fix). Sonra 4 checkpoint WEI arbiter'da karsilastirilir.
# Kullanim: bash run_clean_ab.sh   (#10 integrity GECTIKTEN after)
set -e
cd /c/Users/DE00024082/Desktop/code
PY=".venv/Scripts/python.exe"; export PYTHONPATH=_diffusion_net_repo/src PYTHONWARNINGS=ignore
RECALL="_label_targets _label_targets_2 _label_targets_3 _label_targets_recall _label_targets_recall_hard"

echo "############ A (KONTROL): 95 yeni etiket YOK (recall_hard-esdegeri, seed 0) ############"
$PY train_seg_extra.py --no-extra --partial-dir $RECALL --partial-target connection --seed 0 \
   --checkpoint-out results/seg_extra/ab_A_s0.pt 2>&1 | grep -E "DONE|prepared train|partial-human"

echo "############ B (DENEY): + 95 WEI etiket (_recall_v2, AYNI seed 0) ############"
$PY train_seg_extra.py --no-extra --partial-dir $RECALL _label_targets_recall_v2 --partial-target connection --seed 0 \
   --checkpoint-out results/seg_extra/ab_B_s0.pt 2>&1 | grep -E "DONE|prepared train|partial-human"

echo "############ EVAL: 4 checkpoint WEI arbiter (best + last, A + B) ############"
HELD=$(cat _hw_r3.txt)
for CK in ab_A_s0 ab_A_s0_last ab_B_s0 ab_B_s0_last; do
  echo "--- $CK ---"
  BA_ALLOW_SEEN=1 $PY big_arbiter.py --ckpts results/seg_extra/$CK.pt --only-mfg WEI --only-parts $HELD \
     --axis-aware --cluster-mm 5 --min-v 30 --vertex-conf 0.5 --tag ab_$CK 2>&1 | grep -E "WEI:|ARBITER" | tail -2
done
echo "############ DECISION ############"
echo "B (best VEYA last) > A > baseline 0.563 ise -> 95 etiket CALISTI (confound'suz evidence)"
echo "B <= A ise -> etiket this recete/secimle yardim etmiyor (more very etiket/aux-head)"
