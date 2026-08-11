#!/usr/bin/env bash
# BOYAMA-SONRASI OTOMASYON: indirilen paint JSON -> apply_recall -> retrain 3 seed -> held-out arbiter eval.
# Kullanim:  bash run_after_paint.sh <paint.json> <WEI|PXC>
# Ornek:     bash run_after_paint.sh ~/Downloads/recall_paint.json WEI
set -e
PAINT="$1"; MFG="${2:-WEI}"
PY=".venv/Scripts/python.exe"; export PYTHONPATH=_diffusion_net_repo/src
[ -f "$PAINT" ] || { echo "HATA: paint JSON yok: $PAINT"; exit 1; }

if [ "$MFG" = "WEI" ]; then OUTDIR=_label_targets_recall; else OUTDIR=_label_targets_recall_pxc; fi

echo "=== 1) apply_recall: boyama -> $OUTDIR (+ leakage guard) ==="
$PY apply_recall.py --paint "$PAINT" --out "$OUTDIR" --mfg "$MFG"

# retrain dizinleri: mevcut human + tum recall dirleri (yeni boyama dahil)
RECALL="_label_targets _label_targets_2 _label_targets_3 _label_targets_recall _label_targets_recall_hard"
[ -d _label_targets_recall_pxc ] && RECALL="$RECALL _label_targets_recall_pxc"

echo "=== 2) retrain 3 seed (connection-channel masked loss) ==="
for s in 0 1 2; do
  echo "  -- seed $s --"
  $PY train_seg_extra.py --no-extra --partial-dir $RECALL --partial-target connection --seed $s \
     --checkpoint-out results/seg_extra/recall_v2_s$s.pt 2>&1 | grep -E "DONE best|prepared train"
done

echo "=== 3) held-out arbiter eval (WEI + PXC) ==="
$PY big_arbiter.py --ckpts results/seg_extra/recall_v2_s0.pt results/seg_extra/recall_v2_s1.pt results/seg_extra/recall_v2_s2.pt \
   2>&1 | grep -iE "WEI|PXC|ALL|F1" | grep -v "cache\|construct"

echo "=== BITTI. recall_v2_s{0,1,2}.pt egitildi. Yeni WEI/ALL F1 yukarida. ==="
echo "Kalkarsa: cp_config robot_vote2_checkpoints'i recall_v2 ile guncelle."
