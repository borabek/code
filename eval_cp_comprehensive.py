# -*- coding: utf-8 -*-
"""Comprehensive CP-F1: derive CPs with the CableEntry-else-Contact rule for BOTH the human
labels and the model prediction (so the 18 CableEntry-absent parts now count), match at 5mm.
GT uses min_v=5 (light), model uses --min-v (cleanup, default 10). Dev preview on val; a
held-out estimate on test_locked with the val-selected min_v (no test tuning).

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe eval_cp_comprehensive.py val 10
"""
import sys
import numpy as np
import scheffler_dataset as dataset
import diffusionnet, metrics
from viz_cp_comprehensive import comprehensive_cps

CORPUS = "wscad_corpus_scheffler_exact"; CKPT = "results/scheffler_semantic/refit91.pt"
OPCACHE = "results/scheffler_semantic/operators"; MATCH = 5.0
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "val"
MIN_V = int(sys.argv[2]) if len(sys.argv) > 2 else 10

model, meta, _ = diffusionnet.load_checkpoint(CKPT, device="cuda")
samples = dataset.load_split(CORPUS, SPLIT, allow_locked=(SPLIT == "test_locked"), verify_hashes=False)
print(f"{SPLIT}: {len(samples)} parts  (comprehensive: CableEntry-else-Contact)\n")

TP = FP = FN = 0; per = []
for s in samples:
    V, F = np.asarray(s["verts"], float), np.asarray(s["faces"], int)
    gt = np.array([p for p, _ in comprehensive_cps(V, F, np.asarray(s["labels"]), 5)[0]] or []).reshape(-1, 3)
    plab = np.asarray(diffusionnet.predict(model, meta, V, F, device="cuda", op_cache_dir=OPCACHE))
    pr = np.array([p for p, _ in comprehensive_cps(V, F, plab, MIN_V)[0]] or []).reshape(-1, 3)
    if len(gt) and len(pr):
        m, up, ug = metrics.match_predictions(pr, gt, MATCH)
        tp, fp, fn = len(m), len(up), len(ug)
    else:
        tp, fp, fn = 0, len(pr), len(gt)
    TP += tp; FP += fp; FN += fn
    per.append((s["part_id"], len(gt), len(pr), tp, fp, fn))

f1 = 2 * TP / max(2 * TP + FP + FN, 1); jac = TP / max(TP + FP + FN, 1)
prec = TP / max(TP + FP, 1); rec = TP / max(TP + FN, 1)
print(f"{'part':<11}{'GT':>3}{'pred':>5}{'TP':>4}{'FP':>4}{'FN':>4}")
for pid, ng, npd, tp, fp, fn in per:
    flag = "  <-- FN/FP" if (fn or fp) else ""
    print(f"  {pid:<9}{ng:>3}{npd:>5}{tp:>4}{fp:>4}{fn:>4}{flag}")
print(f"\n>>> COMPREHENSIVE CP-F1 ({SPLIT}, min_v={MIN_V}, {MATCH}mm):")
print(f"    F1={f1:.3f}  Jaccard={jac:.3f}  precision={prec:.3f}  recall={rec:.3f}  TP={TP} FP={FP} FN={FN}")
print("DONE")
