# -*- coding: utf-8 -*-
"""CP-F1 with the user-approved definition (cp_openings.connection_points: Contact+CableEntry
openings, deduped per terminal). GT = human labels, pred = model prediction, both run through
the SAME derivation, matched at MATCH mm. Reports val (dev) and, with the val-selected min_v,
a held-out test_locked estimate. Single source of truth = cp_openings (viz uses the same).

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe eval_cp_openings.py val 10
"""
import sys
import numpy as np
import scheffler_dataset as dataset
import diffusionnet, metrics, cp_openings

CORPUS = "wscad_corpus_scheffler_exact"; CKPT = "results/scheffler_semantic/refit91.pt"
OPCACHE = "results/scheffler_semantic/operators"; MATCH = 5.0
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "val"
MIN_V = int(sys.argv[2]) if len(sys.argv) > 2 else 10
MIN_CONF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0   # component-mean confidence gate
VERTEX_CONF = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0  # per-vertex mask (splits merged openings)
DEDUPE = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0    # cross-class dedupe distance

model, meta, _ = diffusionnet.load_checkpoint(CKPT, device="cuda")
samples = dataset.load_split(CORPUS, SPLIT, allow_locked=(SPLIT == "test_locked"), verify_hashes=False)
print(f"{SPLIT}: {len(samples)} parts  (CP = Contact+CableEntry openings, deduped, min_conf={MIN_CONF})\n")

TP = FP = FN = 0; per = []
for s in samples:
    V, F = np.asarray(s["verts"], float), np.asarray(s["faces"], int)
    gt = np.array([c["point"] for c in cp_openings.connection_points(V, F, np.asarray(s["labels"]), min_v=MIN_V, dedupe_mm=DEDUPE)] or []).reshape(-1, 3)
    plab, probs = diffusionnet.predict(model, meta, V, F, device="cuda", op_cache_dir=OPCACHE, return_probs=True)
    pr = np.array([c["point"] for c in cp_openings.connection_points(V, F, np.asarray(plab), min_v=MIN_V, probs=probs, min_conf=MIN_CONF, vertex_conf=VERTEX_CONF, dedupe_mm=DEDUPE)] or []).reshape(-1, 3)
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
    print(f"  {pid:<9}{ng:>3}{npd:>5}{tp:>4}{fp:>4}{fn:>4}{'  <-- FN/FP' if (fn or fp) else ''}")
print(f"\n>>> CP-F1 ({SPLIT}, min_v={MIN_V}, {MATCH}mm, Contact+CableEntry deduped):")
print(f"    F1={f1:.3f}  Jaccard={jac:.3f}  precision={prec:.3f}  recall={rec:.3f}  TP={TP} FP={FP} FN={FN}")
print("DONE")
