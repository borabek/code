# -*- coding: utf-8 -*-
"""v31 best.ckpt: full-coverage Jaccard + selective coverage curve on the frozen
WSCAD val set, scored against v8 CAD labels. No human-GT, no leakage.

Run:  .venv/Scripts/python.exe measure_baseline.py
"""
import json, os
import numpy as np
import cp_regressor as cpr, cp_targets as ct, json_dataset as jd, metrics as mcp
import train_cp as tc

CKPT = "checkpoints/cp_hp_v31_ftc6_best.ckpt"
VAL = "_val_parts.txt"
CORPUS = "wscad_corpus_v8"
THR = 0.30

val = [l.strip() for l in open(VAL, encoding="utf-8") if l.strip()]
print(f"model: {CKPT}\nval:   {len(val)} parts (v8 CAD labels)\n")
model, meta, _ = cpr.load_model(CKPT, device="cuda")
dedup = tc._nms_radius(None, 5.0)

rows = []
for k, pn in enumerate(val, 1):
    f = os.path.join(CORPUS, pn + ".json")
    if not os.path.exists(f):
        continue
    p = next(iter(jd.iter_parts(f)))
    _, gt, gd = jd.dedup_connection_points(p)
    if not len(gt):
        continue
    V = np.asarray(p.vertices, float)
    Vn, _, scale = cpr.normalize_vertices(V)
    arr = cpr.infer_knngraph(model, meta, Vn, device="cuda", max_gpu_verts=14000,
                             offset_scale=scale, patch=cpr.is_patch_part(p.part_nr),
                             part_nr=pn)
    preds = ct.decode_predictions(V, arr, heatmap_thresh=THR,
                                  nms_radius_mm=dedup, min_votes=1)
    r = mcp.keypoint_report(preds, gt, gd, dist_thresh_mm=5.0)
    scores = [q.get("score", 0.0) for q in preds]
    nvotes = [q.get("n_votes", 0) for q in preds]
    dirs = set(int(np.argmax(np.abs(q.get("direction", [0, 0, 1])))) for q in preds)
    rows.append(dict(pn=pn, tp=r["tp"], fp=r["fp"], fn=r["fn"], ngt=len(gt),
                     npred=len(preds),
                     mean_conf=float(np.mean(scores)) if scores else 0.0,
                     mean_votes=float(np.mean(nvotes)) if nvotes else 0.0,
                     ndir=len(dirs)))
    if k % 40 == 0:
        print(f"  {k}/{len(val)} parts...", flush=True)

TP = sum(x["tp"] for x in rows); FP = sum(x["fp"] for x in rows); FN = sum(x["fn"] for x in rows)
jac = TP / (TP + FP + FN); f1 = 2 * TP / (2 * TP + FP + FN)
print("\n" + "=" * 56)
print("FULL COVERAGE (all parts, automatic)")
print(f"  TP={TP}  FP={FP}  FN={FN}")
print(f"  F1={f1:.4f}   Jaccard (accuracy) = {jac:.4f}")

# risk score: low conf / few votes / pred-gt mismatch / many directions => risky
def risk(x):
    ratio = x["npred"] / max(x["ngt"], 1)
    return (-x["mean_conf"] * 2.0 - min(x["mean_votes"], 20) / 20
            + abs(np.log(ratio + 1e-6)) + x["ndir"] * 0.3)
for x in rows:
    x["risk"] = risk(x)
rows.sort(key=lambda x: x["risk"])

print("\nSELECTIVE (accept least-risky first)")
print(f"  {'coverage':>9}{'parts':>6}{'acc-Jaccard':>13}")
hit90 = 0
for frac in (0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00):
    kk = int(len(rows) * frac)
    acc = rows[:kk]
    tp = sum(x["tp"] for x in acc); den = sum(x["tp"] + x["fp"] + x["fn"] for x in acc)
    j = tp / den if den else 0
    mark = "  <= >=0.90" if j >= 0.90 else ""
    if j >= 0.90:
        hit90 = frac
    print(f"  {frac:>8.0%}{kk:>6}{j:>13.4f}{mark}")
print(f"\n  >>> Jaccard>=0.90 max coverage: {hit90:.0%}"
      + ("  (=> selective works)" if hit90 else "  (=> below 0.90 even at 40%)"))

# individual-part ceiling
js = [x["tp"] / (x["tp"] + x["fp"] + x["fn"]) if (x["tp"] + x["fp"] + x["fn"]) else 0 for x in rows]
print(f"  parts individually >=0.90 Jaccard: {sum(1 for j in js if j >= 0.90)}/{len(rows)}")
json.dump(rows, open("_baseline_rows.json", "w"))
print("\nsaved: _baseline_rows.json")
print("DONE")
