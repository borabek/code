# -*- coding: utf-8 -*-
"""Threshold sweep for the remesh STEP transfer: infer ONCE per bridge part (remeshed
STEP), decode at many heatmap thresholds, report F1 per threshold. The fixed-0.30 transfer
was recall-starved (2/18); a lower threshold may lift it a lot. hierpoint checkpoints only.
Usage: .venv/Scripts/python.exe eval_remesh_sweep.py checkpoints/cp_real_remesh_best.ckpt
"""
import sys, glob, os
import numpy as np
import cp_regressor as cpr, cp_targets as ct, json_dataset as jd, metrics as mcp
import train_cp as tc, step_to_json as sj, cad_eval as ce
from thesis_remesh import remesh_uniform

CK = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/cp_real_remesh_best.ckpt"
TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
TEACHER = r"C:\Users\DE00024082\Desktop\JSON"; STEPDIR = "_cad_eval_pxc"; DEFL = 0.3
dedup = tc._nms_radius(None, 5.0)
bridge = [l.strip() for l in open("_bridge_test.txt", encoding="utf-8") if l.strip()]
model, meta, _ = cpr.load_model(CK, device="cuda")
jp = {str(p.part_nr): p for p in jd.iter_parts(TEACHER) if str(p.part_nr) in set(bridge)}

cache = []                                       # (V_remesh, arr, R, t, gt, gd)
for pn in bridge:
    p = jp.get(pn)
    if p is None:
        continue
    cat = pn.split(".")[-1]
    _, gt, gd = jd.dedup_connection_points(p)
    Vj = np.asarray(p.vertices, float)
    steps = glob.glob(os.path.join(STEPDIR, f"*{cat}*.stp"))
    if not steps:
        continue
    Vs, Fs = sj.load_any_mesh(steps[0], deflection=DEFL)
    Vr, Fr = remesh_uniform(np.asarray(Vs, float), Fs, target=TARGET)
    R, t, _ = ce.align_frames(Vr, Vj)
    Vn, _, s = cpr.normalize_vertices(Vr)
    arr = cpr.infer_knngraph(model, meta, Vn, device="cuda", max_gpu_verts=14000,
                             offset_scale=s, patch=cpr.is_patch_part(pn), part_nr=pn)
    cache.append((Vr, arr, R, t, gt, gd))
print(f"inferred {len(cache)} remeshed bridge parts; sweeping thresholds\n")
print(f"  {'thr':>5} {'F1':>7} {'prec':>6} {'recall':>7}  TP/FP/FN")
best = None
for thr in (0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30):
    TP = FP = FN = 0
    for Vr, arr, R, t, gt, gd in cache:
        preds = ct.decode_predictions(Vr, arr, heatmap_thresh=thr, nms_radius_mm=dedup, min_votes=1)
        P = [{"point": R @ np.asarray(q["point"], float) + t,
              "direction": R @ np.asarray(q.get("direction", [0, 0, 1]), float)} for q in preds]
        r = mcp.keypoint_report(P, gt, gd, dist_thresh_mm=5.0)
        TP += r["tp"]; FP += r["fp"]; FN += r["fn"]
    f1 = 2 * TP / max(2 * TP + FP + FN, 1); pr = TP / max(TP + FP, 1); rc = TP / max(TP + FN, 1)
    mark = ""
    if best is None or f1 > best[1]:
        best = (thr, f1); mark = "  <=="
    print(f"  {thr:>5.2f} {f1:>7.3f} {pr:>6.3f} {rc:>7.3f}  {TP}/{FP}/{FN}{mark}")
print(f"\n>>> best thr={best[0]:.2f} F1={best[1]:.3f}  (fixed-0.30 was 0.148)")
print("DONE")
