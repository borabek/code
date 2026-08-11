# -*- coding: utf-8 -*-
"""End-to-end thesis-pipeline STEP-transfer test: for each of the 9 bridge catalogs,
tessellate the WSCAD STEP twin, UNIFORM-REMESH it to ~6000 verts (same pipeline as the
training corpus), run the checkpoint, frame-align, and score vs the Desktop-JSON GT.
This is the deployment-realistic number for a model trained on the remeshed corpus.

Usage:  .venv/Scripts/python.exe eval_remesh_transfer.py checkpoints/cp_real_remesh_best.ckpt
"""
import sys, glob, os
import numpy as np
import cp_regressor as cpr, cp_targets as ct, json_dataset as jd, metrics as mcp
import train_cp as tc, step_to_json as sj, cad_eval as ce
from thesis_remesh import remesh_uniform

CK = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/cp_real_remesh_best.ckpt"
TEACHER = r"C:\Users\DE00024082\Desktop\JSON"
STEPDIR = "_cad_eval_pxc"
THR = 0.30; DEFL = 0.3; DIST = 5.0
dedup = tc._nms_radius(None, 5.0)
bridge = [l.strip() for l in open("_bridge_test.txt", encoding="utf-8") if l.strip()]
model, meta, _ = cpr.load_model(CK, device="cuda")
jp = {str(p.part_nr): p for p in jd.iter_parts(TEACHER) if str(p.part_nr) in set(bridge)}


def infer(V, F, pn):
    V = np.asarray(V, float); Vn, _, s = cpr.normalize_vertices(V)
    if "k_eig" in meta:                        # DiffusionNet meta has k_eig (no 'backbone' key)
        Vc = np.ascontiguousarray(V, np.float64); Fc = np.ascontiguousarray(F, np.int32)
        arr = cpr.infer_diffusionnet(model, meta, Vc, Fc, Vn,
                                     op_cache_dir="_op_cache", device="cuda", offset_scale=s)
    else:
        arr = cpr.infer_knngraph(model, meta, Vn, device="cuda", max_gpu_verts=14000,
                                 offset_scale=s, patch=cpr.is_patch_part(pn), part_nr=pn)
    return ct.decode_predictions(V, arr, heatmap_thresh=THR, nms_radius_mm=dedup, min_votes=1)


TP = FP = FN = 0
print(f"{'catalog':<10}{'step_v':>8}{'remesh_v':>9}  F1  TP/FP/FN")
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
    Vs = np.asarray(Vs, float)
    Vr, Fr = remesh_uniform(Vs, Fs, target=6000)      # SAME pipeline as training corpus
    R, t, res = ce.align_frames(Vr, Vj)
    preds = infer(Vr, Fr, pn)
    P = [{"point": R @ np.asarray(q["point"], float) + t,
          "direction": R @ np.asarray(q.get("direction", [0, 0, 1]), float)} for q in preds]
    r = mcp.keypoint_report(P, gt, gd, dist_thresh_mm=DIST)
    TP += r["tp"]; FP += r["fp"]; FN += r["fn"]
    f1 = 2 * r["tp"] / max(2 * r["tp"] + r["fp"] + r["fn"], 1)
    print(f"{cat:<10}{len(Vs):>8}{len(Vr):>9}  {f1:.2f}  {r['tp']}/{r['fp']}/{r['fn']}")

f1 = 2 * TP / max(2 * TP + FP + FN, 1); jac = TP / max(TP + FP + FN, 1)
print(f"\n>>> REMESH STEP TRANSFER: F1={f1:.3f} Jaccard={jac:.3f}  TP={TP} FP={FP} FN={FN}")
print(f"    (baseline cp_real_v1 STEP native 0.093 / cp_m0 overfit STEP 0.046)")
print("DONE")
