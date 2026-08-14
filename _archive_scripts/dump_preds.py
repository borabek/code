# -*- coding: utf-8 -*-
"""Dump model predictions to a viz_labels-compatible JSON, using the EXACT
eval_real machinery (infer_knngraph + decode_predictions at the locked threshold).
Then: viz_labels.py --list <parts> --pred _preds.json  (green=GT, red=pred).

Usage:
  .venv/Scripts/python.exe dump_preds.py checkpoints/cp_real_v1_best.ckpt --list _viz_pick.txt --out _preds.json
"""
import argparse, json
import numpy as np
import cp_regressor as cpr, cp_targets as ct, json_dataset as jd
import train_cp as tc

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--list", dest="lst", required=True)
ap.add_argument("--corpus", default=r"C:\Users\DE00024082\Desktop\JSON")
ap.add_argument("--out", default="_preds.json")
ap.add_argument("--thr", type=float, default=0.30)
ap.add_argument("--device", default="cuda")
a = ap.parse_args()

want = set(l.strip() for l in open(a.lst, encoding="utf-8") if l.strip())
model, meta, _ = cpr.load_model(a.ckpt, device=a.device)
dedup = tc._nms_radius(None, 5.0)

parts_out = []
for p in jd.iter_parts(a.corpus):
    pn = str(p.part_nr)
    if pn not in want:
        continue
    V = np.asarray(p.vertices, float)
    Vn, _, scale = cpr.normalize_vertices(V)
    arr = cpr.infer_knngraph(model, meta, Vn, device=a.device, max_gpu_verts=14000,
                             offset_scale=scale, patch=cpr.is_patch_part(pn), part_nr=pn)
    preds = ct.decode_predictions(V, arr, heatmap_thresh=a.thr,
                                  nms_radius_mm=dedup, min_votes=1)
    parts_out.append({"part_nr": pn,
                      "connection_points": [{"point": list(map(float, q["point"]))} for q in preds]})
    print(f"  {pn:<28} {len(preds)} pred CP")

json.dump({"parts": parts_out}, open(a.out, "w"))
print(f"wrote {a.out} ({len(parts_out)} parts)")
