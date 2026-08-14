"""Per-prefix decode threshold auto-tuner.

Sweeps heatmap thresholds on the val set per part-family prefix (wscaduniverse,
PXC, ABB, SYNTH, ...) and finds the F1-maximising threshold for each. Results
are:
  1. Printed as a table
  2. Embedded into the checkpoint's 'decode' dict as 'prefix_thresholds'
     so predict.py / infer_knngraph can auto-select the right threshold at
     inference time without any manual flag.

Usage:
  python tune_thresholds.py checkpoints/cp_knn_v15_best.ckpt corpus/ --device cuda
  python tune_thresholds.py ckpt.ckpt corpus/ --extra-source wscad_corpus \\
      --keep-prefixes wscaduniverse,PXC --split-group geometry --write
"""
import argparse
import logging
import sys

import numpy as np

import json_dataset as jd

logging.basicConfig(level=logging.ERROR)


def _parse_args():
    ap = argparse.ArgumentParser(description="Per-prefix decode threshold tuner")
    ap.add_argument("ckpt", help="checkpoint (.ckpt) to tune and optionally update")
    ap.add_argument("source", help="corpus directory")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-gpu-verts", type=int, default=7000, dest="max_gpu_verts")
    ap.add_argument("--val-frac", type=float, default=0.2, dest="val_frac")
    ap.add_argument("--test-frac", type=float, default=0.15, dest="test_frac")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-group", default="geometry", dest="split_group",
                    choices=["none", "prefix", "geometry"])
    ap.add_argument("--extra-source", action="append", default=[], dest="extra_sources")
    ap.add_argument("--keep-prefixes", default=None)
    ap.add_argument("--max-bbox-mm", type=float, default=None, dest="max_bbox_mm",
                    help="scope filter (MUST match training): drop parts whose bbox "
                         "diagonal exceeds this (mm). Use 250 for the connector model, "
                         "else large drives pollute the eval and tank F1.")
    ap.add_argument("--thresholds", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70",
                    help="comma-separated thresholds to sweep")
    ap.add_argument("--nms", type=float, default=5.0)
    ap.add_argument("--dist", type=float, default=5.0)
    ap.add_argument("--write", action="store_true",
                    help="embed best thresholds into the checkpoint's decode dict")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    return ap.parse_args()


def _prefix(part_nr):
    """Extract family prefix from a PartNr string -- delegates to
    json_dataset.prefix_family, the single source of truth also used by
    predict.py to look up these thresholds at inference time."""
    return jd.prefix_family(part_nr)


def main():
    args = _parse_args()
    thresholds = [float(x) for x in args.thresholds.split(",")]

    import json_dataset as jd
    import cp_regressor as cpr
    import cp_targets as ct
    import metrics as mcp
    import train_cp as tc

    model, meta, backbone = cpr.load_model(args.ckpt, device=args.device)

    parts = list(jd.iter_parts(args.source))
    for extra in args.extra_sources:
        parts.extend(jd.iter_parts(extra))
    if args.keep_prefixes:
        prefs = tuple(s.strip() for s in args.keep_prefixes.split(",") if s.strip())
        parts = [p for p in parts if str(p.part_nr).startswith(prefs)]
    if args.max_bbox_mm:
        # MUST match training scope (train_cp.py:301) or large drives pollute the eval
        kept = []
        for p in parts:
            V = np.asarray(p.vertices, dtype=np.float64)
            diag = float(np.linalg.norm(V.max(0) - V.min(0))) if len(V) else 0.0
            if diag <= args.max_bbox_mm:
                kept.append(p)
        print(f"scope filter --max-bbox-mm {args.max_bbox_mm:.0f}: kept {len(kept)} "
              f"of {len(parts)} parts", flush=True)
        parts = kept

    ids = [p.part_nr for p in parts]
    gk = jd.build_group_keys(parts, mode=args.split_group)
    tr, va, te = tc.three_way_split(ids, val_frac=args.val_frac,
                                    test_frac=args.test_frac, seed=args.seed,
                                    group_keys=gk)
    keep_ids = va if args.split == "val" else te
    eval_parts = [p for p in parts if p.part_nr in keep_ids and p.n_cps > 0]

    print(f"Tuning on {args.split} set: {len(eval_parts)} parts with CPs", flush=True)
    print("Running inference ...", flush=True)

    # cache per-part predictions (threshold-independent)
    cache = []
    for p in eval_parts:
        V = np.asarray(p.vertices, dtype=np.float64)
        _, gt_pts, gt_dirs = jd.dedup_connection_points(p)
        if len(gt_pts) == 0:
            continue
        patch = cpr.is_patch_part(p.part_nr)
        Vn, _, scale = cpr.normalize_vertices(V)
        arr = cpr.infer_knngraph(model, meta, Vn, device=args.device,
                                 max_gpu_verts=args.max_gpu_verts,
                                 offset_scale=scale, patch=patch,
                                 part_nr=p.part_nr)
        cache.append((_prefix(p.part_nr), V, gt_pts, gt_dirs, arr))

    # group by prefix
    prefixes = sorted({r[0] for r in cache})

    def _f1_for_prefix(pfx, thr):
        tp = fp = fn = 0
        for (p, V, gpts, gdirs, arr) in cache:
            if p != pfx:
                continue
            preds = ct.decode_predictions(V, arr, heatmap_thresh=thr,
                                          nms_radius_mm=args.nms, min_votes=1)
            rep = mcp.keypoint_report(preds, gpts, gdirs, dist_thresh_mm=args.dist)
            tp += rep["tp"]; fp += rep["fp"]; fn += rep["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    def _f1_all(thr):
        tp = fp = fn = 0
        for (p, V, gpts, gdirs, arr) in cache:
            preds = ct.decode_predictions(V, arr, heatmap_thresh=thr,
                                          nms_radius_mm=args.nms, min_votes=1)
            rep = mcp.keypoint_report(preds, gpts, gdirs, dist_thresh_mm=args.dist)
            tp += rep["tp"]; fp += rep["fp"]; fn += rep["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    print(f"\n{'Prefix':30s}  {'n':>4}  {'best_thr':>8}  {'F1':>6}")
    print("-" * 55)

    best_per_prefix = {}
    for pfx in prefixes:
        n = sum(1 for r in cache if r[0] == pfx)
        best_thr, best_f1 = 0.5, 0.0
        for thr in thresholds:
            f1 = _f1_for_prefix(pfx, thr)
            if f1 > best_f1:
                best_f1 = f1; best_thr = thr
        best_per_prefix[pfx] = best_thr
        print(f"  {pfx:28s}  {n:4d}  {best_thr:8.2f}  {100*best_f1:6.1f}%")

    # global best (single threshold for every part)
    g_best_thr, g_best_f1 = 0.5, 0.0
    for thr in thresholds:
        f1 = _f1_all(thr)
        if f1 > g_best_f1:
            g_best_f1 = f1; g_best_thr = thr
    print(f"\n  {'GLOBAL':28s}  {len(cache):4d}  {g_best_thr:8.2f}  {100*g_best_f1:6.1f}%")

    # combined: each part decoded at ITS OWN prefix's best threshold, pooled TP/FP/FN.
    # This is the number that actually matters for deployment (what --write embeds).
    tp = fp = fn = 0
    for (pfx, V, gpts, gdirs, arr) in cache:
        thr = best_per_prefix[pfx]
        preds = ct.decode_predictions(V, arr, heatmap_thresh=thr,
                                      nms_radius_mm=args.nms, min_votes=1)
        rep = mcp.keypoint_report(preds, gpts, gdirs, dist_thresh_mm=args.dist)
        tp += rep["tp"]; fp += rep["fp"]; fn += rep["fn"]
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    c_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(f"  {'COMBINED (per-prefix thr)':28s}  {len(cache):4d}  {'mixed':>8}  {100*c_f1:6.1f}%"
          f"   (P={100*prec:.1f}% R={100*rec:.1f}% TP={tp} FP={fp} FN={fn})")

    best_per_prefix["__global__"] = g_best_thr
    print(f"\nBest per-prefix thresholds: {best_per_prefix}")

    if args.write:
        import torch
        ckpt_path = args.ckpt
        # Re-save: load full ckpt, update decode, write back atomically
        full = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        existing_decode = full.get("decode") or {}
        existing_decode["prefix_thresholds"] = best_per_prefix
        existing_decode["global_threshold"] = g_best_thr
        full["decode"] = existing_decode
        tmp = ckpt_path + ".tmp"
        torch.save(full, tmp)
        import os; os.replace(tmp, ckpt_path)
        print(f"Thresholds embedded into {ckpt_path}")
    else:
        print("\n(Pass --write to embed these thresholds into the checkpoint)")


if __name__ == "__main__":
    main()
