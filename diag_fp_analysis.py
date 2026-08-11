"""False Positive location analysis: WHERE does the model hallucinate CPs?

Two FP types have different fixes:
  1. NEAR-MISS FP: close to a real GT CP but outside the match radius — a
     localization error, not a hallucination. Fix: tighten NMS or lower distance
     threshold (already a TP morally). Model is right, metric is harsh.
  2. CLUTTER FP: far from any GT CP — a true hallucination. Fix: more diverse
     training data, higher threshold, or architectural changes.

This script separates and characterizes both types.

Usage:
  python diag_fp_analysis.py checkpoints/cp_knn_v15_best.ckpt corpus/ --device cuda
  python diag_fp_analysis.py ckpt.ckpt corpus/ --keep-prefixes wscaduniverse,PXC \\
      --split-group geometry --thr 0.50
"""
import argparse
import logging
import numpy as np

logging.basicConfig(level=logging.ERROR)

_NEAR_MISS_MM = 10.0   # FP within this distance of a GT CP = "near-miss"


def _parse_args():
    ap = argparse.ArgumentParser(description="FP location analysis")
    ap.add_argument("ckpt"); ap.add_argument("source")
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
                    help="scope filter (mm): drop parts with bbox diagonal above this "
                         "(use ~250 to match a --max-bbox-mm training run)")
    ap.add_argument("--thr", type=float, default=0.50)
    ap.add_argument("--nms", type=float, default=5.0)
    ap.add_argument("--dist", type=float, default=5.0, help="TP match radius mm")
    ap.add_argument("--near-miss-mm", type=float, default=_NEAR_MISS_MM,
                    help="FP within this distance of a GT CP = near-miss")
    ap.add_argument("--split", default="val", choices=["val", "test", "all"])
    return ap.parse_args()


def main():
    args = _parse_args()
    import json_dataset as jd
    import cp_regressor as cpr
    import cp_targets as ct
    import metrics as mcp
    import train_cp as tc

    model, meta, _ = cpr.load_model(args.ckpt, device=args.device)
    parts = list(jd.iter_parts(args.source))
    for extra in args.extra_sources:
        parts.extend(jd.iter_parts(extra))
    if args.keep_prefixes:
        prefs = tuple(s.strip() for s in args.keep_prefixes.split(",") if s.strip())
        parts = [p for p in parts if str(p.part_nr).startswith(prefs)]
    if args.max_bbox_mm:
        def _diag(p):
            V = np.asarray(p.vertices, dtype=np.float64)
            return float(np.linalg.norm(V.max(0) - V.min(0))) if len(V) else 0.0
        before = len(parts)
        parts = [p for p in parts if _diag(p) <= args.max_bbox_mm]
        print(f"scope filter --max-bbox-mm {args.max_bbox_mm:.0f}: kept {len(parts)} "
              f"of {before} parts")
    ids = [p.part_nr for p in parts]
    gk = jd.build_group_keys(parts, mode=args.split_group)
    tr, va, te = tc.three_way_split(ids, val_frac=args.val_frac,
                                    test_frac=args.test_frac, seed=args.seed,
                                    group_keys=gk)
    keep = (va if args.split == "val" else te if args.split == "test"
            else set(tr) | set(va) | set(te))
    eval_parts = [p for p in parts if p.part_nr in keep and p.n_cps > 0]
    print(f"model={args.ckpt}  split={args.split}  parts={len(eval_parts)}  thr={args.thr}")

    near_miss_dists, clutter_scores, clutter_near_gt = [], [], []
    total_tp = total_fp = total_fn = 0
    near_miss_count = clutter_count = 0

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
        preds = ct.decode_predictions(V, arr, heatmap_thresh=args.thr,
                                      nms_radius_mm=args.nms, min_votes=1)
        rep = mcp.keypoint_report(preds, gt_pts, gt_dirs, dist_thresh_mm=args.dist)
        total_tp += rep["tp"]; total_fp += rep["fp"]; total_fn += rep["fn"]

        if not preds or len(gt_pts) == 0:
            continue

        pred_pts = np.array([p2["point"] for p2 in preds], dtype=float)
        pred_scores = np.array([p2.get("score", 0.5) for p2 in preds])
        from metrics import match_predictions
        matches, unmatched_pred, _ = match_predictions(pred_pts, gt_pts, args.dist)
        matched_pred = {pi for pi, _, _ in matches}

        for pi in unmatched_pred:   # FP indices
            fp_pt = pred_pts[pi]
            # distance to nearest GT CP
            dists_to_gt = np.linalg.norm(gt_pts - fp_pt, axis=1)
            nearest_gt_dist = float(dists_to_gt.min())
            score = float(pred_scores[pi])
            if nearest_gt_dist <= args.near_miss_mm:
                near_miss_dists.append(nearest_gt_dist)
                near_miss_count += 1
            else:
                clutter_count += 1
                clutter_scores.append(score)
                clutter_near_gt.append(nearest_gt_dist)

    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    total_fp_all = total_fp
    print(f"\nOverall: TP={total_tp} FP={total_fp_all} FN={total_fn}  "
          f"P={100*prec:.1f}%  R={100*rec:.1f}%  F1={100*f1:.1f}%\n")

    print(f"=== FP breakdown (total FP = {total_fp_all}) ===")
    print(f"  Near-miss FP (within {args.near_miss_mm:.0f}mm of a GT CP): "
          f"{near_miss_count}  ({100*near_miss_count/max(1,total_fp_all):.0f}%)")
    if near_miss_dists:
        print(f"    dist p25={np.percentile(near_miss_dists,25):.1f}  "
              f"median={np.percentile(near_miss_dists,50):.1f}  "
              f"p75={np.percentile(near_miss_dists,75):.1f} mm")
        print(f"    => localization error, not hallucination. "
              f"Lower --dist threshold or tighter NMS to convert to TP.")

    print(f"\n  Clutter FP (far from any GT CP): "
          f"{clutter_count}  ({100*clutter_count/max(1,total_fp_all):.0f}%)")
    if clutter_scores:
        print(f"    score p25={np.percentile(clutter_scores,25):.3f}  "
              f"median={np.percentile(clutter_scores,50):.3f}  "
              f"p75={np.percentile(clutter_scores,75):.3f}")
        print(f"    nearest GT dist: median={np.median(clutter_near_gt):.1f}mm")
        pct_low_score = sum(1 for s in clutter_scores if s < 0.4) / len(clutter_scores)
        print(f"    {100*pct_low_score:.0f}% of clutter FPs have score < 0.40 "
              f"(raisable with higher threshold)")
        print(f"    => {'Threshold too low' if pct_low_score > 0.5 else 'Model confused by surface geometry'}")

    print(f"\n=== Diagnosis ===")
    nm_frac = near_miss_count / max(1, total_fp_all)
    cl_frac = clutter_count / max(1, total_fp_all)
    if nm_frac > 0.5:
        print(f"  LOCALIZATION-limited: {100*nm_frac:.0f}% of FPs are near-miss.")
        print(f"  Fix: match dist={args.dist}mm -> {args.near_miss_mm:.0f}mm gives TP,")
        print(f"       OR: improve offset prediction (w_off).")
    else:
        print(f"  HALLUCINATION-limited: {100*cl_frac:.0f}% of FPs are clutter.")
        if clutter_scores and np.median(clutter_scores) < 0.5:
            print(f"  Fix: raise threshold (many clutter FPs are low-score) OR")
            print(f"       add more negative training examples (cable_distractor).")
        else:
            print(f"  Fix: model is confident about wrong locations.")
            print(f"       Needs more diverse training data / cable_distractor augmentation.")


if __name__ == "__main__":
    main()
