"""Direction-head diagnosis: WHERE does the ~74deg mean angle error live?

Breaks the matched-TP angle error down by (a) part family (PXC = human-made
direction GT vs wscaduniverse = CAD-derived, mesh-clearance-verified GT) and
(b) sign: is the predicted vector roughly the right AXIS but flipped 180deg
(cos < 0, a sign error the loss should have fixed), or genuinely off-axis?

Uses the checkpoint's own split manifest (written at train time) instead of
re-deriving the split -- immune to the reconstruct-the-wrong-split bug class.

Usage:
  python diag_direction.py checkpoints/cp_hp_v22_best.ckpt \\
      --manifest checkpoints/cp_hp_v22_best_manifest.json \\
      --gt-dir "C:\\...\\JSON" --extra-source wscad_corpus_v2 \\
      --thr 0.15 --device cpu
"""
import os
import json
import argparse
import logging

import numpy as np

logging.basicConfig(level=logging.ERROR)


def main(argv=None):
    ap = argparse.ArgumentParser(description="per-family direction-error breakdown")
    ap.add_argument("ckpt")
    ap.add_argument("--manifest", required=True,
                    help="the run's *_manifest.json (split ground truth)")
    ap.add_argument("--gt-dir", required=True, dest="gt_dir")
    ap.add_argument("--extra-source", action="append", default=[],
                    dest="extra_sources")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--thr", type=float, default=0.15)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-gpu-verts", type=int, default=14000,
                    dest="max_gpu_verts")
    args = ap.parse_args(argv)

    import json_dataset as jd
    import cp_regressor as cpr
    import cp_targets as ct
    import metrics as mcp
    import predict as pd

    with open(args.manifest, encoding="utf-8") as fh:
        man = json.load(fh)
    want = {p["part_nr"] for p in man["parts"] if p["split"] == args.split}

    parts = [p for p in jd.iter_parts(args.gt_dir) if str(p.part_nr) in want]
    for src in args.extra_sources:
        parts += [p for p in jd.iter_parts(src) if str(p.part_nr) in want]
    print(f"{args.split}: {len(parts)}/{len(want)} manifest parts found")

    model, meta, backbone, decode = cpr.load_inference(args.ckpt,
                                                       device=args.device)
    decode = dict(decode, heatmap_thresh=args.thr)

    rows = []
    for p in parts:
        _, gt_pts, gt_dirs = jd.dedup_connection_points(p)
        if not len(gt_pts):
            continue
        nodes, _ = pd.predict_part(backbone, model, meta, p, decode,
                                   device=args.device,
                                   max_gpu_verts=args.max_gpu_verts)
        if not nodes:
            continue
        pred_pts = np.array([n["entry_point"] for n in nodes], float)
        pred_dirs = np.array([n["approach_vector"] for n in nodes], float)
        matches, _, _ = mcp.match_predictions(pred_pts, np.asarray(gt_pts, float),
                                              5.0)
        fam = jd.prefix_family(p.part_nr)
        for pi, gj, dist in matches:
            d_pred = pred_dirs[pi] / (np.linalg.norm(pred_dirs[pi]) or 1.0)
            d_gt = np.asarray(gt_dirs[gj], float)
            d_gt = d_gt / (np.linalg.norm(d_gt) or 1.0)
            cos = float(np.clip(np.dot(d_pred, d_gt), -1, 1))
            rows.append({"family": fam, "part": str(p.part_nr),
                         "ang": float(np.degrees(np.arccos(cos))),
                         "cos": cos, "loc": float(dist)})

    if not rows:
        raise SystemExit("no matched detections -- nothing to diagnose")

    fams = sorted({r["family"] for r in rows})
    print(f"\n{'family':16s} {'nTP':>4} {'ang_med':>8} {'ang_mean':>9} "
          f"{'%flipped(cos<0)':>16} {'%axis_ok(|cos|>.9)':>19}")
    for fam in fams + ["ALL"]:
        rs = rows if fam == "ALL" else [r for r in rows if r["family"] == fam]
        angs = np.array([r["ang"] for r in rs])
        coss = np.array([r["cos"] for r in rs])
        print(f"{fam:16s} {len(rs):4d} {np.median(angs):8.1f} {angs.mean():9.1f} "
              f"{100*(coss<0).mean():15.0f}% {100*(np.abs(coss)>0.9).mean():18.0f}%")
    # the tell: high %flipped with high %axis_ok = SIGN problem (labels or loss);
    # low %axis_ok = the head genuinely hasn't learned direction.


if __name__ == "__main__":
    main()
