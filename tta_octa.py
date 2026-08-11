"""Octahedral test-time augmentation for the WSCAD->CP model. FULL-COVERAGE lever.

The corpus is 100% axis-aligned (measured 2026-07-14: 1556/1556 CP directions are
exactly +-X/+-Y/+-Z). So the 24 octahedral rotations (90-degree multiples) map the
part onto a pose the model MIGHT read differently -- averaging over them is a
legitimate ensemble that cancels per-pose noise without inventing off-axis geometry.

IMPORTANT: this is NOT the uniform-SO(3) TTA that cost ~-10pp historically -- that
broke axis-alignment. Octahedral preserves it (see tests/test_octahedral.py).

Method (per part):
  for each rotation R in the octahedral set:
      rotate vertices by R, re-normalize, run the SAME inference,
      decode in the rotated frame, then rotate the predicted CP points back by R^T.
  Collect every pose's predictions, then keep only points that CONSENSUS-agree
  across >= --consensus poses (within the dedup radius). This filters pose-specific
  false positives and stabilises the true ones.

Run AFTER training (needs the GPU). Compares against the single-pose baseline.
Usage:
  python tta_octa.py --ckpt checkpoints/cp_hp_v31_ftc6_best.ckpt \
      --parts _val_parts.txt --n-poses 8 --consensus 2 --thr 0.30
"""
import argparse, json, os
import numpy as np
import cp_regressor as cpr, cp_targets as ct, json_dataset as jd, metrics as mcp
import train_cp as tc


def octahedral_matrices():
    """All 24 proper rotations (det=+1) that map the axes onto themselves."""
    import itertools
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            M = np.zeros((3, 3))
            for i in range(3):
                M[i, perm[i]] = signs[i]
            if abs(np.linalg.det(M) - 1.0) < 1e-9:   # proper only
                mats.append(M)
    return mats                                       # exactly 24


def cluster_consensus(points, dirs, scores, radius, min_agree):
    """Greedy merge of points from multiple poses; keep clusters seen by >= min_agree
    DISTINCT poses. Returns consensus (point, direction, score)."""
    if len(points) == 0:
        return []
    order = np.argsort(-scores)
    used = np.zeros(len(points), bool)
    out = []
    P = np.asarray(points)
    for i in order:
        if used[i]:
            continue
        d = np.linalg.norm(P - P[i], axis=1)
        members = np.where((d < radius) & (~used))[0]
        used[members] = True
        if len(members) >= min_agree:
            w = scores[members]
            out.append({"point": (P[members] * w[:, None]).sum(0) / w.sum(),
                        "direction": np.asarray(dirs)[members][np.argmax(w)],
                        "score": float(w.max()), "n_votes": int(len(members))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--parts", default="_val_parts.txt")
    ap.add_argument("--corpus", default="wscad_corpus_v8")
    ap.add_argument("--n-poses", type=int, default=8, dest="n_poses",
                    help="how many of the 24 octahedral poses to use (8 = fast, 24 = full)")
    ap.add_argument("--consensus", type=int, default=2,
                    help="min distinct poses that must agree to keep a detection")
    ap.add_argument("--thr", type=float, default=0.30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="_tta_result.json")
    a = ap.parse_args()

    val = [l.strip() for l in open(a.parts, encoding="utf-8") if l.strip()]
    model, meta, _ = cpr.load_model(a.ckpt, device=a.device)
    Rs = octahedral_matrices()[: a.n_poses]
    dedup = tc._nms_radius(None, 5.0)

    def infer_pose(V, R):
        Vr = V @ R.T
        Vn, _, scale = cpr.normalize_vertices(Vr)
        arr = cpr.infer_knngraph(model, meta, Vn, device=a.device, max_gpu_verts=14000,
                                 offset_scale=scale, patch=cpr.is_patch_part("wscaduniverse"),
                                 part_nr="tta")
        preds = ct.decode_predictions(Vr, arr, heatmap_thresh=a.thr,
                                      nms_radius_mm=dedup, min_votes=1)
        pts = np.array([q["point"] for q in preds]) if preds else np.zeros((0, 3))
        drs = np.array([q["direction"] for q in preds]) if preds else np.zeros((0, 3))
        scs = np.array([q.get("score", 0.0) for q in preds]) if preds else np.zeros(0)
        # rotate points AND directions back to the original frame
        return (pts @ R, drs @ R, scs) if len(pts) else (pts, drs, scs)

    base = dict(tp=0, fp=0, fn=0)
    tta = dict(tp=0, fp=0, fn=0)
    per_part = []
    for k, pn in enumerate(val, 1):
        f = os.path.join(a.corpus, pn + ".json")
        if not os.path.exists(f):
            continue
        p = next(iter(jd.iter_parts(f)))
        _, gt, gd = jd.dedup_connection_points(p)
        if not len(gt):
            continue
        V = np.asarray(p.vertices, float)

        # BASELINE (single canonical pose = identity)
        bpts, bdrs, bscs = infer_pose(V, np.eye(3))
        bpreds = [{"point": pp, "direction": dd} for pp, dd in zip(bpts, bdrs)]
        rb = mcp.keypoint_report(bpreds, gt, gd, dist_thresh_mm=5.0)
        base["tp"] += rb["tp"]; base["fp"] += rb["fp"]; base["fn"] += rb["fn"]

        # TTA (all poses -> consensus)
        allp, alld, alls = [], [], []
        for R in Rs:
            pts, drs, scs = infer_pose(V, R)
            allp.extend(pts); alld.extend(drs); alls.extend(scs)
        cons = cluster_consensus(np.array(allp) if allp else np.zeros((0, 3)),
                                 alld, np.array(alls) if alls else np.zeros(0),
                                 dedup, a.consensus)
        rt = mcp.keypoint_report(cons, gt, gd, dist_thresh_mm=5.0)
        tta["tp"] += rt["tp"]; tta["fp"] += rt["fp"]; tta["fn"] += rt["fn"]
        per_part.append((pn, rb["tp"], rb["fp"], rb["fn"], rt["tp"], rt["fp"], rt["fn"]))
        if k % 30 == 0:
            print(f"  {k}/{len(val)} parts...", flush=True)

    def jac(d): return d["tp"] / (d["tp"] + d["fp"] + d["fn"]) if d["tp"] else 0
    def f1(d): return 2 * d["tp"] / (2 * d["tp"] + d["fp"] + d["fn"]) if d["tp"] else 0
    print("\n=== OCTAHEDRAL TTA ({} poses, consensus>={}) ===".format(len(Rs), a.consensus))
    print(f"  BASELINE (1 poz): TP={base['tp']} FP={base['fp']} FN={base['fn']}  "
          f"Jaccard={jac(base):.4f} F1={f1(base):.4f}")
    print(f"  TTA             : TP={tta['tp']} FP={tta['fp']} FN={tta['fn']}  "
          f"Jaccard={jac(tta):.4f} F1={f1(tta):.4f}")
    print(f"  DELTA Jaccard: {jac(tta)-jac(base):+.4f}")
    json.dump({"base": base, "tta": tta, "per_part": per_part}, open(a.out, "w"))
    print("DONE")


if __name__ == "__main__":
    main()
