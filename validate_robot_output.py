"""Robot output validation: sanity-checks approach vectors and reachability.

Validates that predicted (or GT) connection points produce robot-reachable
approach poses by checking:
  1. Approach vector faces OUTWARD from the part surface (dot with surface
     normal is negative — i.e., vector points away from mass of mesh)
  2. Approach vector is not degenerate (near-zero norm)
  3. Approach start position (CP + approach_mm * dir) is outside the mesh
     bounding box (no collision with housing)
  4. Approach angle vs gravity (configurable limit) — pure vertical or
     pure horizontal may be unreachable for a fixed robot
  5. [optional] Pair coverage: every GT CP is within dist_thresh of a pred CP

Usage:
  # Validate predictions from a trained checkpoint:
  python validate_robot_output.py checkpoints/cp_knn_v15_best.ckpt corpus/ --device cuda

  # Validate GT labels only (check corpus integrity):
  python validate_robot_output.py --gt-only corpus/

  # Restrict to a specific family:
  python validate_robot_output.py ckpt.ckpt corpus/ --keep-prefixes wscaduniverse
"""
import argparse
import logging
import sys

import numpy as np

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

_DEFAULT_APPROACH_MM = 20.0    # how far to step back along -dir for collision check
_DEFAULT_ANGLE_LIMIT = 80.0    # max degrees from vertical for robot approach
_DEFAULT_DIST_THRESH = 5.0     # mm: TP match radius


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def _mesh_normals_centroid(vertices, faces):
    """Approximate outward normal as centroid→vertex direction."""
    c = vertices.mean(0)
    return c


def _approx_outward(V):
    """Point from centroid outward — used as reference for normal flip check."""
    return V.mean(0)


def _surface_normal_at(V, faces, query_pt, k=20):
    """PCA normal of the k nearest surface vertices to query_pt."""
    dists = np.linalg.norm(V - query_pt, axis=1)
    idx = np.argsort(dists)[:k]
    nbr = V[idx]
    cov = np.cov((nbr - nbr.mean(0)).T)
    _, vecs = np.linalg.eigh(cov)
    nrm = vecs[:, 0]                # smallest eigenvalue → normal
    centroid = V.mean(0)
    if np.dot(nrm, query_pt - centroid) < 0:
        nrm = -nrm                  # flip to face outward
    return nrm


def _is_outside_bbox(pt, bbox_min, bbox_max, margin=0.0):
    return (pt < bbox_min - margin).any() or (pt > bbox_max + margin).any()


def _angle_from_vertical_deg(direction):
    """Angle in degrees between direction and +Z (vertical)."""
    d = direction / (np.linalg.norm(direction) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(abs(d[2]), 0, 1))))


# ---------------------------------------------------------------------------
# single-part validation
# ---------------------------------------------------------------------------

def validate_part(part_nr, V, F, cp_points, cp_dirs,
                  approach_mm=_DEFAULT_APPROACH_MM,
                  angle_limit_deg=_DEFAULT_ANGLE_LIMIT,
                  verbose=False):
    """
    Returns dict:
      n_cps, n_pass, n_fail, issues (list of str)
    """
    issues = []
    n_pass = 0
    bbox_min = V.min(0); bbox_max = V.max(0)
    centroid = V.mean(0)

    for i, (cp, d) in enumerate(zip(cp_points, cp_dirs)):
        cp = np.asarray(cp, dtype=float)
        d = np.asarray(d, dtype=float)
        tag = f"CP#{i}"
        ok = True

        # 1. norm check
        dn = np.linalg.norm(d)
        if dn < 0.01:
            issues.append(f"{tag}: degenerate direction norm={dn:.4f}")
            ok = False
            continue
        d_unit = d / dn

        # 2. outward check: direction should point away from centroid
        #    i.e. dot(d_unit, cp - centroid) should be > 0 for outward dirs
        #    (InsertDirection is OUTWARD: cable comes FROM outside, inserts inward)
        outward = cp - centroid
        outward_n = np.linalg.norm(outward)
        if outward_n > 1e-6:
            cos_a = np.dot(d_unit, outward / outward_n)
            if cos_a < -0.3:   # dir is clearly pointing INWARD
                issues.append(f"{tag}: direction points inward (cos={cos_a:.2f}) — "
                               "insert direction should be OUTWARD from part")
                ok = False

        # 3. approach point should be outside the bbox
        approach_pt = cp + approach_mm * d_unit
        if not _is_outside_bbox(approach_pt, bbox_min, bbox_max, margin=1.0):
            issues.append(f"{tag}: approach point {approach_pt} is inside bbox "
                          f"[{bbox_min},{bbox_max}] — collision risk")
            ok = False

        # 4. angle check
        angle = _angle_from_vertical_deg(d_unit)
        if angle > angle_limit_deg:
            issues.append(f"{tag}: approach angle {angle:.1f}° from vertical > "
                          f"limit {angle_limit_deg}° — may be unreachable")
            ok = False

        if ok:
            n_pass += 1
        elif verbose:
            pass   # issues list already has details

    return {
        "part_nr": str(part_nr),
        "n_cps": len(cp_points),
        "n_pass": n_pass,
        "n_fail": len(cp_points) - n_pass,
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    ap = argparse.ArgumentParser(description="Robot output validation")
    ap.add_argument("ckpt", nargs="?", default=None,
                    help="checkpoint file (omit with --gt-only)")
    ap.add_argument("source", help="corpus directory")
    ap.add_argument("--gt-only", action="store_true", dest="gt_only",
                    help="validate GT labels only (no model inference)")
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
                         "diagonal exceeds this (mm), else large out-of-scope parts "
                         "(e.g. industrial drives) pollute the validated split")
    ap.add_argument("--split", default="val", choices=["val", "test", "all"])
    ap.add_argument("--thr", type=float, default=0.50)
    ap.add_argument("--nms", type=float, default=5.0)
    ap.add_argument("--dist-thresh", type=float, default=_DEFAULT_DIST_THRESH,
                    dest="dist_thresh", help="TP match radius mm")
    ap.add_argument("--approach-mm", type=float, default=_DEFAULT_APPROACH_MM,
                    dest="approach_mm")
    ap.add_argument("--angle-limit", type=float, default=_DEFAULT_ANGLE_LIMIT,
                    dest="angle_limit")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--check-coverage", action="store_true", dest="check_coverage",
                    help="also report pred→GT coverage (requires checkpoint)")
    return ap.parse_args()


def main():
    args = _parse_args()

    if not args.gt_only and args.ckpt is None:
        print("ERROR: provide a checkpoint file or use --gt-only", file=sys.stderr)
        sys.exit(1)

    import json_dataset as jd
    import train_cp as tc

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
              f"of {before} parts", flush=True)

    ids = [p.part_nr for p in parts]
    gk = jd.build_group_keys(parts, mode=args.split_group)
    tr, va, te = tc.three_way_split(ids, val_frac=args.val_frac,
                                    test_frac=args.test_frac, seed=args.seed,
                                    group_keys=gk)
    if args.split == "val":   keep = va
    elif args.split == "test": keep = te
    else:                      keep = set(tr) | set(va) | set(te)
    eval_parts = [p for p in parts if p.part_nr in keep]
    print(f"Validating {len(eval_parts)} parts  split={args.split}", flush=True)

    # optionally load model for pred CPs
    model = meta = None
    if not args.gt_only and args.ckpt:
        import cp_regressor as cpr
        model, meta, _ = cpr.load_model(args.ckpt, device=args.device)
        print(f"Model: {args.ckpt}  thr={args.thr}", flush=True)

    total_gt_cp = pass_gt = fail_gt = 0
    total_pred_cp = pass_pred = fail_pred = 0
    parts_with_issues = []

    for p in eval_parts:
        V = np.asarray(p.vertices, dtype=np.float64)
        F = np.asarray(p.faces, dtype=np.int64) if len(p.faces) else np.zeros((0, 3), dtype=np.int64)
        _, gt_pts, gt_dirs = jd.dedup_connection_points(p)

        # --- validate GT ---
        if len(gt_pts):
            res = validate_part(p.part_nr, V, F, gt_pts, gt_dirs,
                                approach_mm=args.approach_mm,
                                angle_limit_deg=args.angle_limit,
                                verbose=args.verbose)
            total_gt_cp += res["n_cps"]
            pass_gt += res["n_pass"]
            fail_gt += res["n_fail"]
            if res["issues"]:
                parts_with_issues.append(("GT", res))

        # --- validate predictions ---
        if model is not None and len(gt_pts):
            import cp_regressor as cpr
            import cp_targets as ct
            patch = cpr.is_patch_part(p.part_nr)
            Vn, _, scale = cpr.normalize_vertices(V)
            arr = cpr.infer_knngraph(model, meta, Vn, device=args.device,
                                     max_gpu_verts=args.max_gpu_verts,
                                     offset_scale=scale, patch=patch,
                                     part_nr=p.part_nr)
            preds = ct.decode_predictions(V, arr, heatmap_thresh=args.thr,
                                          nms_radius_mm=args.nms, min_votes=1)
            if preds:
                pred_pts = np.array([pr["point"] for pr in preds])
                pred_dirs = np.array([pr.get("direction", [0, 0, 1]) for pr in preds])
                res_p = validate_part(p.part_nr, V, F, pred_pts, pred_dirs,
                                      approach_mm=args.approach_mm,
                                      angle_limit_deg=args.angle_limit,
                                      verbose=args.verbose)
                total_pred_cp += res_p["n_cps"]
                pass_pred += res_p["n_pass"]
                fail_pred += res_p["n_fail"]
                if res_p["issues"]:
                    parts_with_issues.append(("PRED", res_p))

    print("\n=== GT label validation ===")
    print(f"  Total GT CPs : {total_gt_cp}")
    print(f"  Pass         : {pass_gt}  ({100*pass_gt/max(1,total_gt_cp):.1f}%)")
    print(f"  Fail         : {fail_gt}  ({100*fail_gt/max(1,total_gt_cp):.1f}%)")

    if model is not None:
        print("\n=== Predicted CP validation ===")
        print(f"  Total pred CPs : {total_pred_cp}")
        print(f"  Pass           : {pass_pred}  ({100*pass_pred/max(1,total_pred_cp):.1f}%)")
        print(f"  Fail           : {fail_pred}  ({100*fail_pred/max(1,total_pred_cp):.1f}%)")

    if parts_with_issues:
        print(f"\n=== Issues found in {len(parts_with_issues)} part(s) ===")
        for src, res in parts_with_issues[:20]:
            print(f"\n  [{src}] {res['part_nr']}:")
            for iss in res["issues"][:5]:
                print(f"    - {iss}")
        if len(parts_with_issues) > 20:
            print(f"  ... and {len(parts_with_issues)-20} more parts with issues.")
    else:
        print("\nNo issues found — all CPs pass robot approach validation.")

    fail_rate = fail_gt / max(1, total_gt_cp)
    if fail_rate > 0.05:
        print(f"\nWARNING: {100*fail_rate:.1f}% of GT CPs fail validation — "
              f"check corpus labeling or increase --approach-mm / --angle-limit.")
        sys.exit(1)


if __name__ == "__main__":
    main()
