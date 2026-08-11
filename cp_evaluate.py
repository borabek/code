# -*- coding: utf-8 -*-
"""THE single CP evaluator. Reads cp_config.json (one CP definition version + one eval config),
keeps GT and PREDICTION post-processing strictly separate, and writes a machine-readable receipt.

GT  = human labels -> connection_points(gt_classes, gt_min_vertices) with NO prediction filters,
      so GT points/count are byte-stable no matter what prediction knobs change.
PRED= model labels -> connection_points(gt_classes, prediction min_vertices, vertex_conf mask).

Receipt (results/cp_eval/<split>_<cpver>.json): metrics (P/R/F1/Jaccard, TP/FP/FN, CP-count
error, coverage, median/P95 position error), per-part + per-family, and SHA-256 of config /
checkpoint / this script for reproducibility.

This is region-derived CP CONSISTENCY (GT auto-derived from human REGION labels), NOT a human
physical-CP benchmark. Named as such in the receipt.

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe cp_evaluate.py --split val
"""
import argparse, json, hashlib, os
import numpy as np
import scheffler_dataset as dataset
import diffusionnet, metrics, cp_openings, connector3d

NAME2LBL = {"CableEntry": int(connector3d.CABLE_ENTRY), "Contact": int(connector3d.CONTACT)}


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16] if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--config", default="cp_config.json")
    ap.add_argument("--out-dir", default="results/cp_eval")
    ap.add_argument("--allow-locked", action="store_true",
                    help="required to touch test_locked (BURNED for CP-final; guards accidental test-peeking)")
    a = ap.parse_args()
    if a.split == "test_locked" and not a.allow_locked:
        raise SystemExit("test_locked is BURNED for CP-final and is a diagnostic-touched set. "
                         "Pass --allow-locked to run it deliberately (see split_manifest.json).")
    cfg = json.load(open(a.config))
    ver = cfg["cp_def_version"]; d = cfg["definitions"][ver]
    gt_classes = tuple(NAME2LBL[c] for c in d["gt_classes"])
    gt_min = int(d["gt_min_vertices"]); match_mm = float(d["match_mm"]); dedupe = float(d.get("cross_class_dedupe_mm", 0.0))
    pp = cfg["prediction_postproc"]; pred_min = int(pp["min_vertices"]); vconf = float(pp["vertex_confidence_mask"])
    ckpt = cfg["checkpoint"]
    os.makedirs(a.out_dir, exist_ok=True)

    model, meta, _ = diffusionnet.load_checkpoint(ckpt, device="cuda")
    samples = dataset.load_split(cfg["corpus"], a.split, allow_locked=(a.split == "test_locked"), verify_hashes=False)

    CT = int(connector3d.CONTACT)
    TP = FP = FN = 0; TP3 = FP3 = FN3 = 0; per = []; poserr = []; fam = {}; gt_incomplete = []; empty = []
    for s in samples:
        pid = s["part_id"]; V = np.asarray(s["verts"], float); F = np.asarray(s["faces"], int)
        L = np.asarray(s["labels"])
        # GT: fixed definition, NO prediction filters
        gt = np.array([c["point"] for c in cp_openings.connection_points(
            V, F, L, min_v=gt_min, dedupe_mm=dedupe, classes=gt_classes)] or []).reshape(-1, 3)
        # coverage honesty: a part with 0 CableEntry GT but Contact labels present is
        # gt_incomplete (connections labelled Contact, not CableEntry) -> it is NOT scorable by
        # cp-v2 and must be reported, not silently treated as a neutral 0/0.
        if len(gt) == 0:
            has_contact = any(int(f.label) == CT for f in connector3d.build_fragments(V, F, L, min_vertices=gt_min))
            (gt_incomplete if has_contact else empty).append(pid)
        # PRED: model + prediction post-processing only
        plab, probs = diffusionnet.predict(model, meta, V, F, device="cuda",
                                           op_cache_dir="results/scheffler_semantic/operators", return_probs=True)
        pr = np.array([c["point"] for c in cp_openings.connection_points(
            V, F, np.asarray(plab), min_v=pred_min, probs=probs, vertex_conf=vconf,
            dedupe_mm=dedupe, classes=gt_classes)] or []).reshape(-1, 3)
        if len(gt) and len(pr):
            m, up, ug = metrics.match_predictions(pr, gt, match_mm)
            tp, fp, fn = len(m), len(up), len(ug)
            for pi, gi, dist in m:
                poserr.append(float(dist))
            # STRICT 3mm radius reported alongside 5mm (dense small parts: 5mm can be optimistic)
            m3, up3, ug3 = metrics.match_predictions(pr, gt, 3.0)
            TP3 += len(m3); FP3 += len(up3); FN3 += len(ug3)
        else:
            tp, fp, fn = 0, len(pr), len(gt)
            FP3 += len(pr); FN3 += len(gt)
        TP += tp; FP += fp; FN += fn
        per.append({"part_id": pid, "gt": len(gt), "pred": len(pr), "tp": tp, "fp": fp, "fn": fn,
                    "cp_count_error": len(pr) - len(gt)})
        fk = pid[:3]; fam.setdefault(fk, [0, 0, 0]); fam[fk][0] += tp; fam[fk][1] += fp; fam[fk][2] += fn

    f1 = 2 * TP / max(2 * TP + FP + FN, 1); jac = TP / max(TP + FP + FN, 1)
    prec = TP / max(TP + FP, 1); rec = TP / max(TP + FN, 1)
    cov = sum(1 for p in per if p["gt"] > 0 and p["tp"] > 0) / max(sum(1 for p in per if p["gt"] > 0), 1)
    receipt = {
        "metric_name": "region_derived_cp_consistency",
        "metric_caveat": cfg["metric_names"]["region_derived_cp_consistency"],
        "NOT": "not human_physical_cp; not untouched-held-out if split==test_locked (rules were diagnosed on it)",
        "split": a.split, "cp_def_version": ver, "n_parts": len(samples),
        "coverage_honesty": {
            "scored_parts_gt_positive": sum(1 for p in per if p["gt"] > 0),
            "gt_incomplete_excluded (0 CableEntry but Contact labelled)": len(gt_incomplete),
            "gt_incomplete_parts": gt_incomplete,
            "empty_no_connection_label": empty,
            "warning": "F1/P/R below are over the scored (CableEntry-labelled) parts only; "
                       "gt_incomplete parts have connections labelled Contact and are NOT scored by cp-v2"},
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "jaccard": round(jac, 4),
        "TP": TP, "FP": FP, "FN": FN,
        "strict_3mm": {"f1": round(2 * TP3 / max(2 * TP3 + FP3 + FN3, 1), 4),
                       "precision": round(TP3 / max(TP3 + FP3, 1), 4), "recall": round(TP3 / max(TP3 + FN3, 1), 4),
                       "TP": TP3, "FP": FP3, "FN": FN3,
                       "note": "match radius 3mm (stricter than the 5mm headline); gap 5mm->3mm shows position sloppiness"},
        "cp_count_mae": round(float(np.mean([abs(p["cp_count_error"]) for p in per])), 3),
        "coverage_parts_with_a_hit": round(cov, 4),
        "position_error_mm_median": round(float(np.median(poserr)), 3) if poserr else None,
        "position_error_mm_p95": round(float(np.percentile(poserr, 95)), 3) if poserr else None,
        "direction_error": "N/A - GT direction is auto-derived, not human; measured only vs human-CP benchmark",
        "config": {"gt_classes": d["gt_classes"], "gt_min_vertices": gt_min, "match_mm": match_mm,
                   "prediction_min_vertices": pred_min, "vertex_confidence_mask": vconf, "cross_class_dedupe_mm": dedupe},
        "sha": {"config": sha(a.config), "checkpoint": sha(ckpt), "evaluator": sha(__file__),
                "cp_openings": sha("cp_openings.py")},
        "per_family": {k: {"tp": v[0], "fp": v[1], "fn": v[2],
                           "f1": round(2 * v[0] / max(2 * v[0] + v[1] + v[2], 1), 3)} for k, v in sorted(fam.items())},
        "per_part": per,
    }
    out = os.path.join(a.out_dir, f"{a.split}_{ver}.json")
    json.dump(receipt, open(out, "w"), indent=1)
    print(f"[{a.split}] {ver}  P={prec:.3f} R={rec:.3f} F1={f1:.3f} Jac={jac:.3f}  "
          f"TP={TP} FP={FP} FN={FN}  cov={cov:.3f}  posMedian={receipt['position_error_mm_median']}mm")
    print(f"receipt -> {out}")


if __name__ == "__main__":
    main()
