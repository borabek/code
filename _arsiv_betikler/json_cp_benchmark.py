# -*- coding: utf-8 -*-
"""IN-DOMAIN manufacturer-CP benchmark from the JSON corpus (Desktop\\JSON): 479 watertight meshes
each carrying real manufacturer ConnectionPoints (Point + InsertDirection). THIS is the resource
that makes metric #3 (human/physical CP with point+direction) measurable -- it had been sidelined.

What this gives:
  - dataset accessor: JSON -> (V, F, cp_points, cp_dirs, part_id, sha),
  - a leakage-safe, SHA-deduped, deterministic train/val/holdout split (json_cp_split.json),
  - a scorer at (point <= 5 mm AND direction <= 15 deg) -> precision/recall/F1,
  - a trivial floor baseline to exercise the harness.

Honesty: this is IN-DOMAIN for the JSON families (A-B etc.), a CAPABILITY benchmark. It is NOT the
WSCAD-STEP terminal-block target (different families + the JSON->STEP domain gap). A learned
detector is known feasible in-domain (M0 gate overfit F1~1.0); a clean held-out number needs
training and is left as the next step. Manufacturer CP points sit ~4.7 mm off-surface (a standoff),
so a fair scorer/target must use the manufacturer point, not an on-surface opening midpoint.

Usage: .venv/Scripts/python.exe json_cp_benchmark.py            # build split + run floor baseline
"""
import json, glob, os, hashlib
import numpy as np

SRC = r"C:/Users/DE00024082/Desktop/JSON"
SPLIT_FILE = "json_cp_split.json"


def load_part(path):
    d = json.load(open(path, encoding="utf-8"))
    g = d.get("Graphic3d", {})
    V = np.array([[p["X"], p["Y"], p["Z"]] for p in g.get("Points", [])], float)
    idx = g.get("Indices", [])
    F = np.array(idx, int).reshape(-1, 3) if idx else np.zeros((0, 3), int)
    cps = d.get("ConnectionPoints", [])
    P = np.array([[c["Point"]["X"], c["Point"]["Y"], c["Point"]["Z"]] for c in cps], float).reshape(-1, 3)
    D = np.array([[c["InsertDirection"]["X"], c["InsertDirection"]["Y"], c["InsertDirection"]["Z"]]
                 for c in cps], float).reshape(-1, 3)
    Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
    pid = d.get("PartNr", os.path.splitext(os.path.basename(path))[0])
    return {"part_id": pid, "path": path, "V": V, "F": F, "cp_points": P, "cp_dirs": Dn,
            "sha": hashlib.sha256(open(path, "rb").read()).hexdigest()}


def build_split():
    parts = []
    seen = set()
    for f in sorted(glob.glob(os.path.join(SRC, "*.json"))):
        p = load_part(f)
        if p["sha"] in seen:
            continue
        seen.add(p["sha"])
        # deterministic bucket from sha -> 70 train / 15 val / 15 holdout
        h = int(p["sha"][:8], 16) % 100
        role = "train" if h < 70 else ("val" if h < 85 else "holdout")
        parts.append({"part_id": p["part_id"], "path": os.path.basename(f), "sha": p["sha"],
                      "n_cp": int(len(p["cp_points"])), "n_verts": int(len(p["V"])), "role": role})
    split = {"frozen": "2026-07-18", "source": "manufacturer JSON (Desktop/JSON)",
             "n_parts_deduped": len(parts),
             "counts": {r: sum(1 for x in parts if x["role"] == r) for r in ("train", "val", "holdout")},
             "total_cps": sum(x["n_cp"] for x in parts),
             "match": {"point_mm": 5.0, "direction_deg": 15.0},
             "parts": parts}
    json.dump(split, open(SPLIT_FILE, "w"), indent=1)
    return split


def load_split(role):
    sp = json.load(open(SPLIT_FILE))
    return [load_part(os.path.join(SRC, x["path"])) for x in sp["parts"] if x["role"] == role]


def score(pred_pts, pred_dirs, gt_pts, gt_dirs, point_mm=5.0, dir_deg=15.0):
    """TP = matched within point_mm AND direction within dir_deg."""
    from scipy.optimize import linear_sum_assignment
    pred_pts = np.asarray(pred_pts, float).reshape(-1, 3); gt_pts = np.asarray(gt_pts, float).reshape(-1, 3)
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return 0, len(pred_pts), len(gt_pts)
    C = np.linalg.norm(pred_pts[:, None] - gt_pts[None], axis=2)
    ri, ci = linear_sum_assignment(C)
    tp = 0
    for i, j in zip(ri, ci):
        if C[i, j] <= point_mm:
            cang = float(np.clip(np.dot(pred_dirs[i], gt_dirs[j]), -1, 1))
            if np.degrees(np.arccos(cang)) <= dir_deg:
                tp += 1
    return tp, len(pred_pts) - tp, len(gt_pts) - tp


def main():
    sp = build_split()
    print(f"split: {sp['counts']} (deduped {sp['n_parts_deduped']}), total CPs {sp['total_cps']} -> {SPLIT_FILE}")
    # floor baseline on holdout: predict the single body centroid (exercises the scorer)
    TP = FP = FN = 0
    for p in load_split("holdout"):
        c = p["V"].mean(0)[None] if len(p["V"]) else np.zeros((1, 3))
        tp, fp, fn = score(c, np.array([[0, 1, 0]]), p["cp_points"], p["cp_dirs"])
        TP += tp; FP += fp; FN += fn
    f1 = 2 * TP / max(2 * TP + FP + FN, 1)
    print(f"FLOOR baseline (centroid) holdout: F1={f1:.3f} TP={TP} FP={FP} FN={FN}  "
          f"(expected ~0; proves harness + scorer run; a learned detector is the next step)")


if __name__ == "__main__":
    main()
