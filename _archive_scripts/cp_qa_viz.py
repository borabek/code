# -*- coding: utf-8 -*-
"""CP quality-assurance visualisation with a FIXED colour standard (audit P0 + P2). No more
ambiguous single-colour pictures: every QA image says exactly what each marker is.

COLOUR STANDARD (frozen):
  GREEN  = GT CP (human-region-derived, CableEntry = cp-v2 metric)
  BLUE   = model CP that MATCHES a GT (True Positive)
  RED    = model CP with no GT within match_mm (False Positive)
  ORANGE = GT CP the model missed (False Negative)

Per part it writes (into _cp_qa/): a numbered 2-view PNG (primary, reliable) titled with
TP/FP/FN, plus three GLBs (gt_only / model_only / overlay) for 3D. Balls are modest so the
mesh + openings stay visible.

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe cp_qa_viz.py --split val
"""
import argparse, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
import scheffler_dataset as dataset
import diffusionnet, metrics, cp_openings, connector3d, viz_semantic_full

CE = int(connector3d.CABLE_ENTRY)
OP = "results/scheffler_semantic/operators"; CKPT = "results/scheffler_semantic/refit91.pt"
GREEN = (30, 200, 60); BLUE = (40, 120, 245); RED = (235, 40, 40); ORANGE = (245, 150, 20)
MPL = {"GT (green)": "limegreen", "model TP (blue)": "dodgerblue", "FP (red)": "red", "FN (orange)": "orange"}


def classify(gt, pr, match_mm):
    """Return lists of (point, kind) with kind in gt/tp/fp/fn."""
    out = []
    if len(gt) and len(pr):
        m, up, ug = metrics.match_predictions(pr, gt, match_mm)
        matched_pred = {i for i, _, _ in m}; matched_gt = {j for _, j, _ in m}
        for i, _, _ in m:
            out.append((pr[i], "tp"))
        for i in range(len(pr)):
            if i not in matched_pred:
                out.append((pr[i], "fp"))
        for j in range(len(gt)):
            if j not in matched_gt:
                out.append((gt[j], "fn"))
    else:
        out += [(p, "fp") for p in pr] + [(g, "fn") for g in gt]
    return out


def png(V, F, marks, out, title):
    thin = int(np.argmin(V.max(0) - V.min(0)))
    angs = {0: [(0, 0), (22, -40)], 1: [(0, -90), (22, -62)], 2: [(90, -90), (58, -62)]}[thin]
    col = {"tp": "dodgerblue", "fp": "red", "fn": "orange", "gt": "limegreen"}
    fig = plt.figure(figsize=(13, 6.5)); ctr = V.mean(0); rng = (V.max(0) - V.min(0)).max()
    for k, (el, az) in enumerate(angs, 1):
        ax = fig.add_subplot(1, 2, k, projection="3d")
        ax.add_collection3d(Poly3DCollection(V[F], facecolors=(0.62, 0.62, 0.62, 0.22), edgecolors="none"))
        for j, (p, kind) in enumerate(marks, 1):
            ax.scatter([p[0]], [p[1]], [p[2]], c=col[kind], s=240, edgecolors="k", linewidths=1.2, depthshade=False, zorder=10)
        ax.set_xlim(ctr[0]-rng/2, ctr[0]+rng/2); ax.set_ylim(ctr[1]-rng/2, ctr[1]+rng/2); ax.set_zlim(ctr[2]-rng/2, ctr[2]+rng/2)
        ax.view_init(elev=el, azim=az); ax.set_axis_off(); ax.set_title(title if k == 1 else "green=GT blue=TP red=FP orange=FN")
    plt.tight_layout(); plt.savefig(out, dpi=90, bbox_inches="tight"); plt.close()


def glb(V, F, pts_cols, out):
    R, ctr = viz_semantic_full.reorient(V); Vt = (V - ctr) @ R.T
    sc = trimesh.Scene()
    sc.add_geometry(trimesh.Trimesh(Vt, F, vertex_colors=np.tile((165, 165, 165, 255), (len(Vt), 1)).astype(np.uint8), process=False), node_name="mesh")
    diag = float(np.linalg.norm(Vt.max(0) - Vt.min(0))) or 1.0; r = diag * 0.02
    for i, (p, rgb) in enumerate(pts_cols):
        pt = (np.asarray(p, float) - ctr) @ R.T
        s = trimesh.creation.icosphere(subdivisions=3, radius=r); s.apply_translation(pt)
        s.visual.vertex_colors = np.tile(tuple(rgb) + (255,), (len(s.vertices), 1)).astype(np.uint8)
        sc.add_geometry(s, node_name=f"m{i}")
    sc.export(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val"); ap.add_argument("--out", default="_cp_qa")
    ap.add_argument("--match", type=float, default=5.0); ap.add_argument("--only", default="")
    ap.add_argument("--vertex-conf", type=float, default=0.9); ap.add_argument("--min-v", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model, meta, _ = diffusionnet.load_checkpoint(CKPT, device="cuda")
    samples = dataset.load_split("wscad_corpus_scheffler_exact", a.split, allow_locked=(a.split == "test_locked"), verify_hashes=False)
    rows = []
    for s in samples:
        pid = s["part_id"]
        if a.only and pid != a.only:
            continue
        V = np.asarray(s["verts"], float); F = np.asarray(s["faces"], int)
        gt = np.array([c["point"] for c in cp_openings.connection_points(V, F, np.asarray(s["labels"]), min_v=1, classes=(CE,))] or []).reshape(-1, 3)
        plab, probs = diffusionnet.predict(model, meta, V, F, device="cuda", op_cache_dir=OP, return_probs=True)
        pr = np.array([c["point"] for c in cp_openings.connection_points(V, F, np.asarray(plab), min_v=a.min_v, probs=probs, vertex_conf=a.vertex_conf, classes=(CE,))] or []).reshape(-1, 3)
        marks = classify(gt, pr, a.match)
        tp = sum(1 for _, k in marks if k == "tp"); fp = sum(1 for _, k in marks if k == "fp"); fn = sum(1 for _, k in marks if k == "fn")
        png(V, F, marks, os.path.join(a.out, f"{pid}_QA.png"), f"{pid}: TP={tp} FP={fp} FN={fn} (GT={len(gt)} model={len(pr)})")
        glb(V, F, [(g, GREEN) for g in gt], os.path.join(a.out, f"{pid}_GTonly.glb"))
        glb(V, F, [(p, BLUE) for p in pr], os.path.join(a.out, f"{pid}_MODELonly.glb"))
        cols = {"tp": BLUE, "fp": RED, "fn": ORANGE}
        glb(V, F, [(p, cols[k]) for p, k in marks], os.path.join(a.out, f"{pid}_OVERLAY.glb"))
        rows.append((pid, len(gt), len(pr), tp, fp, fn))
        print(f"  {pid}: GT={len(gt)} model={len(pr)}  TP={tp} FP={fp} FN={fn}")
    print(f"\n{len(rows)} parts -> {a.out}/  (colour standard: GREEN=GT, BLUE=TP, RED=FP, ORANGE=FN)")


if __name__ == "__main__":
    main()
