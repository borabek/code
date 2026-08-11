# -*- coding: utf-8 -*-
"""BIAS-SAFE labelling aid for the 22-part queue (_label_targets/).

The model is WRONG on exactly these parts (that is why they are queued), so pre-filling the label
template with the model's prediction would bias the annotator toward the model's ERRORS (the project
has been burned by fake/automation-biased confirmation before). So this does NOT touch the neutral
all-Housing template. It only renders a 3-view VISUAL reference of the model's current guess, clearly
marked "VERIFY — model unreliable here", so the annotator can ORIENT (see the part, see where the
model thinks Contact/CableEntry are) and knows WHERE to look hardest -- without any copy-pasteable
pre-labels. Run on CPU so it never competes with a training run for the 4GB GPU.

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe label_assist.py [--device cpu]
"""
import os, glob, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import diffusionnet as D
import connector3d
from infer_step_cp import load_any

QDIR = "_label_targets"; OP = "results/step_infer/ops"
CKPT = "results/scheffler_semantic/refit91.pt"
SEM = {int(connector3d.HOUSING): (0.72, 0.72, 0.72), int(connector3d.CONTACT): (0.15, 0.45, 0.95),
       int(connector3d.SNAP_POINT): (0.95, 0.85, 0.15), int(connector3d.CABLE_ENTRY): (0.1, 0.8, 0.25),
       int(connector3d.LABEL_SURFACE): (0.95, 0.6, 0.15)}
NAMES = {int(getattr(connector3d, n)): n for n in ("HOUSING", "CONTACT", "SNAP_POINT", "CABLE_ENTRY", "LABEL_SURFACE")}


def load_obj(p):
    V, F = [], []
    for ln in open(p):
        if ln.startswith("v "):
            V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("f "):
            F.append([int(t.split("/")[0]) - 1 for t in ln.split()[1:4]])
    return np.asarray(V, float), np.asarray(F, int)


def render(V, F, plab, out, pid):
    thin = int(np.argmin(V.max(0) - V.min(0)))
    views = {0: [(0, 0), (0, -90)], 1: [(0, -90), (90, -90)], 2: [(90, -90), (0, 0)]}[thin]
    fig = plt.figure(figsize=(13, 6.2)); ctr = V.mean(0); rng = (V.max(0) - V.min(0)).max()
    fc_all = np.array([SEM.get(int(x), (0.6, 0.6, 0.6)) for x in plab])[F].mean(1)
    for k, (ev, az) in enumerate(views, 1):
        ax = fig.add_subplot(1, 2, k, projection="3d")
        ax.add_collection3d(Poly3DCollection(V[F], facecolors=fc_all, edgecolors="none"))
        for st, lo, hi in [(ax.set_xlim, ctr[0]-rng/2, ctr[0]+rng/2), (ax.set_ylim, ctr[1]-rng/2, ctr[1]+rng/2),
                           (ax.set_zlim, ctr[2]-rng/2, ctr[2]+rng/2)]:
            st(lo, hi)
        try: ax.set_box_aspect(V.max(0) - V.min(0))
        except Exception: pass
        ax.view_init(elev=ev, azim=az); ax.set_axis_off()
    cc = {NAMES[c]: int((plab == c).sum()) for c in SEM}
    fig.suptitle(f"{pid}  MODEL GUESS -- VERIFY, model is UNRELIABLE on this part (paint deliberately)\n"
                 f"grey=Housing blue=Contact yellow=SnapPoint green=CableEntry orange=LabelSurface   {cc}",
                 fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.93]); plt.savefig(out, dpi=90, bbox_inches="tight"); plt.close()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cpu"); a = ap.parse_args()
    model, meta, _ = load_any(CKPT, dev=a.device)
    parts = sorted(d for d in glob.glob(os.path.join(QDIR, "*")) if os.path.isdir(d))
    done = 0
    for d in parts:
        pid = os.path.basename(d)
        obj = os.path.join(d, f"{pid}.obj")
        if not os.path.exists(obj):
            continue
        V, F = load_obj(obj)
        V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
        plab = np.asarray(D.predict(model, meta, V, F, device=a.device, op_cache_dir=OP))
        out = os.path.join(d, f"{pid}.MODELGUESS.png")
        render(V, F, plab, out, pid)
        ce = int((plab == int(connector3d.CABLE_ENTRY)).sum())
        print(f"  {pid}: CableEntry_verts={ce:5d} -> {out}", flush=True)
        done += 1
    print(f"DONE {done} reference renders (neutral template UNCHANGED -- no automation bias).")


if __name__ == "__main__":
    main()
