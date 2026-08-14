# -*- coding: utf-8 -*-
"""ROBOT DEMO: throw a STEP file in, the trained model finds its CPs. Exactly the product use case.

Renders each part with:
  GREEN spheres  = manufacturer's ConnectionPoints (the truth, for reference)
  RED arrows     = what the MODEL (product recall_s2) outputs: CP point + insertion direction
So you can SEE what the robot actually produces on parts it has never trained on.

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe demo_robot.py <pid1> <pid2> ...
"""
import os, sys, glob, json
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diffusionnet as D, connector3d, cp_openings, thesis_remesh
from cad_eval import align_frames
from infer_step_cp import step_to_mesh, load_any

CE = int(connector3d.CABLE_ENTRY); CT = int(connector3d.CONTACT); OP = "results/step_infer/ops"
dev = "cuda" if torch.cuda.is_available() else "cpu"
STEP = {os.path.basename(s).split("_")[1]: s for s in glob.glob("all_wscad_stp/*.stp")}


def main():
    pids = sys.argv[1:]
    model, meta, _ = load_any("results/seg_extra/recall_s2.pt", dev=dev)
    n = len(pids); cols = 5; rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(4.2 * cols, 4.6 * rows))
    for k, pid in enumerate(pids):
        if pid not in STEP:
            continue
        Vr, Fr = step_to_mesh(STEP[pid])
        V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
        V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
        _, probs = D.predict(model, meta, V, F, device=dev, op_cache_dir=OP, return_probs=True)
        probs = np.asarray(probs, float); lab = probs.argmax(-1)
        # PRODUCT post-proc (min_v 30, vc 0.5)
        cps = cp_openings.connection_points(V, F, lab, min_v=30, classes=(CE, CT), dedupe_mm=10.0,
                                            probs=probs, vertex_conf=0.5, ct_depth_min_mm=1.0, cluster_mm=5.0)
        P = np.array([np.asarray(c["point"]) for c in cps], float) if cps else np.zeros((0, 3))
        Pd = np.array([np.asarray(c["direction"]) for c in cps], float) if cps else np.zeros((0, 3))
        conf = [float(c.get("confidence", 0)) for c in cps]
        # manufacturer CPs for reference
        G = np.zeros((0, 3))
        jf = [f for f in glob.glob("_ds1/DataSet/*.json") if os.path.basename(f).split("_")[0].endswith("." + pid)]
        mfg = "?"
        if jf:
            j = json.load(open(jf[0], encoding="utf-8-sig")); mfg = os.path.basename(jf[0]).split(".")[0]
            Gj = np.array([[c["Point"]["X"], c["Point"]["Y"], c["Point"]["Z"]] for c in j.get("ConnectionPoints", [])], float)
            Vj = np.array([[p["X"], p["Y"], p["Z"]] for p in j["Graphic3d"]["Points"]], float)
            if len(Gj):
                R, t, _ = align_frames(Vr, Vj); G = (Gj - t) @ R  # into mesh frame

        tri = V[F]; nrm = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)
        lt = np.array([.4, .3, 1.]); lt = lt/np.linalg.norm(lt)
        col = np.stack([0.55 + 0.4*np.abs(nrm @ lt)]*3, 1)
        conn = np.isin(lab, (CE, CT)); col[conn[F].any(1)] = [0.30, 0.55, 0.95]

        ax = fig.add_subplot(rows, cols, k+1, projection="3d")
        ax.add_collection3d(Poly3DCollection(tri, facecolors=np.clip(col, 0, 1), edgecolors="none", alpha=0.9))
        ext = V.max(0)-V.min(0); L = ext.max()*0.22; c0 = V.mean(0); rr = ext.max()*0.55
        if len(G):
            ax.scatter(G[:, 0], G[:, 1], G[:, 2], s=90, c="lime", marker="o",
                       depthshade=False, edgecolors="black", linewidths=1, zorder=5)
        for i in range(len(P)):
            p = P[i]; d = Pd[i]/(np.linalg.norm(Pd[i])+1e-9)*L
            ax.quiver(p[0], p[1], p[2], d[0], d[1], d[2], color="red", linewidth=2.4,
                      arrow_length_ratio=0.35, zorder=10)
            ax.scatter(*p, s=40, c="red", depthshade=False, zorder=10)
        ax.set_xlim(c0[0]-rr, c0[0]+rr); ax.set_ylim(c0[1]-rr, c0[1]+rr); ax.set_zlim(c0[2]-rr, c0[2]+rr)
        try: ax.set_box_aspect(ext)
        except Exception: pass
        thin = int(np.argmin(ext)); ev, az = {0: (0, 0), 1: (0, -90), 2: (90, -90)}[thin]
        ax.view_init(elev=ev, azim=az); ax.set_axis_off()
        cbar = f" | guven {min(conf):.2f}-{max(conf):.2f}" if conf else ""
        ax.set_title(f"{pid} ({mfg})  uretici {len(G)} / model {len(P)}{cbar}", fontsize=9)
        print(f"  {pid} ({mfg}): uretici {len(G)} CP, model {len(P)} CP bulundu", flush=True)
    fig.suptitle("ROBOT DEMO: STEP -> model CP'leri (kirmizi ok=model nokta+yon, yesil=uretici gercegi)  --  hic egitilmemis parcalar",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("results/_robot_demo.png", dpi=95, bbox_inches="tight"); print("-> results/_robot_demo.png")


if __name__ == "__main__":
    main()
