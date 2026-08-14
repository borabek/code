# -*- coding: utf-8 -*-
"""Render a GLB (Scene: grey mesh + green CP balls) from 4 fixed angles into ONE PNG, so CP
markers can be verified WITHOUT depending on the user's 3D viewer / its camera. Angles: front,
side, top, iso. Every ball is drawn opaque on top, so none hide behind the slab.

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe contact_sheet.py <glb> [out.png]
"""
import sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

path = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else path.rsplit(".", 1)[0] + "_SHEET.png"
scene = trimesh.load(path)
geos = list(scene.geometry.values()) if isinstance(scene, trimesh.Scene) else [scene]
# separate mesh (grey, many verts) from CP balls (green ~162 verts)
mesh = max(geos, key=lambda g: len(g.vertices))
balls = [g for g in geos if g is not mesh]
V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
allV = np.vstack([V] + [np.asarray(b.vertices) for b in balls]) if balls else V
ctr = allV.mean(0); rng = (allV.max(0) - allV.min(0)).max()
views = [("front", 12, -72), ("side", 12, 18), ("top", 82, -90), ("iso", 26, -52)]
fig = plt.figure(figsize=(12, 12))
for i, (name, elev, azim) in enumerate(views, 1):
    ax = fig.add_subplot(2, 2, i, projection="3d")
    ax.add_collection3d(Poly3DCollection(V[F], facecolors=[[0.66, 0.66, 0.66]] * len(F), edgecolors="none"))
    for b in balls:
        c = np.asarray(b.vertices).mean(0)
        ax.scatter([c[0]], [c[1]], [c[2]], c="lime", s=220, edgecolors="k", linewidths=0.8, depthshade=False)
    for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        pass
    ax.set_xlim(ctr[0]-rng/2, ctr[0]+rng/2); ax.set_ylim(ctr[1]-rng/2, ctr[1]+rng/2); ax.set_zlim(ctr[2]-rng/2, ctr[2]+rng/2)
    ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
    ax.set_title(f"{name}  ({len(balls)} CP)", color="black")
plt.tight_layout(); plt.savefig(out, dpi=85, bbox_inches="tight"); plt.close()
print(f"wrote {out}  ({len(balls)} CP balls)")
