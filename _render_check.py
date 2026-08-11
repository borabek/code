# -*- coding: utf-8 -*-
"""Headless render of a GLB (all Scene geometries, embedded vertex colours) to PNG via
matplotlib Agg, viewed down the thin bbox axis (the large face). Self-verify without a viewer."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

path, out = sys.argv[1], sys.argv[2]
scene = trimesh.load(path)
geos = scene.geometry.values() if isinstance(scene, trimesh.Scene) else [scene]
allV = np.vstack([g.vertices for g in geos])
thin = int(np.argmin(allV.max(0) - allV.min(0)))
if len(sys.argv) >= 5:  # explicit iso/angle override
    elev, azim = float(sys.argv[3]), float(sys.argv[4])
else:
    elev, azim = (90, -90) if thin == 2 else (0, -90) if thin == 1 else (0, 0)
fig = plt.figure(figsize=(7, 7)); ax = fig.add_subplot(111, projection="3d")
for g in geos:
    V, F = np.asarray(g.vertices), np.asarray(g.faces)
    vc = getattr(g.visual, "vertex_colors", None)
    if vc is None or len(vc) != len(V):
        fc = np.tile([0.66, 0.66, 0.66], (len(F), 1))
    else:
        fc = (vc[F][:, :, :3].mean(1) / 255.0)
    pc = Poly3DCollection(V[F], facecolors=fc, edgecolors="none", linewidths=0)
    ax.add_collection3d(pc)
ax.set_xlim(allV[:, 0].min(), allV[:, 0].max()); ax.set_ylim(allV[:, 1].min(), allV[:, 1].max())
ax.set_zlim(allV[:, 2].min(), allV[:, 2].max())
try:
    ax.set_box_aspect(allV.max(0) - allV.min(0))
except Exception:
    pass
ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
plt.tight_layout(); plt.savefig(out, dpi=95, bbox_inches="tight"); print("wrote", out)
