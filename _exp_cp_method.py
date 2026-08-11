# -*- coding: utf-8 -*-
"""EXPERIMENT (not production): compare CP derivation methods on one part, render each with
semantic-coloured mesh + balls at the derived openings, so render->Read shows which method
lands balls ON the visible physical openings. Methods:
  A = CableEntry-only openings (current viz)
  B = Contact-only openings
  C = Contact + CableEntry, deduped per terminal (merge openings within `merge_mm`)
Ball = green sphere offset outward along the thesis bbox-normal.
"""
import sys, numpy as np, trimesh
import scheffler_dataset as dataset, connector3d
CE = int(connector3d.CABLE_ENTRY); CT = int(connector3d.CONTACT)
COL = {int(connector3d.HOUSING): (180,180,180), CT: (55,110,235),
       int(connector3d.SNAP_POINT): (240,215,40), CE: (25,225,55), int(connector3d.LABEL_SURFACE): (235,140,40)}
pid = sys.argv[1] if len(sys.argv) > 1 else "3011041"
split = sys.argv[2] if len(sys.argv) > 2 else "val"

s = {x["part_id"]: x for x in dataset.load_split("wscad_corpus_scheffler_exact", split, verify_hashes=False)}[pid]
V = np.asarray(s["verts"], float); F = np.asarray(s["faces"], int); L = np.asarray(s["labels"])
bc = V.mean(0)

def openings(classes, min_v=10):
    frags = connector3d.build_fragments(V, F, L, min_vertices=min_v)
    out = []
    for f in frags:
        if int(f.label) not in classes: continue
        cp = connector3d.ConnectionPoint([f])
        try:
            cp.compute_direction(V, F, smooth_subdiv=0, body_center=bc); d = np.asarray(cp.approach_vector, float)
        except Exception:
            d = np.asarray(cp.entry_point, float) - bc
        out.append((np.asarray(cp.entry_point, float), d, int(f.label)))
    return out

def dedupe(cps, merge_mm=8.0):
    kept = []
    for p, d, lb in cps:
        if any(np.linalg.norm(p - q[0]) < merge_mm for q in kept): continue
        kept.append((p, d, lb))
    return kept

def rgba(labels):
    a = np.array([COL.get(int(x),(150,150,150)) for x in labels], np.uint8)
    return np.concatenate([a, np.full((len(a),1),255,np.uint8)],1)

def render(cps, tag):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    thin = int(np.argmin(V.max(0)-V.min(0)))
    elev, azim = (18, -60)  # iso so protruding balls show
    fig = plt.figure(figsize=(7,7)); ax = fig.add_subplot(111, projection="3d")
    fc = rgba(L)[F][:,:,:3].mean(1)/255.0
    ax.add_collection3d(Poly3DCollection(V[F], facecolors=fc, edgecolors="none"))
    diag = float(np.linalg.norm(V.max(0)-V.min(0)))
    for p, d, lb in cps:
        n = np.linalg.norm(d); d = d/n if n>1e-6 else (p-bc)/ (np.linalg.norm(p-bc)+1e-9)
        q = p + d*diag*0.05
        ax.scatter([q[0]],[q[1]],[q[2]], c="magenta", s=260, edgecolors="k", depthshade=False)
    for setter,a,b in [(ax.set_xlim,V[:,0].min(),V[:,0].max()),(ax.set_ylim,V[:,1].min(),V[:,1].max()),(ax.set_zlim,V[:,2].min(),V[:,2].max())]:
        setter(a,b)
    try: ax.set_box_aspect(V.max(0)-V.min(0))
    except Exception: pass
    ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
    out = f"_exp_{pid}_{tag}.png"; plt.tight_layout(); plt.savefig(out, dpi=90, bbox_inches="tight"); plt.close()
    print(f"  {tag}: {len(cps)} balls -> {out}")

A = openings({CE}); B = openings({CT}); C = dedupe(openings({CE, CT}))
print(f"{pid}: CableEntry-only={len(A)}  Contact-only={len(B)}  Contact+CableEntry-deduped={len(C)}")
render(A, "A_cableentry"); render(B, "B_contact"); render(C, "C_both_deduped")
print("DONE")
