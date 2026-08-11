# -*- coding: utf-8 -*-
"""Comprehensive CP viz across the FULL part. CP rule (data-driven): use CableEntry
connected components; if the part has NO CableEntry, fall back to Contact components (the
18 CableEntry-absent parts have their connections labelled Contact -- verified on 2002-7214).
Each component -> one CP (centroid + outward direction). Markers = big MAGENTA SPIKES
(cylinder+cone) sticking out along the direction, so they show even if a viewer ignores
vertex colours. Writes <id>_human.glb (human labels) and <id>_MODEL.glb (model prediction).

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe viz_cp_comprehensive.py --split val
"""
import argparse, os
import numpy as np
import trimesh
import scheffler_dataset as dataset
import diffusionnet, connector3d

CKPT = "results/scheffler_semantic/refit91.pt"; OPCACHE = "results/scheffler_semantic/operators"
CABLE = int(connector3d.CABLE_ENTRY); CONTACT = int(connector3d.CONTACT)
GREY = (165, 165, 165, 255); MAG = (235, 30, 220, 255)


def comprehensive_cps(V, F, labels, min_v=10):
    """CableEntry components, else Contact components. Returns [(point, direction), ...]
    and the source class used."""
    frags = connector3d.build_fragments(V, F, labels, min_vertices=min_v)
    ce = [f for f in frags if int(f.label) == CABLE]
    src = CABLE if ce else CONTACT
    use = ce if ce else [f for f in frags if int(f.label) == CONTACT]
    bc = V.mean(0); out = []
    for f in use:
        cp = connector3d.ConnectionPoint([f])
        try:
            cp.compute_direction(V, F, smooth_subdiv=0, body_center=bc)
            dv = np.asarray(cp.approach_vector, float)
        except Exception:
            dv = np.asarray(cp.entry_point, float) - bc
        out.append((np.asarray(cp.entry_point, float), dv))
    return out, src


def spike(pt, dv, r, h):
    d = np.asarray(dv, float); n = np.linalg.norm(d)
    d = d / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])
    shaft = trimesh.creation.cylinder(radius=r, height=h, sections=16)
    tip = trimesh.creation.cone(radius=r * 2.2, height=h * 0.5, sections=16)
    tip.apply_translation([0, 0, h / 2 + h * 0.25])
    arrow = trimesh.util.concatenate([shaft, tip])
    arrow.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], d))
    arrow.apply_translation(pt + d * (h * 0.3))
    arrow.visual.vertex_colors = np.tile(MAG, (len(arrow.vertices), 1)).astype(np.uint8)
    return arrow


def build(V, F, cps, out):
    diag = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0
    r = diag * 0.012; h = diag * 0.16
    mesh = trimesh.Trimesh(vertices=V, faces=F,
                           vertex_colors=np.tile(GREY, (len(V), 1)).astype(np.uint8), process=False)
    parts = [mesh] + [spike(p, d, r, h) for p, d in cps]
    trimesh.util.concatenate(parts).export(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="_scheffler_cp_glb")
    ap.add_argument("--min-v", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model, meta, _ = diffusionnet.load_checkpoint(CKPT, device="cuda")
    samples = dataset.load_split("wscad_corpus_scheffler_exact", a.split,
                                 allow_locked=(a.split == "test_locked"), verify_hashes=False)
    cls = {CABLE: "CableEntry", CONTACT: "Contact"}
    n = 0
    for s in samples:
        pid = s["part_id"]; V = np.asarray(s["verts"], float); F = np.asarray(s["faces"], int)
        h_cps, h_src = comprehensive_cps(V, F, np.asarray(s["labels"]), a.min_v)
        build(V, F, h_cps, os.path.join(a.out, f"{pid}_human.glb"))
        plab = np.asarray(diffusionnet.predict(model, meta, V, F, device="cuda", op_cache_dir=OPCACHE))
        m_cps, m_src = comprehensive_cps(V, F, plab, a.min_v)
        build(V, F, m_cps, os.path.join(a.out, f"{pid}_MODEL.glb"))
        print(f"  {pid}: human {len(h_cps)}CP({cls.get(h_src,'-')}) model {len(m_cps)}CP({cls.get(m_src,'-')})")
        n += 1
        if a.limit and n >= a.limit:
            break
    print(f"\nwrote {n} parts (comprehensive) -> {a.out}/  MAGENTA spikes = CP points")
    print("DONE")


if __name__ == "__main__":
    main()
