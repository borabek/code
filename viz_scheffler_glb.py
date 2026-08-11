# -*- coding: utf-8 -*-
"""CP review viz -- GLB per part, SELF-VERIFIED via PNG. The mesh is painted plain GREY
except CableEntry vertices = BRIGHT GREEN (isolating the CP regions so they pop -- a full
per-class colouring buried the small green under blue/orange). A MAGENTA sphere marks each
derived CP entry point. GLB embeds per-vertex colours + the sphere geometry, so Windows 3D
Viewer shows it reliably (OBJ+.mtl did not). Writes <part>_human.glb (human CableEntry +
human-derived CPs = the GT to review) and <part>_MODEL.glb (model prediction).

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe viz_scheffler_glb.py --split val
"""
import argparse, json, os
import numpy as np
import trimesh
import scheffler_dataset as dataset
import diffusionnet, connector3d, metrics

CKPT = "results/scheffler_semantic/refit91.pt"; OPCACHE = "results/scheffler_semantic/operators"
CABLE = int(connector3d.CABLE_ENTRY)
GREY = (170, 170, 170, 255); GREEN = (25, 225, 55, 255); MAGENTA = (230, 40, 220, 255)


def cps_from_labels(V, F, labels, min_v):
    frags = connector3d.build_fragments(V, F, labels, min_vertices=min_v)
    bc = V.mean(0); pts = []
    for f in frags:
        if int(f.label) != CABLE:
            continue
        cp = connector3d.ConnectionPoint([f])
        try:
            cp.compute_direction(V, F, smooth_subdiv=0, body_center=bc)
        except Exception:
            pass
        pts.append(np.asarray(cp.entry_point, float))
    return np.array(pts, float) if pts else np.zeros((0, 3))


def build_glb(V, F, labels, cps, out):
    labels = np.asarray(labels)
    green = labels == CABLE
    # dilate the green region by one face-ring so a small CableEntry patch stays visible
    fmask = green[F].any(1)
    green[np.unique(F[fmask])] = True
    cols = np.where(green[:, None], np.array(GREEN, np.uint8), np.array(GREY, np.uint8)).astype(np.uint8)
    mesh = trimesh.Trimesh(vertices=V, faces=F, vertex_colors=cols, process=False)
    meshes = [mesh]
    r = (float(np.linalg.norm(V.max(0) - V.min(0))) * 0.035) or 1.0   # big, unmistakable CP marker
    for p in cps:
        s = trimesh.creation.icosphere(subdivisions=2, radius=r)
        s.apply_translation(p)
        s.visual.vertex_colors = np.tile(MAGENTA, (len(s.vertices), 1)).astype(np.uint8)
        meshes.append(s)
    trimesh.util.concatenate(meshes).export(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="_scheffler_cp_glb")
    ap.add_argument("--min-v", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cands = {p["part_id"]: p for p in json.load(open(
        "results/scheffler_cp_bridge/development_candidates.json", encoding="utf-8"))["parts"]}
    model, meta, _ = diffusionnet.load_checkpoint(CKPT, device="cuda")
    samples = dataset.load_split("wscad_corpus_scheffler_exact", a.split,
                                 allow_locked=(a.split == "test_locked"), verify_hashes=False)
    n = 0
    for s in samples:
        pid = s["part_id"]; V = np.asarray(s["verts"], float); F = np.asarray(s["faces"], int)
        hlab = np.asarray(s["labels"])
        # human CPs from the frozen candidate file (fall back to on-the-fly)
        cp_entry = cands.get(pid, {}).get("candidates") or []
        h_cps = (np.array([c["obj_frame"]["entry_point"] for c in cp_entry], float)
                 if cp_entry else cps_from_labels(V, F, hlab, 1))
        build_glb(V, F, hlab, h_cps, os.path.join(a.out, f"{pid}_human.glb"))
        # model prediction + cleaned CPs
        plab = np.asarray(diffusionnet.predict(model, meta, V, F, device="cuda", op_cache_dir=OPCACHE))
        m_cps = cps_from_labels(V, F, plab, a.min_v)
        build_glb(V, F, plab, m_cps, os.path.join(a.out, f"{pid}_MODEL.glb"))
        print(f"  {pid}: human CP={len(h_cps)} model CP={len(m_cps)} (CableEntry v: human {int((hlab==CABLE).sum())}, model {int((plab==CABLE).sum())})")
        n += 1
        if a.limit and n >= a.limit:
            break
    print(f"\nwrote {n} parts -> {a.out}/  GREEN=CableEntry, MAGENTA sphere=CP point")
    print("Open <part>_human.glb (GT to review) and <part>_MODEL.glb (model) in Windows 3D Viewer.")
    print("DONE")


if __name__ == "__main__":
    main()
