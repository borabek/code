# -*- coding: utf-8 -*-
"""Full-semantic + CP-review GLB. Two products per part:
  <id>_SEMANTIC_human/model.glb : all 5 classes coloured (Housing grey, Contact blue,
     SnapPoint yellow, CableEntry green, LabelSurface orange) -- shows the regions.
  <id>_<state>_human/model.glb  : CP-review -- GREY mesh + a round GREEN ball at every CP.
     CP = cp_openings.connection_points (Contact + CableEntry openings, deduped per terminal,
     user-approved 2026-07-17). Ball offset a little OUTWARD along the opening direction so it
     sits ON the opening (not buried) and shows from any angle / any viewer.
     state = CPREVIEW if >=1 CP, else REVIEW_EMPTY (no Contact & no CableEntry -> human check).
  <id>.meta.json : class vertex counts (human+model), CP counts + per-class breakdown,
     review status, checkpoint hash, quality-gate results.
Camera points at the large face. Positive control: 0311142 (4 CP).

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe viz_semantic_full.py --split val
"""
import argparse, json, os, hashlib
import numpy as np
import trimesh
import scheffler_dataset as dataset
import diffusionnet, connector3d, cp_openings

CKPT = "results/scheffler_semantic/refit91.pt"; OPCACHE = "results/scheffler_semantic/operators"
H, C, S, CE, LS = (int(connector3d.HOUSING), int(connector3d.CONTACT), int(connector3d.SNAP_POINT),
                   int(connector3d.CABLE_ENTRY), int(connector3d.LABEL_SURFACE))
COL = {H: (175, 175, 175), C: (55, 110, 235), S: (240, 215, 40), CE: (25, 225, 55), LS: (235, 140, 40)}
NAME = {H: "Housing", C: "Contact", S: "SnapPoint", CE: "CableEntry", LS: "LabelSurface"}
BALL = (20, 235, 55, 255)   # green round CP marker


def rgba(labels, cmap):
    a = np.array([cmap.get(int(l), (150, 150, 150)) for l in labels], np.uint8)
    return np.concatenate([a, np.full((len(a), 1), 255, np.uint8)], 1)


def _look_at(eye, target, up):
    f = target - eye; f = f / (np.linalg.norm(f) + 1e-9)
    r = np.cross(f, up); r = r / (np.linalg.norm(r) + 1e-9)
    u = np.cross(r, f)
    T = np.eye(4); T[:3, 0] = r; T[:3, 1] = u; T[:3, 2] = -f; T[:3, 3] = eye
    return T


def _cam(scene, V):
    """Point the embedded camera at the LARGE face (down the thinnest bbox axis)."""
    ext = V.max(0) - V.min(0); thin = int(np.argmin(ext)); c = V.mean(0)
    eye = c.copy(); eye[thin] += float(max(ext)) * 2.2
    up = np.zeros(3); up[(thin + 1) % 3] = 1.0
    try:
        scene.camera_transform = _look_at(eye, c, up)
    except Exception:
        pass
    return scene


def reorient(V):
    """Return (R, c) rotating the part to an ISOMETRIC pose so NO axis-aligned viewer default
    ever opens on a pure edge. Rather than guess which axis a given viewer looks down (Windows
    3D Viewer ignores the glTF camera and does not use -Z), we align the LARGE-face normal (the
    thin bbox axis) to the (1,1,1) diagonal -> the flat face is oblique to every primary axis, so
    a front/top/side/iso default all show the markers. Proper rotation (det +1, no mirror)."""
    ext = V.max(0) - V.min(0)
    n = np.zeros(3); n[int(np.argmin(ext))] = 1.0          # large-face normal = thin axis
    t = np.ones(3) / np.sqrt(3.0)                           # isometric target direction
    v = np.cross(n, t); s = np.linalg.norm(v); c = float(np.dot(n, t))
    if s < 1e-8:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))
    return R, V.mean(0)


def _ap(P, M, c):
    return (np.asarray(P, float) - c) @ M.T


def semantic_glb(V, F, labels, out):
    sc = trimesh.Scene(trimesh.Trimesh(V, F, vertex_colors=rgba(labels, COL), process=False))
    _cam(sc, V).export(out)


BALL_CT = (40, 120, 235, 255)   # blue Contact marker (auxiliary feature, per thesis)
CLASS_BALL = {CE: BALL, C: BALL_CT}


def cpreview_glb(V, F, cps, out):
    """Grey mesh + a ball per connection feature, coloured BY CLASS (green=CableEntry=CP,
    blue=Contact=auxiliary). Both shown per the thesis so no connection part looks empty."""
    grey = np.tile((170, 170, 170, 255), (len(V), 1)).astype(np.uint8)
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.Trimesh(V, F, vertex_colors=grey, process=False), node_name="mesh")
    diag = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0
    r = diag * 0.032
    for i, cp in enumerate(cps):
        p = np.asarray(cp["point"], float); d = np.asarray(cp["direction"], float)
        col = CLASS_BALL.get(int(cp["source_label"]), BALL)
        s = trimesh.creation.icosphere(subdivisions=2, radius=r)
        s.apply_translation(p + d * diag * 0.012)   # small proud offset -> sits on the opening
        s.visual.vertex_colors = np.tile(col, (len(s.vertices), 1)).astype(np.uint8)
        scene.add_geometry(s, node_name=f"CP_{i}")
    _cam(scene, V).export(out)


def check_glb(path, expect_cp):
    """Quality gates -> (failures, sha16). Counts CP-marker balls (green OR blue)."""
    fails = []
    m = trimesh.load(path, force="mesh")
    if len(m.vertices) < 100 or len(m.faces) < 100:
        fails.append("mesh empty/degenerate")
    vc = getattr(m.visual, "vertex_colors", None)
    n_ball = 0
    if vc is not None:
        for col in (BALL, BALL_CT):
            n_ball += int((np.abs(vc[:, :3].astype(int) - list(col[:3])).sum(1) < 40).sum())
    got = n_ball // 162  # icosphere(sub=2) ~162 verts
    if expect_cp is not None and got != expect_cp:
        fails.append(f"CP marker count {got} != expected {expect_cp}")
    return fails, hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val"); ap.add_argument("--out", default="_scheffler_viz")
    ap.add_argument("--min-v", type=int, default=10); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-conf", type=float, default=0.0)     # component-mean gate (superseded)
    ap.add_argument("--vertex-conf", type=float, default=0.9)  # per-vertex mask on MODEL preds
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model, meta, _ = diffusionnet.load_checkpoint(CKPT, device="cuda")
    ck_hash = hashlib.sha256(open(CKPT, "rb").read()).hexdigest()[:16]
    samples = dataset.load_split("wscad_corpus_scheffler_exact", a.split,
                                 allow_locked=(a.split == "test_locked"), verify_hashes=False)
    all_fail = []; n = 0
    for s in samples:
        pid = s["part_id"]
        if a.only and pid != a.only:
            continue
        V = np.asarray(s["verts"], float); F = np.asarray(s["faces"], int)
        hlab = np.asarray(s["labels"])
        plab, probs = diffusionnet.predict(model, meta, V, F, device="cuda", op_cache_dir=OPCACHE, return_probs=True)
        plab = np.asarray(plab)
        # show BOTH connection features (thesis): green CableEntry + blue Contact, no cross-class
        # merge. The CP-count metric stays CableEntry-only (cp_evaluate); this is the review viz.
        both = (CE, C)
        h_cps = cp_openings.connection_points(V, F, hlab, min_v=a.min_v, classes=both, dedupe_mm=0.0)
        m_cps = cp_openings.connection_points(V, F, plab, min_v=a.min_v, probs=probs,
                                              min_conf=a.min_conf, vertex_conf=a.vertex_conf,
                                              classes=both, dedupe_mm=0.0)
        # bake orientation so any viewer opens on the large face (not the thin edge)
        M, c = reorient(V); Vt = _ap(V, M, c)
        def tf(cps):
            return [{"point": _ap(x["point"], M, c), "direction": np.asarray(x["direction"], float) @ M.T,
                     "source_label": x["source_label"]} for x in cps]
        h_t, m_t = tf(h_cps), tf(m_cps)
        semantic_glb(Vt, F, hlab, os.path.join(a.out, f"{pid}_SEMANTIC_human.glb"))
        semantic_glb(Vt, F, plab, os.path.join(a.out, f"{pid}_SEMANTIC_model.glb"))
        state = "CPREVIEW" if h_cps else "REVIEW_EMPTY"
        cpreview_glb(Vt, F, h_t, os.path.join(a.out, f"{pid}_{state}_human.glb"))
        cpreview_glb(Vt, F, m_t, os.path.join(a.out, f"{pid}_{state}_model.glb"))
        fh, hh = check_glb(os.path.join(a.out, f"{pid}_{state}_human.glb"), len(h_cps))
        fm, mh = check_glb(os.path.join(a.out, f"{pid}_{state}_model.glb"), len(m_cps))
        if hh == mh and (h_cps or m_cps):
            fh.append("human==model byte-identical")

        def brk(cps):
            return {NAME[C]: sum(1 for c in cps if c["source_label"] == C),
                    NAME[CE]: sum(1 for c in cps if c["source_label"] == CE)}
        counts_h = {NAME[k]: int((hlab == k).sum()) for k in COL}
        counts_m = {NAME[k]: int((plab == k).sum()) for k in COL}
        meta_out = {"part_id": pid, "split": a.split, "review_status": state,
                    "cp_definition": "VIZ shows both features (green CableEntry + blue Contact); "
                                     "CP-COUNT metric = CableEntry only (cp-v2, see cp_config.json)",
                    "human_class_verts": counts_h, "model_class_verts": counts_m,
                    "human_cp": len(h_cps), "model_cp": len(m_cps),
                    "human_cp_by_class": brk(h_cps), "model_cp_by_class": brk(m_cps),
                    "physical_connection_expected": (counts_h[NAME[C]] + counts_h[NAME[CE]]) > 0,
                    "checkpoint_sha16": ck_hash, "gate_failures_human": fh, "gate_failures_model": fm}
        json.dump(meta_out, open(os.path.join(a.out, f"{pid}.meta.json"), "w"), indent=1)
        if fh or fm:
            all_fail.append((pid, fh + fm))
        print(f"  {pid}: [{state}] hCP={len(h_cps)}{brk(h_cps)} mCP={len(m_cps)} "
              f"gates={'FAIL '+str(fh+fm) if (fh or fm) else 'ok'}")
        n += 1
        if a.limit and n >= a.limit:
            break
    print(f"\nwrote {n} parts x4 GLB + meta -> {a.out}/")
    print(f"QUALITY GATE FAILURES: {len(all_fail)} parts" + ("" if not all_fail else f" -> {all_fail[:6]}"))
    print("DONE")


if __name__ == "__main__":
    main()
