# -*- coding: utf-8 -*-
"""Test-time augmentation for the DiffusionNet segmentation (working-product lever, no training,
no labels). DiffusionNet with XYZ input is ORIENTATION-sensitive; unseen STEP parts arrive in
varied frames -> the model misses CableEntry. TTA runs the SAME intrinsic operators (computed
once) with several ROTATED xyz inputs and averages the per-vertex probabilities -> steadier
segmentation, recovering openings the single-view model misses.

`tta_predict(model, meta, V, F, device, op_cache, n_rot)` -> per-vertex label (numpy).
"""
import numpy as np
import torch
import diffusionnet as D


def _rots(n):
    R = [np.eye(3)]
    rng = np.random.default_rng(0)
    # rotations about each axis + a few random small tilts (orientation robustness)
    for ax in range(3):
        for ang in (np.pi / 2, np.pi):
            c, s = np.cos(ang), np.sin(ang); M = np.eye(3)
            a, b = [x for x in range(3) if x != ax]
            M[a, a] = c; M[a, b] = -s; M[b, a] = s; M[b, b] = c
            R.append(M)
    while len(R) < n:
        v = rng.normal(size=3); v /= np.linalg.norm(v); ang = rng.uniform(-0.4, 0.4)
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R.append(np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K)
    return R[:n]


def tta_predict(model, meta, V, F, device, op_cache, n_rot=8):
    V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
    ops = D.precompute_operators(V, F, meta["n_eig"] if "n_eig" in meta else 64, op_cache_dir=op_cache)
    ops = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in ops.items()}
    model.eval(); probs = None
    with torch.no_grad():
        for R in _rots(n_rot):
            Vr = np.ascontiguousarray(V @ R.T, np.float32)
            x = torch.tensor(Vr, dtype=torch.float32, device=device)
            out = D._forward(model, ops, x)               # log-probs (NLL model)
            p = out.exp()
            probs = p if probs is None else probs + p
    return (probs / n_rot).argmax(-1).cpu().numpy()


if __name__ == "__main__":
    import sys, json, connector3d, cp_openings, thesis_remesh
    from infer_step_cp import step_to_mesh
    CE = int(connector3d.CABLE_ENTRY)
    model, meta, _ = D.load_checkpoint("results/scheffler_semantic/refit91.pt", device="cuda")
    cand = {c["part_id"]: c for c in json.load(open("benchmark_candidates.json"))["candidates"]}
    for pid in sys.argv[1:]:
        Vr, Fr = step_to_mesh(cand[pid]["step_file"])
        V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
        V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
        base = np.asarray(D.predict(model, meta, V, F, device="cuda", op_cache_dir="results/step_infer/ops"))
        tta = tta_predict(model, meta, V, F, "cuda", "results/step_infer/ops", n_rot=8)
        nb = int((base == CE).sum()); nt = int((tta == CE).sum())
        cb = len(cp_openings.connection_points(V, F, base, min_v=20, classes=(CE,)))
        ct = len(cp_openings.connection_points(V, F, tta, min_v=20, classes=(CE,)))
        print(f"  {pid}: CableEntry verts base={nb} tta={nt} | CP base={cb} tta={ct}", flush=True)
