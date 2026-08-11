# -*- coding: utf-8 -*-
"""TODO A-1: does a SEED ENSEMBLE of the human-trained models beat the best single seed?

Ensembling was measured negative on the OLD (pre-human-label) models, but never tried on the
human-trained ones. Softmax-average the seeds' per-vertex class probabilities, then run the SAME
cp-v3.1 derivation + the same 9-part manufacturer arbiter. Free (no training).

Baselines: human77c seeds 0.778/0.667/0.650 (mean 0.698, product = seed0). Adopt the ensemble only
if it beats the seed MEAN, and remember a single seed's 0.778 is a lucky draw, not a target.

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe measure_ensemble_human.py \
         --ckpts results/seg_extra/human77c_s0.pt results/seg_extra/human77c_s1.pt results/seg_extra/human77c_s2.pt
"""
import os, sys, glob, json, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diffusionnet as D
import connector3d, cp_openings, thesis_remesh
from cad_eval import align_frames
from infer_step_cp import step_to_mesh, load_any

CE = int(connector3d.CABLE_ENTRY); CT = int(connector3d.CONTACT)
OP = "results/step_infer/ops"; JDIR = "C:/Users/DE00024082/Desktop/JSON"


def greedy_match(P, G, tol):
    if not len(P) or not len(G):
        return 0, len(P), len(G)
    dm = np.linalg.norm(P[:, None, :] - G[None, :, :], axis=2)
    order = sorted((dm[i, j], i, j) for i in range(len(P)) for j in range(len(G)))
    up, ug, tp = set(), set(), 0
    for d, i, j in order:
        if d > tol: break
        if i in up or j in ug: continue
        up.add(i); ug.add(j); tp += 1
    return tp, len(P) - tp, len(G) - tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--cluster-mm", type=float, default=10.0)
    ap.add_argument("--min-v", type=int, default=60)
    ap.add_argument("--inward-mm", type=float, default=0.0, help="shift predictions inward along their own axis before matching (manufacturer-convention probe)")
    ap.add_argument("--remesh-target", type=int, default=6000, help="mesh resolution; must MATCH what the checkpoints were trained at (6000 default, 9000 for the 9k models)")
    ap.add_argument("--outward-min", type=float, default=0.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    models = [load_any(c, dev=a.device) for c in a.ckpts]
    print(f"ensemble of {len(models)}: {[os.path.basename(c) for c in a.ckpts]}", flush=True)

    T = Fp = Fn = 0
    for stp in sorted(glob.glob("_cad_eval_pxc/*.stp")):
        pid = os.path.basename(stp).split("_")[1]
        jf = os.path.join(JDIR, f"PXC.{pid}.json")
        if not os.path.exists(jf): continue
        j = json.load(open(jf, encoding="utf-8-sig"))
        Vj = np.array([[p["X"], p["Y"], p["Z"]] for p in j["Graphic3d"]["Points"]], float)
        G = np.array([[c["Point"]["X"], c["Point"]["Y"], c["Point"]["Z"]]
                      for c in j.get("ConnectionPoints", [])], float)
        Vr, Fr = step_to_mesh(stp)
        V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=a.remesh_target)
        V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
        # SOFTMAX-AVERAGE across seeds (the ensemble); each model sees the identical mesh
        acc = None
        for model, meta, _ in models:
            _, probs = D.predict(model, meta, V, F, device=a.device, op_cache_dir=OP, return_probs=True)
            probs = np.asarray(probs, float)
            acc = probs if acc is None else acc + probs
        probs = acc / len(models)
        lab = probs.argmax(-1)
        cps = cp_openings.connection_points(V, F, lab, min_v=a.min_v, classes=(CE, CT), dedupe_mm=10.0,
                                            probs=probs, vertex_conf=0.7, ct_depth_min_mm=1.0,
                                            cluster_mm=a.cluster_mm, outward_min=a.outward_min)
        R, t, ares = align_frames(Vr, Vj)
        if cps:
            P0 = np.array([np.asarray(c["point"]) for c in cps], float)
            if a.inward_mm:
                Dv = np.array([np.asarray(c["direction"], float) for c in cps], float)
                Dv = Dv / (np.linalg.norm(Dv, axis=1, keepdims=True) + 1e-9)
                P0 = P0 - a.inward_mm * Dv
            P = P0 @ R.T + t
        else:
            P = np.zeros((0, 3))
        tol = max(3.0, 0.06 * float(np.linalg.norm(Vj.max(0) - Vj.min(0))))
        tp, fp, fn = greedy_match(P, G, tol)
        T += tp; Fp += fp; Fn += fn
        print(f"  {pid}: mfg{len(G)} pred{len(P)} -> {tp}/{fp}/{fn}", flush=True)

    pr = T / max(T + Fp, 1); rc = T / max(T + Fn, 1)
    f1 = 2 * pr * rc / max(pr + rc, 1e-9)
    print(f"\n=== ENSEMBLE ({len(models)} seed, outward_min={a.outward_min}) ===")
    print(f"  TP{T} FP{Fp} FN{Fn} | P={pr:.3f} R={rc:.3f} F1={f1:.3f}")
    print(f"  (human77c single seeds: 0.778/0.667/0.650, mean 0.698)")


if __name__ == "__main__":
    main()
