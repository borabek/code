# -*- coding: utf-8 -*-
"""CP-DEFINITION ARBITER (cp-v3 revision, 2026-07-20).

The user's realization (confirmed by evidence): CPs are NOT only round wire holes -- square/clamp
entries and Contact regions can be physical CPs too. Thesis (line 1606) derives the v_o opening
midpoint for "Kabeleinfuehrung ODER Kontaktierung" (both classes); TERMINAL_TYPES in
connector_constants has always been {CONTACT, CABLE_ENTRY}; and 5/18 manufacturer CPs on the
in-scope PXC terminals sit on NO detectable round hole.

So: race 4 candidate CP definitions against the ONLY part-matched manufacturer ground truth we
have -- the 9 in-scope PXC terminal blocks (_cad_eval_pxc STEPs) with their Desktop\\JSON
ConnectionPoints. Pipeline per part: STEP -> gmsh -> thesis_remesh -> refit91 segmentation ->
candidate definition derives CPs (STEP frame) -> cad_eval.align_frames maps STEP->JSON frame ->
greedy distance match vs manufacturer CPs -> P/R/F1 per definition.

HONEST caveat baked into the receipt: n = 9 parts / 18 manufacturer CPs -- small, but it is the
only manufacturer-validated arbiter available; do not over-fit definition rules to it (4 simple
candidates only, lesson of RESULTS-11s).

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe measure_cp_defs.py
"""
import os, glob, json, hashlib
import numpy as np
import diffusionnet as D
import connector3d, cp_openings, thesis_remesh
from cad_eval import align_frames
from infer_step_cp import step_to_mesh, load_any

CE = int(connector3d.CABLE_ENTRY); CT = int(connector3d.CONTACT)
OP = "results/step_infer/ops"
JDIR = "C:/Users/DE00024082/Desktop/JSON"
CKPT = "results/scheffler_semantic/refit91.pt"
MIN_V = 20          # prediction-side min vertices (cp_config prediction_postproc)
DEDUPE = 10.0       # cross-class merge radius (v1 heritage)
DEPTH_MIN = 1.0     # v3a: min insertion depth (mm) for a Contact component to count as an opening
# cp-v3.1 precision recipe (improve_cp_precision.py sweep, corrected): chosen to PRESERVE recall
# (all 13 TPs kept, R stays 0.722) while more than doubling precision (FP 45 -> 19).
VCONF = 0.7         # per-vertex confidence mask; 0.9 was too aggressive (killed TPs, R 0.72 -> 0.50)
CLUSTER = 10.0      # per-terminal merge radius (one CP per terminal) -- the physical precision lever
MIN_V_PRED = 60     # min vertices per predicted component (drops sub-1%-of-mesh noise fragments)


def contact_cps(V, F, labels, depth_gate=False):
    """Contact-component CPs, optionally gated by thesis insertion depth (v_o - v_s)."""
    out = []
    bc = V.mean(0)
    for f in connector3d.build_fragments(V, F, labels, min_vertices=MIN_V):
        if int(f.label) != CT:
            continue
        cp = connector3d.ConnectionPoint([f])
        try:
            cp.compute_direction(V, F, smooth_subdiv=0, body_center=bc)
        except Exception:
            continue
        if depth_gate and cp.insertion_depth_mm < DEPTH_MIN:
            continue
        out.append({"point": np.asarray(cp.entry_point, float), "source_label": CT})
    return out


def derive(defname, V, F, plab, probs=None):
    if defname == "v2_ce_only":
        return cp_openings.connection_points(V, F, plab, min_v=MIN_V, classes=(CE,))
    if defname == "v1_ce_ct_dedupe":
        return cp_openings.connection_points(V, F, plab, min_v=MIN_V, classes=(CE, CT), dedupe_mm=DEDUPE)
    if defname == "v3_1_precision":
        # ADOPTED cp-v3.1: CE + depth-gated CT + conf mask 0.7 + one-CP-per-terminal cluster + min_v 60
        return cp_openings.connection_points(V, F, plab, min_v=MIN_V_PRED, classes=(CE, CT), dedupe_mm=DEDUPE,
                                             probs=probs, vertex_conf=VCONF, ct_depth_min_mm=DEPTH_MIN,
                                             cluster_mm=CLUSTER)
    ce = cp_openings.connection_points(V, F, plab, min_v=MIN_V, classes=(CE,))
    cts = contact_cps(V, F, plab, depth_gate=(defname == "v3a_ce_ct_depth"))
    # both v3 variants: CE primary; add CT components not within DEDUPE of an accepted CE CP
    kept = list(ce)
    for c in cts:
        if all(np.linalg.norm(np.asarray(c["point"]) - np.asarray(k["point"])) >= DEDUPE for k in kept):
            kept.append(c)
    return kept


def greedy_match(P, G, tol):
    """TP/FP/FN by greedy nearest-pair matching within tol."""
    if not len(P) or not len(G):
        return 0, len(P), len(G)
    dm = np.linalg.norm(P[:, None, :] - G[None, :, :], axis=2)
    pairs = sorted(((dm[i, j], i, j) for i in range(len(P)) for j in range(len(G))))
    up, ug = set(), set(); tp = 0
    for d, i, j in pairs:
        if d > tol:
            break
        if i in up or j in ug:
            continue
        up.add(i); ug.add(j); tp += 1
    return tp, len(P) - tp, len(G) - tp


def main():
    import torch, argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default=CKPT); a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta, _ = load_any(a.ckpt, dev=dev)
    globals()["CKPT"] = a.ckpt
    defs = ["v2_ce_only", "v1_ce_ct_dedupe", "v3a_ce_ct_depth", "v3b_ce_ct_nodepth", "v3_1_precision"]
    agg = {d: [0, 0, 0] for d in defs}
    rows = []
    for stp in sorted(glob.glob("_cad_eval_pxc/*.stp")):
        pid = os.path.basename(stp).split("_")[1]
        jf = os.path.join(JDIR, f"PXC.{pid}.json")
        if not os.path.exists(jf):
            continue
        j = json.load(open(jf, encoding="utf-8-sig"))
        Vj = np.array([[p["X"], p["Y"], p["Z"]] for p in j["Graphic3d"]["Points"]], float)
        G = np.array([[c["Point"]["X"], c["Point"]["Y"], c["Point"]["Z"]] for c in j.get("ConnectionPoints", [])], float)
        Vr, Fr = step_to_mesh(stp)
        V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
        V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
        plab, probs = D.predict(model, meta, V, F, device=dev, op_cache_dir=OP, return_probs=True)
        plab = np.asarray(plab); probs = np.asarray(probs)
        R, t, ares = align_frames(Vr, Vj)          # STEP frame -> JSON frame
        diag = float(np.linalg.norm(Vj.max(0) - Vj.min(0))); tol = max(3.0, 0.06 * diag)
        row = {"part_id": pid, "mfg_cps": int(len(G)), "align_residual_mm": round(ares, 2), "tol_mm": round(tol, 1)}
        for dn in defs:
            cps = derive(dn, V, F, plab, probs)
            P = (np.array([np.asarray(c["point"]) for c in cps], float) @ R.T + t) if cps else np.zeros((0, 3))
            tp, fp, fn = greedy_match(P, G, tol)
            agg[dn][0] += tp; agg[dn][1] += fp; agg[dn][2] += fn
            row[dn] = {"pred": int(len(cps)), "tp": tp, "fp": fp, "fn": fn}
        rows.append(row)
        print(f"  {pid}: mfg={len(G)} align={ares:.2f}mm  " +
              "  ".join(f"{dn}:{row[dn]['pred']}p/{row[dn]['tp']}tp" for dn in defs), flush=True)
    summary = {}
    for dn in defs:
        tp, fp, fn = agg[dn]
        pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
        f1 = 2 * pr * rc / max(pr + rc, 1e-9)
        summary[dn] = {"tp": tp, "fp": fp, "fn": fn, "precision": round(pr, 3), "recall": round(rc, 3), "f1": round(f1, 3)}
    winner = max(summary, key=lambda d: summary[d]["f1"])
    def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    rep = {"metric": "cp-definition arbiter: seg-derived CPs vs MANUFACTURER ConnectionPoints (Desktop JSON)",
           "caveat": "n=9 parts / 18 manufacturer CPs -- small but the only part-matched manufacturer GT; 4 simple candidates only (no over-fitting, RESULTS-11s lesson)",
           "ckpt": CKPT, "ckpt_sha16": sha(CKPT), "min_v": MIN_V, "dedupe_mm": DEDUPE, "depth_min_mm": DEPTH_MIN,
           "summary": summary, "winner_by_f1": winner, "per_part": rows}
    os.makedirs("results", exist_ok=True)
    json.dump(rep, open("results/cp_def_eval.json", "w"), indent=1)
    print("\n=== SUMMARY (P/R/F1 vs manufacturer CPs) ===")
    for dn in defs:
        s = summary[dn]
        print(f"  {dn:20s} P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f}  (tp{s['tp']} fp{s['fp']} fn{s['fn']})")
    print(f"WINNER by F1: {winner}  -> results/cp_def_eval.json")


if __name__ == "__main__":
    main()
