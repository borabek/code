# -*- coding: utf-8 -*-
"""Re-tune the prediction post-proc (cluster_mm, min_v, vertex_conf) on the 1811-CP arbiter.

WHY: the product's cluster_mm=5 / min_v=45 / vertex_conf=0.7 were all chosen on tiny sets -- the
9-part (18-CP) manufacturer arbiter and the 82-CP human held-out. Tonight the ensemble decision
proved those sets mislead. Now a 1811-CP two-manufacturer arbiter exists (242 PXC + 182 WEI), so the
post-proc should be re-selected there, and judged on the COMBINED F1 so it can't overfit one maker.

SPEED: big_arbiter reruns the network per setting (~20 min each). Here the segmentation is run ONCE
per part and its probabilities cached; every (cluster_mm, min_v, vertex_conf) combination is then just
a cheap re-derivation + match. One pass, dozens of settings.

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe tune_postproc.py [--ckpt ...]
"""
import os, sys, glob, json, argparse, itertools, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diffusionnet as D, connector3d, cp_openings, thesis_remesh
from cad_eval import align_frames
from infer_step_cp import step_to_mesh, load_any

CE = int(connector3d.CABLE_ENTRY); CT = int(connector3d.CONTACT)
OP = "results/step_infer/ops"; DS = "_ds1/DataSet"


def matched(P, G, Gd, tol, axis_tol=40.0):
    if not len(P) or not len(G):
        return 0, len(P), len(G)
    diff = P[:, None, :] - G[None, :, :]
    along = (diff * Gd[None, :, :]).sum(-1)
    dm = np.linalg.norm(diff - along[..., None] * Gd[None, :, :], axis=-1)
    dm = np.where(np.abs(along) <= axis_tol, dm, np.inf)
    order = sorted((dm[i, j], i, j) for i in range(len(P)) for j in range(len(G)) if dm[i, j] <= tol)
    up, ug, tp = set(), set(), 0
    for d, i, j in order:
        if i in up or j in ug: continue
        up.add(i); ug.add(j); tp += 1
    return tp, len(P) - tp, len(G) - tp


def eligible(mfg):
    step = {os.path.basename(s).split("_")[1]: s for s in glob.glob("all_wscad_stp/*.stp")}
    seen = set()
    for d in ("_label_targets", "_label_targets_2", "_label_targets_3", "_label_targets_4"):
        seen |= {os.path.basename(os.path.normpath(p)) for p in glob.glob(d + "/*/")}
    import scheffler_dataset as ds
    for sp in ("train", "val"):
        seen |= {s["part_id"] for s in ds.load_split("wscad_corpus_scheffler_exact", sp, verify_hashes=False)}
    out = []
    for f in sorted(glob.glob(os.path.join(DS, "*ElectricalTerminal*.json"))):
        head = os.path.basename(f).split("_")[0]
        m, pid = (head.split(".", 1) + [""])[:2]
        if pid in seen or pid not in step or m != mfg: continue
        out.append((pid, f, step[pid]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/seg_extra/human77c_s0.pt")
    ap.add_argument("--per-mfg", type=int, default=90, help="cap parts per manufacturer (speed)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    model, meta, _ = load_any(a.ckpt, dev=a.device)

    # 1) run the net ONCE per part, cache probs + geometry + manufacturer CPs
    cache = []
    for mfg in ("PXC", "WEI"):
        parts = eligible(mfg)[:a.per_mfg]
        print(f"  {mfg}: {len(parts)} parca hazirlaniyor", flush=True)
        for k, (pid, jf, stp) in enumerate(parts):
            try:
                j = json.load(open(jf, encoding="utf-8-sig"))
                G = np.array([[c["Point"]["X"], c["Point"]["Y"], c["Point"]["Z"]]
                              for c in j.get("ConnectionPoints", [])], float)
                Gd = np.array([[c["InsertDirection"]["X"], c["InsertDirection"]["Y"], c["InsertDirection"]["Z"]]
                              for c in j.get("ConnectionPoints", [])], float)
                if not len(G): continue
                Gd = Gd / (np.linalg.norm(Gd, axis=1, keepdims=True) + 1e-9)
                Vj = np.array([[p["X"], p["Y"], p["Z"]] for p in j["Graphic3d"]["Points"]], float)
                Vr, Fr = step_to_mesh(stp)
                V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
                V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
                _, probs = D.predict(model, meta, V, F, device=a.device, op_cache_dir=OP, return_probs=True)
                R, t, _ = align_frames(Vr, Vj)
                tol = max(3.0, 0.06 * float(np.linalg.norm(V.max(0) - V.min(0))))
                cache.append({"mfg": mfg, "V": V, "F": F, "probs": np.asarray(probs, float),
                              "Gs": (G - t) @ R, "Gds": Gd @ R, "tol": tol})
            except Exception:
                continue
            if (k + 1) % 20 == 0: print(f"    {k+1}", flush=True)
    print(f"  toplam {len(cache)} parca onbellekte", flush=True)

    # 2) sweep post-proc cheaply
    grid = list(itertools.product([5.0], [15, 12, 10, 8, 5, 3], [0.5]))
    print(f"\n{'cluster':>7s} {'min_v':>5s} {'vconf':>5s} | {'PXC':>6s} {'WEI':>6s} {'BIRLESIK':>8s}")
    results = []
    for cl, mv, vc in grid:
        agg = {"PXC": [0, 0, 0], "WEI": [0, 0, 0]}
        for c in cache:
            lab = c["probs"].argmax(-1)
            cps = cp_openings.connection_points(c["V"], c["F"], lab, min_v=mv, classes=(CE, CT),
                                                dedupe_mm=10.0, probs=c["probs"], vertex_conf=vc,
                                                ct_depth_min_mm=1.0, cluster_mm=cl)
            P = np.array([np.asarray(x["point"]) for x in cps], float) if cps else np.zeros((0, 3))
            tp, fp, fn = matched(P, c["Gs"], c["Gds"], c["tol"])
            agg[c["mfg"]][0] += tp; agg[c["mfg"]][1] += fp; agg[c["mfg"]][2] += fn

        def f1(v):
            t, f, n = v; p = t / max(t + f, 1); r = t / max(t + n, 1)
            return 2 * p * r / max(p + r, 1e-9)
        T = [sum(x) for x in zip(agg["PXC"], agg["WEI"])]
        comb = f1(T)
        results.append((comb, cl, mv, vc, f1(agg["PXC"]), f1(agg["WEI"])))
        print(f"{cl:7.1f} {mv:5d} {vc:5.2f} | {f1(agg['PXC']):6.3f} {f1(agg['WEI']):6.3f} {comb:8.3f}", flush=True)

    results.sort(reverse=True)
    b = results[0]
    print(f"\n  EN IYI: cluster {b[1]} / min_v {b[2]} / vconf {b[3]}  -> BIRLESIK {b[0]:.3f} "
          f"(PXC {b[4]:.3f} WEI {b[5]:.3f})")
    print(f"  MEVCUT URUN: cluster 5 / min_v 45 / vconf 0.7")
    json.dump([{"comb": r[0], "cluster": r[1], "min_v": r[2], "vconf": r[3],
                "pxc": r[4], "wei": r[5]} for r in results],
              open("results/postproc_tune3.json", "w"), indent=1)


if __name__ == "__main__":
    main()
