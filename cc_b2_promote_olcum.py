# -*- coding: utf-8 -*-
"""CC-B2: 'yukseltme' (conn_promote) cok-CP ADAY-RECALL'unu yukseltiyor mu?

CC-B bulgusu: kacirilan GT'lerin %76'sinda yakinda CE+CT olasiligi >= 0.30 var ama argmax orani 0.00.
Sinyal VARDI, aday olusumu argmax'tan basladigi icin kapi kapaliydi.

BU OLCUM: aday-recall (kac GT icin en az bir aday olusuyor) -- gate'ten ONCEKI tavan.
Recall tavani yukselmezse gate'e bakmanin anlami yok.
KILL KRITERI (F1 degil cunku burada gate yok): aday-recall katkisi < 0.02 ise OLU.
"""
import os, sys, json, time
import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HIGH_CP = 8
PROMOTES = (0.0, 0.25, 0.30, 0.40, 0.50)


def main(limit=45):
    import torch, trimesh
    import thesis_remesh, cp_openings
    import diffusionnet as D
    from cad_eval import align_frames
    from infer_step_cp import step_to_mesh, load_any
    from big_arbiter import eligible
    from connector_constants import CABLE_ENTRY as CE, CONTACT as CT

    lock = json.load(open("results/split_lock.json")); LOCK = set(lock["locked_parts"])
    cfg = json.load(open("cp_config.json"))
    pp = cfg.get("prediction_postproc", {})
    MV = int(pp.get("min_vertices", 30)); VC = float(pp.get("vertex_confidence_mask", 0.5))
    CL = float(pp.get("cluster_mm", 5.0))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = [load_any(c, dev=dev)[:2] for c in cfg["current_product"]["checkpoints"]]

    parts = []
    for mfg, pid, jf, stp in eligible():
        if pid in LOCK: continue
        try: n = len(json.load(open(jf, encoding="utf-8-sig"))["ConnectionPoints"])
        except Exception: continue
        if n >= HIGH_CP: parts.append((mfg, pid, jf, stp, n))
    rng = np.random.RandomState(0)
    if len(parts) > limit:
        parts = [parts[i] for i in sorted(rng.choice(len(parts), limit, replace=False))]
    print(f"{len(parts)} cok-CP parca | min_v={MV} vc={VC} cluster={CL}\n", flush=True)

    # iki post-isleme rejimi: MEVCUT (mv30/vc0.50) ve GEVSEK (mv10/vc0.30)
    REGIMES = {"mevcut mv30/vc0.50": (30, 0.50, 5.0), "gevsek mv10/vc0.30": (10, 0.30, 3.0)}
    acc = {(rk, pv): [0, 0, 0] for rk in REGIMES for pv in PROMOTES}   # [tp_gt, n_gt, n_cand]
    t0 = time.time()
    for k, (mfg, pid, jf, stp, ngt) in enumerate(parts, 1):
        try:
            Vr, Fr = step_to_mesh(stp)
            V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
            V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
            a = None
            for model, meta in models:
                _, pb = D.predict(model, meta, V, F, device=dev,
                                  op_cache_dir=f"results/step_infer/ops_k{int(meta.get('k_eig',64))}",
                                  return_probs=True)
                pb = np.asarray(pb, float); a = pb if a is None else a + pb
            probs = a / len(models); lab = probs.argmax(-1)

            j = json.load(open(jf, encoding="utf-8-sig"))
            Vj = np.array([[q[c] for c in "XYZ"] for q in j["Graphic3d"]["Points"]], float)
            G = np.array([[c["Point"][q] for q in "XYZ"] for c in j["ConnectionPoints"]], float)
            Gd = np.array([[c["InsertDirection"][q] for q in "XYZ"] for c in j["ConnectionPoints"]], float)
            Gd = Gd / (np.linalg.norm(Gd, axis=1, keepdims=True) + 1e-9)
            R, t, _ = align_frames(Vr, Vj)
            Gm = (G - t) @ R; Gdm = Gd @ R
            tol = max(3.0, 0.06 * float(np.linalg.norm(V.max(0) - V.min(0))))

            for rk, (mv, vc, cl) in REGIMES.items():
                for pv in PROMOTES:
                    cps = cp_openings.connection_points(
                        V, F, lab, min_v=mv, classes=(CE, CT), dedupe_mm=10.0, probs=probs,
                        vertex_conf=vc, ct_depth_min_mm=1.0, cluster_mm=cl, conn_promote=pv)
                    n_hit = 0
                    if cps:
                        P = np.array([c["point"] for c in cps], float)
                        diff = P[:, None, :] - Gm[None, :, :]
                        al = (diff * Gdm[None, :, :]).sum(-1)
                        perp = np.linalg.norm(diff - al[..., None] * Gdm[None, :, :], axis=-1)
                        perp = np.where(np.abs(al) <= 40.0, perp, np.inf)
                        hit = np.zeros(len(Gm), bool); used = set()
                        for d_, a_, b_ in sorted((perp[a_, b_], a_, b_)
                                                 for a_ in range(len(P)) for b_ in range(len(Gm))):
                            if d_ > tol or a_ in used or hit[b_]: continue
                            hit[b_] = True; used.add(a_)
                        n_hit = int(hit.sum())
                    e = acc[(rk, pv)]
                    e[0] += n_hit; e[1] += len(Gm); e[2] += len(cps) if cps else 0
        except Exception as ex:
            print(f"  {pid}: HATA {type(ex).__name__} {str(ex)[:45]}", flush=True); continue
        if k % 10 == 0:
            print(f"  {k}/{len(parts)}  {time.time()-t0:.0f}s", flush=True)

    print(f"\n{'rejim':<22}{'promote':>9}{'aday-recall':>13}{'aday/GT':>9}")
    out = {}
    for rk in REGIMES:
        base = None
        for pv in PROMOTES:
            tp, gt, nc = acc[(rk, pv)]
            rec = tp / max(gt, 1)
            if pv == 0.0: base = rec
            d = f"  ({rec-base:+.4f})" if pv > 0 else ""
            print(f"{rk:<22}{pv:>9.2f}{rec:>13.4f}{nc/max(gt,1):>9.2f}{d}")
            out[f"{rk}|{pv}"] = {"recall": rec, "cand_per_gt": nc / max(gt, 1)}
        print()
    json.dump(out, open("results/cc_b2_promote.json", "w"), indent=1)
    print("makbuz -> results/cc_b2_promote.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 45)
