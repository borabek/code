# -*- coding: utf-8 -*-
"""AUTO-tier esik kalibrasyonu (vote>=2 urunu).

conf_auto 0.75 TEK model icin ayarlanmisti; vote toplulugunda temsilci-guveni = en yuksek uyenin
guveni oldugundan semantik degisti (robot_e2e: PXC AUTO precision 0.709->0.594 dustu). Burada AUTO
tier'i, topluluğun asil guvenilirlik sinyaliyle -- OY SAYISI + guven -- yeniden kalibre ediyoruz.

Sizinti-siz held-out (145 WEI + PXC ornegi) uzerinde her hayatta-kalan vote CP'sinin (guven, oy,
TP/FP) kaydini BIR kez cache'le; sonra (conf_auto x min_auto_votes) izgarasini ucuza tara ve her hucre
icin AUTO precision / AUTO recall / REVIEW yuku raporla. Hedef: robotun OTONOM eyledigi noktalarin
GERCEK aciklik olma orani (AUTO precision) yuksek; AUTO recall makul.
"""
import os, sys, json, argparse, time, itertools
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diffusionnet as D, cp_openings, thesis_remesh, connector3d
from cad_eval import align_frames
from infer_step_cp import step_to_mesh, load_any
from big_arbiter import eligible, CE, CT, OP
from robot_cp import _vote2

HELD = set(open("_hw_r3.txt").read().split())


def matched_preds(P, G, Gd, tol, axis_tol=40.0):
    """axis-aware greedy; her pred icin TP mi (True/False) dizisi dondur."""
    tp = np.zeros(len(P), bool)
    if not len(P) or not len(G):
        return tp
    diff = P[:, None, :] - G[None, :, :]
    al = (diff * Gd[None, :, :]).sum(-1)
    pe = np.linalg.norm(diff - al[..., None] * Gd[None, :, :], axis=-1)
    pe = np.where(np.abs(al) <= axis_tol, pe, np.inf)
    order = sorted((pe[i, k], i, k) for i in range(len(P)) for k in range(len(G)) if pe[i, k] <= tol)
    up, ug = set(), set()
    for d, i, k in order:
        if i in up or k in ug: continue
        up.add(i); ug.add(k); tp[i] = True
    return tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pxc-cap", type=int, default=100, help="PXC parca siniri (hiz)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cks = json.load(open("cp_config.json"))["robot_vote2_checkpoints"]
    models = [load_any(c, dev=a.device)[:2] for c in cks]
    os.environ["BA_ALLOW_SEEN"] = "1"
    parts = [p for p in eligible() if (p[0] == "WEI" and p[1] in HELD)]
    pxc = [p for p in eligible() if p[0] == "PXC"][:a.pxc_cap]
    parts += pxc
    print(f"{len(parts)} parca ({len(parts)-len(pxc)} WEI + {len(pxc)} PXC) | {len(models)} model vote", flush=True)

    # her parca: vote CP'leri -> (mfg, conf, votes, tp)
    recs = []; n_gt = {"WEI": 0, "PXC": 0}; t0 = time.time()
    for k, (mfg, pid, jf, stp) in enumerate(parts, 1):
        try:
            j = json.load(open(jf, encoding="utf-8-sig"))
            Vj = np.array([[p["X"], p["Y"], p["Z"]] for p in j["Graphic3d"]["Points"]], float)
            G = np.array([[c["Point"]["X"], c["Point"]["Y"], c["Point"]["Z"]] for c in j.get("ConnectionPoints", [])], float)
            Gd = np.array([[c["InsertDirection"]["X"], c["InsertDirection"]["Y"], c["InsertDirection"]["Z"]] for c in j.get("ConnectionPoints", [])], float)
            if not len(G): continue
            Gd = Gd / (np.linalg.norm(Gd, axis=1, keepdims=True) + 1e-9)
            Vr, Fr = step_to_mesh(stp)
            V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
            V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
            per_model = []
            for model, meta in models:
                _, pb = D.predict(model, meta, V, F, device=a.device, op_cache_dir=OP, return_probs=True)
                pb = np.asarray(pb, float); lab = pb.argmax(-1)
                per_model.append(cp_openings.connection_points(
                    V, F, lab, min_v=30, classes=(CE, CT), dedupe_mm=10.0,
                    probs=pb, vertex_conf=0.5, ct_depth_min_mm=1.0, cluster_mm=5.0))
            cps = _vote2(per_model)
            R, t, _ = align_frames(Vr, Vj)
            P = (np.array([np.asarray(c["point"]) for c in cps], float) @ R.T + t) if cps else np.zeros((0, 3))
            tol = max(3.0, 0.06 * float(np.linalg.norm(Vj.max(0) - Vj.min(0))))
            tp = matched_preds(P, G, Gd, tol)
            n_gt[mfg] += len(G)
            for c, is_tp in zip(cps, tp):
                recs.append((mfg, float(c.get("confidence", 0.0)), int(c.get("_votes", 1)), bool(is_tp)))
        except Exception:
            continue
        if k % 25 == 0: print(f"  {k}/{len(parts)}  {time.time()-t0:.0f}s", flush=True)

    recs = np.array([(1 if m == "WEI" else 0, cf, v, tp) for (m, cf, v, tp) in recs], float)
    json.dump({"n_gt": n_gt, "recs": recs.tolist()}, open("results/calib_conf.json", "w"))
    print(f"\n=== AUTO-tier KALIBRASYON ({len(recs)} vote CP; GT WEI {n_gt['WEI']} / PXC {n_gt['PXC']}) ===")
    print(f"{'min_oy':>6s} {'conf>=':>6s} | {'WEI Pauto':>9s} {'Rauto':>6s} | {'PXC Pauto':>9s} {'Rauto':>6s} | {'AUTO%':>6s}")
    best = None
    for mv, cf in itertools.product([2, 3, 4], [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]):
        auto = (recs[:, 2] >= mv) & (recs[:, 1] >= cf)
        row = []
        for mflag, name in ((1, "WEI"), (0, "PXC")):
            m = recs[:, 0] == mflag
            atp = int(((recs[:, 3] == 1) & auto & m).sum())
            afp = int(((recs[:, 3] == 0) & auto & m).sum())
            P = atp / max(atp + afp, 1); Rc = atp / max(n_gt[name], 1)
            row.append((P, Rc))
        auto_frac = 100.0 * auto.mean() if len(auto) else 0
        (wp, wr), (pp, pr) = row
        print(f"{mv:6d} {cf:6.2f} | {wp:9.3f} {wr:6.3f} | {pp:9.3f} {pr:6.3f} | {auto_frac:5.1f}%", flush=True)
        # hedef: iki ureticide de AUTO precision >=0.75, sonra toplam AUTO recall'i maksimize et
        if wp >= 0.75 and pp >= 0.75:
            score = row[0][1] + row[1][1]     # WEI Rauto + PXC Rauto
            if best is None or score > best[0]:
                best = (score, mv, cf, wp, wr, pp, pr)
    print()
    if best:
        _, mv, cf, wp, wr, pp, pr = best
        print(f"  ONERI: min_auto_votes={mv}, conf_auto={cf:.2f}  -> "
              f"WEI AUTO P{wp:.3f}/R{wr:.3f}, PXC AUTO P{pp:.3f}/R{pr:.3f}")
        print(f"  (hedef: iki uretici AUTO precision >=0.75, sonra AUTO recall max)")
    else:
        print("  UYARI: hicbir hucre iki ureticide de AUTO P>=0.75 vermedi -- esigi dusur veya oyu artir")


if __name__ == "__main__":
    main()
