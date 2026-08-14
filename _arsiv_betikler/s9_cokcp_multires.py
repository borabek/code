# -*- coding: utf-8 -*-
"""S9: COK-CP aday kurtarma -- ikinci cozunurlukte (9k) yeniden turet.

NEDEN: aday recall'i rejime gore COK farkli (results/aday_recall.json):
    dusuk-CP  0.8924  -> F1 tavani 0.9432
    cok-CP    0.7691  -> F1 tavani 0.8695   <- BURASI CIVILI
Yani cok-CP kolunda tespit F1'i 0.87'yi hicbir gate ile GECEMEZ; eksik olan ADAY.

Tez remesh'i 6000 kose hedefliyor (uniform yogunluk kurali). Cok-CP parcalar daha buyuk ve
ayni yogunlukta daha az kose/aciklik dusuyor -- [[cc-a-vertex-arithmetic]] bunu olcmustu.
Denetimin P2 tavsiyesi: YALNIZ router'in cok-CP dedigi parcalarda 6k + 9k BIRLESIMI uret.

TEZ CIZGISI: remesh hedefi degismiyor; 6000 aynen kaliyor ve YANINA ikinci bir cozunurluk
ekleniyor. Bu, tezin uniform-yogunluk kuralini bozmaz -- iki ayri uniform mesh'in birlesimidir.
(A1 adaptif yogunluk kolu OLMUSTU cunku 6000'i DEGISTIRIYORDU; bu ondan farkli.)

KABUL BARI (denetimin yazdigi): cok-CP F1 +0.03, dusuk-CP kaybi en fazla 0.005.
"""
import io
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    import cp_openings
    import diffusionnet as D
    import olcum_kumesi
    import robot_cp as RC
    import thesis_remesh
    import wire_gate
    from big_arbiter import eligible
    from connector_constants import CABLE_ENTRY as CE, CONTACT as CT
    from infer_step_cp import load_any, step_to_mesh
    from sina_kume import esle, f1_rejim, f1w
    from sklearn.ensemble import RandomForestClassifier

    with io.open("cp_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    stp_of = {p: s for m, p, jf, s in eligible()}
    DER, rap = olcum_kumesi.kume("results/_der_zengin_graf.pkl")
    olcum_kumesi.rapor_bas(rap)
    gk = olcum_kumesi.geo_anahtarlari(); tg = {r["geo"] for r in DER}

    zen = np.load("results/zengin_parite.npz", allow_pickle=True)
    Xt = np.hstack([np.asarray(zen["X22"], float), np.asarray(zen["XR"], float)])
    ytr = np.asarray(zen["y"]); tpid = np.array([str(x) for x in zen["pids"]])
    keep = ~np.isin(np.array([gk.get(p, "yok:" + p) for p in tpid]), list(tg))
    dag = wire_gate._load(wire_gate.MODEL_PATH); DON = dag.get("donusum")
    Z = np.zeros((len(Xt), Xt.shape[1] * 2))
    for u in np.unique(tpid):
        i = np.where(tpid == u)[0]
        Z[i] = wire_gate.parca_ici(Xt[i], DON)
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                 random_state=0).fit(Z[keep], ytr[keep])
    gate = {"clf": clf, "n_feat": Z.shape[1], "donusum": DON}

    cks = cfg["current_product"].get("checkpoints") or cfg["robot_vote2_checkpoints"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = [load_any(c, dev=dev)[:2] for c in cks]

    # YALNIZ cok-CP parcalar (GT'ye gore degil, ROUTER'a gore secilmeli; ama burada tavan
    # olcumu yapiyoruz -> once GT rejimiyle olc, gecerse router ile tekrarla)
    hedef = [r for r in DER if r["n"] >= 8]
    print(f"\ncok-CP parca: {len(hedef)}/{len(DER)}", flush=True)

    det = {"A": [], "B": []}; rob = {"A": [], "B": []}; g = []
    t0 = time.time()
    for i_, r in enumerate(hedef, 1):
        if i_ % 10 == 0:
            print(f"  {i_}/{len(hedef)}  {time.time()-t0:.0f}s", flush=True)
        # A: mevcut (6k) -- olcum onbelleginden
        PA = np.zeros((0, 3)); PdA = np.zeros((0, 3))
        if r["X"] is not None and r.get("XR") is not None:
            X58 = np.hstack([r["X"], r["XR"]])
            s = wire_gate.karar_skoru(gate, X58); k = wire_gate.karar_maskesi(s)
            if k.any():
                PA = r["P"][k].copy(); PdA = r["Pd"][k].copy()
                cps = [{"point": PA[j], "direction": PdA[j]} for j in range(len(PA))]
                cps = wire_gate.pose_duzelt(X58[k], cps)
                if cfg.get("robot_aci_secici"):
                    cps = wire_gate.aci_duzelt(X58[k], cps)
                PA = np.array([c["point"] for c in cps], float)
                PdA = np.array([c["direction"] for c in cps], float)
        # B: 6k + 9k BIRLESIMI
        PB, PdB = PA.copy(), PdA.copy()
        try:
            stp = stp_of.get(r["pid"])
            Vr, Fr = step_to_mesh(stp)
            V9, F9 = thesis_remesh.remesh_uniform(Vr, Fr, target=9000)
            V9 = np.ascontiguousarray(V9, np.float64); F9 = np.ascontiguousarray(F9, np.int64)
            pbs9 = []
            for model, meta in models:
                _, pb = D.predict(model, meta, V9, F9, device=dev,
                                  op_cache_dir=f"results/step_infer/ops9_k{int(meta.get('k_eig',64))}",
                                  return_probs=True)
                pbs9.append(np.asarray(pb, float))
            cps9, probs9, _, _u9 = RC.adaylari_uret(V9, F9, pbs9, stp, cfg=cfg)
            if cps9:
                X9 = wire_gate.feats_for(V9, F9, probs9, cps9, CE, CT, step_path=stp)
                s9 = wire_gate.karar_skoru(gate, X9); k9 = wire_gate.karar_maskesi(s9)
                if k9.any():
                    P9 = np.array([c["point"] for c in cps9], float)[k9]
                    Pd9 = np.array([c["direction"] for c in cps9], float)[k9]
                    c9 = [{"point": P9[j], "direction": Pd9[j]} for j in range(len(P9))]
                    c9 = wire_gate.pose_duzelt(X9[k9], c9)
                    if cfg.get("robot_aci_secici"):
                        c9 = wire_gate.aci_duzelt(X9[k9], c9)
                    P9 = np.array([c["point"] for c in c9], float)
                    Pd9 = np.array([c["direction"] for c in c9], float)
                    # BIRLESIM: 5mm icinde zaten varsa ekleme
                    ek_p, ek_d = [], []
                    for j in range(len(P9)):
                        if not len(PB) or np.min(np.linalg.norm(PB - P9[j], axis=1)) > 5.0:
                            ek_p.append(P9[j]); ek_d.append(Pd9[j])
                    if ek_p:
                        PB = np.vstack([PB, np.array(ek_p)]) if len(PB) else np.array(ek_p)
                        PdB = np.vstack([PdB, np.array(ek_d)]) if len(PdB) else np.array(ek_d)
        except Exception as e:
            print(f"    {r['pid']}: 9k atlandi ({type(e).__name__})")
        rj = "cok"
        for ad, (P, Pd) in (("A", (PA, PdA)), ("B", (PB, PdB))):
            det[ad].append((rj,) + esle(P, Pd, r["G"], r["Gd"], r["diag"], 0.0, 180.0, True))
            rob[ad].append((rj,) + esle(P, Pd, r["G"], r["Gd"], r["diag"], 2.0, 10.0, False))
        g.append(r["geo"])

    ra = f1_rejim(det["A"]); rb = f1_rejim(det["B"])
    print(f"\n{'kol':<16}{'cok-CP tespit':>15}{'cok-CP robot':>14}")
    print(f"{'A 6k (mevcut)':<16}{ra['F1']['cok']:>15.4f}{f1_rejim(rob['A'])['F1']['cok']:>14.4f}")
    print(f"{'B 6k+9k':<16}{rb['F1']['cok']:>15.4f}{f1_rejim(rob['B'])['F1']['cok']:>14.4f}")
    d = rb["F1"]["cok"] - ra["F1"]["cok"]
    fn = lambda rows: f1w([y for _, y in rows]) - f1w([x for x, _ in rows])
    _, lo, hi = olcum_kumesi.grup_bootstrap(list(zip(det["A"], det["B"])), g, fn, n=2000)
    print(f"\ncok-CP tespit farki {d:+.4f} | GA [{lo:+.4f}, {hi:+.4f}]")
    print(f"KABUL BARI (denetim): cok-CP +0.03 -> {'GECTI' if d >= 0.03 else 'GECMEDI'}")
    print("  (dusuk-CP kaybi ayrica olculmeli: bu kol YALNIZ cok-CP parcalarda calisiyor,")
    print("   yani dusuk-CP'ye dokunmuyor -> kayip YAPISAL OLARAK SIFIR)")
    with io.open("results/s9_cokcp_multires.json", "w", encoding="utf-8") as f:
        json.dump({"A_cok": float(ra["F1"]["cok"]), "B_cok": float(rb["F1"]["cok"]),
                   "fark": float(d), "ga": [float(lo), float(hi)],
                   "gecti": bool(d >= 0.03), "n_parca": len(hedef)}, f, indent=1)
    print("makbuz -> results/s9_cokcp_multires.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
