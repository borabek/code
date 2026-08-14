# -*- coding: utf-8 -*-
"""B2: TURETME GEVSETME -- adaysiz CP'ler esikten mi dusuyor, gorulmuyor mu?

B1 DOLAYLI olarak "segmentasyon konusuyor" dedi (%70.5) ama olcumu mekansal olarak kaba:
GT noktasi kanalin ICINDE, adaylar AGIZDA; 4mm'lik pencere kaymis olabilir. Ulasan CP'lerin
de %70.9'unda CE/CT argmax tepesi yoktu -- bu, pencerenin guvenilmez oldugunun isareti.

BU BETIK DOGRUDAN SORAR: turetme parametrelerini GEVSETINCE o CP'ler geliyor mu?
    URUN     : min_v=4,  vertex_conf=0.30, cluster_mm=3.0
    GEVSEK-1 : min_v=2,  vertex_conf=0.20, cluster_mm=3.0
    GEVSEK-2 : min_v=1,  vertex_conf=0.10, cluster_mm=5.0
Olculen: ADAY RECALL'i (kac GT'ye aday ulasiyor) ve ADAY SAYISI (bedel).

OKUMA:
  recall belirgin YUKSELIYORSA  -> esik sorunu; ucuz duzeltme mumkun (bedel gate'e kalir)
  recall KIPIRDAMIYORSA         -> segmentasyon o acikliklari GERCEKTEN gormuyor
                                   -> Rota B = etiket/ontoloji, ve pahalidir

Bu bir ADAY-DUZEYI olcumdur; uctan uca kazanc AYRICA olculur (aday sayisi artarsa gate'in
isi zorlasir -- havuz genisletme daha once OLU cikmisti, o yuzden recall artisi TEK BASINA
yetmez, sadece kapiyi acar).
"""
import io
import json
import os
import sys
import time

import numpy as np
import torch

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

AYAR = [("URUN     ", 4, 0.30, 3.0), ("GEVSEK-1 ", 2, 0.20, 3.0), ("GEVSEK-2 ", 1, 0.10, 5.0)]


def main():
    import cp_openings
    import diffusionnet as D_
    import olcum_kumesi
    import robot_cp as RC
    import thesis_remesh
    from big_arbiter import eligible
    from connector_constants import CABLE_ENTRY as CE, CONTACT as CT
    from infer_step_cp import load_any, step_to_mesh
    from gece_kilit import bekci

    bekci("b2")
    with io.open("cp_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    stp = {p: s for m, p, jf, s in eligible()}
    DER, rap = olcum_kumesi.kume("results/_der_tam.pkl")
    olcum_kumesi.rapor_bas(rap)
    cks = cfg["current_product"].get("checkpoints") or cfg["robot_vote2_checkpoints"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = [load_any(c, dev=dev)[:2] for c in cks]

    SAY = {ad: {"ulasan": 0, "aday": 0} for ad, _, _, _ in AYAR}
    top_gt = 0
    t0 = time.time()
    for k, r in enumerate(DER, 1):
        if k % 20 == 0:
            print(f"  {k}/{len(DER)}  {time.time()-t0:.0f}s", flush=True)
            bekci(f"b2 {k}")
        G = np.asarray(r["G"], float); Gd = np.asarray(r["Gd"], float)
        if not len(G):
            continue
        try:
            Vr, Fr = step_to_mesh(stp[r["pid"]])
            V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
            V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
            pbs = []
            for model, meta in models:
                _, pb = D_.predict(model, meta, V, F, device=dev,
                                   op_cache_dir=f"results/step_infer/ops_k{int(meta.get('k_eig',64))}",
                                   return_probs=True)
                pbs.append(np.asarray(pb, float))
        except Exception as e:
            print(f"    {r['pid']}: {type(e).__name__}")
            continue
        top_gt += len(G)
        tt = max(3.0, 0.06 * float(r["diag"]))
        for ad, mv, vc, cl in AYAR:
            try:
                listeler = [cp_openings.connection_points(
                    V, F, q.argmax(-1), min_v=mv, classes=(CE, CT), dedupe_mm=10.0,
                    probs=q, vertex_conf=vc, ct_depth_min_mm=1.0, cluster_mm=cl,
                    step_path=stp[r["pid"]]) for q in pbs]
                cps = RC._vote2(listeler, cluster_mm=5.0, min_votes=1)
            except Exception:
                continue
            if not cps:
                continue
            P = np.array([c["point"] for c in cps], float)
            diff = P[:, None, :] - G[None, :, :]
            al = (diff * Gd[None, :, :]).sum(-1)
            pe = np.linalg.norm(diff - al[..., None] * Gd[None, :, :], axis=-1)
            pe = np.where(np.abs(al) > 40, np.inf, pe)
            SAY[ad]["ulasan"] += int((pe.min(0) <= tt).sum())
            SAY[ad]["aday"] += len(P)

    print(f"\nGT toplam {top_gt}")
    print(f"{'ayar':<12}{'aday recall':>13}{'aday sayisi':>13}{'aday/GT':>10}")
    taban = None
    for ad, mv, vc, cl in AYAR:
        s = SAY[ad]; rc = s["ulasan"] / max(top_gt, 1)
        if taban is None:
            taban = rc
        print(f"{ad:<12}{rc:>13.4f}{s['aday']:>13}{s['aday']/max(top_gt,1):>10.2f}")
    son = SAY[AYAR[-1][0]]["ulasan"] / max(top_gt, 1)
    kaz = son - taban
    print(f"\nrecall kazanci (GEVSEK-2 vs URUN): {kaz:+.4f}")
    bedel = SAY[AYAR[-1][0]]["aday"] / max(SAY[AYAR[0][0]]["aday"], 1)
    print(f"aday sayisi carpani: {bedel:.2f}x")
    if kaz >= 0.03:
        print("\nKARAR: ESIK SORUNU -- adaysiz CP'lerin onemli bolumu gevsetmeyle GELIYOR.")
        print("       Kapi acildi: simdi bedel (aday sayisi) gate'e havale edilip UCTAN UCA olculur.")
    else:
        print("\nKARAR: SEGMENTASYON GORMUYOR -- gevsetme recall'i kurtarmiyor.")
        print("       Rota B = etiket/ontoloji ve PAHALIDIR; ucuz esik duzeltmesi YOK.")
    with io.open("results/b2_turetme.json", "w", encoding="utf-8") as f:
        json.dump({"gt": top_gt, "sonuc": {a: SAY[a] for a in SAY},
                   "recall_kazanci": float(kaz), "aday_carpani": float(bedel)}, f, indent=1)
    print("makbuz -> results/b2_turetme.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
