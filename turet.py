# -*- coding: utf-8 -*-
"""TURET: verilen bir TOPLULUK icin olcum onbellegini SIFIRDAN uretir.

NEDEN YAZILDI (denetim bulgusu 12): butun sayilarin dayandigi `results/_der_tam.pkl`'i
URETEN betik agacta YOKTU. 24 betik onu OKUYOR, hicbiri YAZMIYOR. Zincir
`_u4_der.pkl -> _der_zengin.pkl -> _der_tam.pkl` seklinde eski kosulardan devralinmis;
hangi checkpoint listesi, hangi `min_v`/`cluster_mm`/`conn_promote` ile turetildigi
YENIDEN URETILEMEZ durumdaydi. Bu, sessiz sapmanin en genis kapisiydi: `cp_config`
degisse manset DEGISMEZDI.

BU BETIK O KAPIYI KAPATIR ve ayrica TOPLULUK DENEYLERINI mumkun kilar: `--ckpt` ile
istenen checkpoint listesi verilir, cikti ayri bir dosyaya yazilir, iki topluluk AYNI
kod yolundan gecerek karsilastirilir.

Uretilen kayit alanlari, mevcut `_der_tam.pkl` ile BIREBIR ayni sozlesmedir:
    pid, mfg, geo, kume, diag, n           parca kimligi ve rejim
    P, Pd                                  aday nokta ve yonleri (URUNUN yolundan)
    X   (22 sutun)  wire_gate.feats_for    gate'in taban ozellikleri
    XR  (36 sutun)  build_zengin_parite.zengin   zengin blok
    G, Gd                                  uretici GT (STEP cercevesine tasinmis)
    UYE                                    birlestirmede atilan uye listeleri

Kullanim:
    python turet.py --cikti results/_der_B.pkl --ckpt a.pt b.pt c.pt d.pt
    python turet.py --cikti results/_der_yeniden.pkl          # config'teki toplulukla
"""
import argparse
import io
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cikti", required=True)
    ap.add_argument("--ckpt", nargs="*", default=None,
                    help="topluluk; verilmezse cp_config'teki dagitilan liste")
    ap.add_argument("--sinir", type=int, default=0, help="ilk N parca (deneme icin)")
    a = ap.parse_args()

    import cad_eval
    import diffusionnet as D_
    import olcum_kumesi
    import robot_cp as RC
    import thesis_remesh
    import wire_gate
    from big_arbiter import eligible
    from build_zengin_parite import _normaller, zengin
    from connector_constants import CABLE_ENTRY as CE, CONTACT as CT
    from infer_step_cp import load_any, step_to_mesh

    with io.open("cp_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    cks = a.ckpt or (cfg["current_product"].get("checkpoints")
                     or cfg["robot_vote2_checkpoints"])
    print(f"TOPLULUK ({len(cks)} uye):")
    for c in cks:
        print(f"  {os.path.basename(c)}")

    # --- HANGI PARCALAR: mevcut olcum kumesiyle AYNI (karsilastirilabilirlik icin)
    ESKI, rap = olcum_kumesi.kume("results/_der_tam.pkl")
    olcum_kumesi.rapor_bas(rap)
    hedef = {r["pid"]: r for r in ESKI}
    E = {p: (jf, s) for m, p, jf, s in eligible()}
    sira = [p for p in hedef if p in E]
    if a.sinir:
        sira = sira[:a.sinir]
    print(f"\n{len(sira)} parca turetilecek", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = [load_any(c, dev=dev)[:2] for c in cks]
    OUT, hata = [], 0
    t0 = time.time()
    for k, pid in enumerate(sira, 1):
        if k % 20 == 0:
            print(f"  {k}/{len(sira)}  {time.time()-t0:.0f}s  hata={hata}", flush=True)
        jf, stp = E[pid]
        eski = hedef[pid]
        try:
            Vr, Fr = step_to_mesh(stp)
            V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
            V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
            pbs = []
            for model, meta in models:
                _, pb = D_.predict(model, meta, V, F, device=dev,
                                   op_cache_dir=f"results/step_infer/ops_k{int(meta.get('k_eig',64))}",
                                   return_probs=True)
                pbs.append(np.asarray(pb, float))
            cps, probs, _, uyeler = RC.adaylari_uret(V, F, pbs, stp, cfg=cfg)
            if not cps:
                P = np.zeros((0, 3)); Pd = np.zeros((0, 3)); X = None; XR = None
            else:
                P = np.array([c["point"] for c in cps], float)
                Pd = np.array([c["direction"] for c in cps], float)
                X = wire_gate.feats_for(V, F, probs, cps, CE, CT, step_path=stp)
                try:
                    XR = zengin(V, F, probs, cps, _normaller(V, F))
                except Exception as e:
                    print(f"    {pid} zengin: {type(e).__name__}"); XR = None
            # --- GT'yi STEP cercevesine tasi (eski onbellekle AYNI yontem)
            j = json.load(io.open(jf, encoding="utf-8-sig"))
            g = j.get("ConnectionPoints") or []
            G = np.array([[c["Point"][q] for q in "XYZ"] for c in g], float)
            Gd = np.array([[c["InsertDirection"][q] for q in "XYZ"] for c in g], float)
            Gd = Gd / (np.linalg.norm(Gd, axis=1, keepdims=True) + 1e-9)
            Vj = np.array([[q["X"], q["Y"], q["Z"]] for q in j["Graphic3d"]["Points"]], float)
            R, t_, _ = cad_eval.align_frames(Vr, Vj)
            G = (G - t_) @ R; Gd = Gd @ R
            OUT.append({"pid": pid, "mfg": eski["mfg"], "geo": eski["geo"],
                        "kume": eski.get("kume"), "diag": float(np.linalg.norm(V.max(0) - V.min(0))),
                        "n": len(G), "P": P, "Pd": Pd, "X": X, "XR": XR,
                        "G": G, "Gd": Gd, "UYE": uyeler})
        except Exception as e:
            hata += 1
            print(f"    {pid}: {type(e).__name__}: {e}")
    with open(a.cikti, "wb") as f:
        pickle.dump(OUT, f)
    nx = sum(1 for r in OUT if r["X"] is None)
    nxr = sum(1 for r in OUT if r["XR"] is None)
    print(f"\n{len(OUT)} kayit -> {a.cikti}")
    print(f"  X yok: {nx} | XR yok: {nxr} | hata: {hata}")
    print(f"  GT toplam: {sum(r['n'] for r in OUT)} | aday toplam: {sum(len(r['P']) for r in OUT)}")
    with io.open(a.cikti.replace(".pkl", "_koken.json"), "w", encoding="utf-8") as f:
        json.dump({"ckpt": cks, "n_parca": len(OUT), "X_yok": nx, "XR_yok": nxr,
                   "hata": hata, "postproc": cfg.get("prediction_postproc", {}),
                   "robot_min_votes": cfg.get("robot_min_votes"),
                   "conn_promote": cfg.get("current_product", {}).get("params", {}).get("conn_promote")},
                  f, indent=1)
    print(f"  koken -> {a.cikti.replace('.pkl','_koken.json')}")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
