# -*- coding: utf-8 -*-
"""F2-11: PARITE KAPISI -- JSON Graphic3d agi, STEP remesh'inin yerini tutar mi?

NEDEN BU MADDE HAYATI: korpusta 4706 parca var ama **2759'unun STEP'i YOK** (A-B'nin
822'si, CWT'nin 785'i, ABB 324, KLM 101 TAMAMEN). Ve WSCAD'den indirmek KESIN KAPALI
([[wscad-indirme-kesin-kapali]]: 24 denemede 0 basari, sebep 3D VERI YOKLUGU).
Hepsinde `Graphic3d` UCGEN AGI VAR (Points + Indices). Yani bu veriyi kullanmanin TEK
yolu JSON agini STEP remesh'inin yerine koyabilmek.

TEZ CIZGISI KORUNUR: her iki yol da AYNI uniform izotropik ~6000 remesh'ten gecer --
zaten tezin domain-gap cozumu budur ([[thesis-remesh-solution]]). Ag, `v_o` tanimi ve
aday uretimi DEGISMEZ. Degisen tek sey MESH KAYNAGI.

UC KOL (ayristirma kritik):
    A  STEP agi   + STEP ozellikleri   = bugunku urun
    B  JSON agi   + STEP ozellikleri   = YALNIZ mesh etkisi
    C  JSON agi   + STEP YOK           = 2759 parcanin GERCEK kosulu
        (brep_axes ve WG_FIZ_FEATS uretilemez -- o blok tespitin +0.0704'uydu)

YAN FAYDA: B ve C'de GT'ye `align_frames` UYGULANMAZ -- mesh zaten JSON cercevesinde,
GT de oyle. A'daki hizalama artigi (bir hata kaynagi) burada YOK.

KILL (onceden yazildi): aday recall farki <= 0.01 VE uretici basina tespit F1 farki
<= 0.015 -> JSON agi kullanilabilir. C kolu ayrica raporlanir (B-rep bedeli).
"""
import argparse
import io
import json
import os
import pickle
import sys
import time

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"; os.environ["WG_TOPO"] = "1"; os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CIKTI = "results/_der_json_{}.pkl"


def json_mesh(jf):
    """JSON Graphic3d -> (V, F). Indices ucgen listesi (3'er)."""
    j = json.load(io.open(jf, encoding="utf-8-sig"))
    g = j.get("Graphic3d") or {}
    P = g.get("Points") or []
    I = g.get("Indices") or []
    if not P or len(I) < 3:
        return None, None, j
    V = np.array([[q["X"], q["Y"], q["Z"]] for q in P], float)
    F = np.array(I, np.int64).reshape(-1, 3)
    F = F[(F >= 0).all(1) & (F < len(V)).all(1)]
    return V, F, j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kol", choices=["B", "C"], default="B")
    ap.add_argument("--sinir", type=int, default=0)
    a = ap.parse_args()

    import protokol
    protokol.tez_dogrula()
    import torch
    import diffusionnet as D_
    import olcum_kumesi
    import robot_cp as RC
    import thesis_remesh
    import wire_gate
    from big_arbiter import eligible
    from build_zengin_parite import _normaller, zengin
    from connector_constants import CABLE_ENTRY as CE, CONTACT as CT
    from infer_step_cp import load_any

    with io.open("cp_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    ESKI, rap = olcum_kumesi.kume("results/_der_tam.pkl")
    olcum_kumesi.rapor_bas(rap)
    hedef = {r["pid"]: r for r in ESKI}
    E = {p: (jf, s) for m, p, jf, s in eligible()}
    sira = [p for p in hedef if p in E]
    if a.sinir:
        sira = sira[:a.sinir]
    step_ver = (a.kol == "B")
    print(f"\nKOL {a.kol}: JSON agi + {'STEP ozellikleri' if step_ver else 'STEP YOK'}"
          f" | {len(sira)} parca", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cks = cfg["current_product"].get("checkpoints") or cfg["robot_vote2_checkpoints"]
    models = [load_any(c, dev=dev)[:2] for c in cks]
    OUT, hata, bos = [], 0, 0
    t0 = time.time()
    for k, pid in enumerate(sira, 1):
        if k % 20 == 0:
            print(f"  {k}/{len(sira)}  {time.time()-t0:.0f}s  hata={hata} bos={bos}", flush=True)
        jf, stp = E[pid]
        eski = hedef[pid]
        try:
            Vj, Fj, j = json_mesh(jf)
            if Vj is None or len(Fj) < 4:
                bos += 1
                continue
            V, F = thesis_remesh.remesh_uniform(Vj, Fj, target=6000)
            V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
            pbs = []
            for model, meta in models:
                _, pb = D_.predict(model, meta, V, F, device=dev,
                                   op_cache_dir=f"results/step_infer/ops_json_k{int(meta.get('k_eig',64))}",
                                   return_probs=True)
                pbs.append(np.asarray(pb, float))
            sp = stp if step_ver else None
            cps, probs, _, uyeler = RC.adaylari_uret(V, F, pbs, sp, cfg=cfg)
            if not cps:
                P = np.zeros((0, 3)); Pd = np.zeros((0, 3)); X = None; XR = None
            else:
                P = np.array([c["point"] for c in cps], float)
                Pd = np.array([c["direction"] for c in cps], float)
                X = wire_gate.feats_for(V, F, probs, cps, CE, CT, step_path=sp)
                try:
                    XR = zengin(V, F, probs, cps, _normaller(V, F))
                except Exception:
                    XR = None
            # GT: JSON cercevesinde ZATEN -- hizalama YOK (A kolundaki artik burada yok)
            g = j.get("ConnectionPoints") or []
            G = np.array([[c["Point"][q] for q in "XYZ"] for c in g], float)
            Gd = np.array([[c["InsertDirection"][q] for q in "XYZ"] for c in g], float)
            Gd = Gd / (np.linalg.norm(Gd, axis=1, keepdims=True) + 1e-9)
            OUT.append({"pid": pid, "mfg": eski["mfg"], "geo": eski["geo"],
                        "kume": eski.get("kume"),
                        "diag": float(np.linalg.norm(V.max(0) - V.min(0))),
                        "n": len(G), "P": P, "Pd": Pd, "X": X, "XR": XR,
                        "G": G, "Gd": Gd, "UYE": uyeler,
                        "json_tepe": len(Vj), "json_yuz": len(Fj)})
        except Exception as e:
            hata += 1
            if hata <= 3:
                print(f"    {pid}: {type(e).__name__}: {str(e)[:80]}")
    yol = CIKTI.format(a.kol)
    with open(yol, "wb") as f:
        pickle.dump(OUT, f)
    print(f"\n{len(OUT)} kayit -> {yol} | hata {hata} | bos mesh {bos}")
    print(f"  GT toplam {sum(r['n'] for r in OUT)} | aday toplam {sum(len(r['P']) for r in OUT)}")
    jt = np.array([r["json_tepe"] for r in OUT])
    print(f"  JSON agi tepe sayisi: medyan {np.median(jt):.0f} | %10 {np.percentile(jt,10):.0f}"
          f" | max {jt.max()}")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
