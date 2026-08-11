# -*- coding: utf-8 -*-
"""T4: GERCEK B-REP BOSLUK GRAFI -- denetimin hakli buldugu, HIC DENENMEMIS kol.

DOGRULANDI: eski "B-rep graf" kolu (s7_brep_graf.py) gercek bir graf KURMAMIS.
`graf_ozellik` 10 slot aciyor ama yalniz 0-7'yi dolduruyor; `g_agiz_cev` ve `g_yuz_alan`
kalici olarak SIFIR. Dahasi hicbir yerde yuz-kenar komsulugu yok -- yalniz silindir ve
duzlem AGIRLIK MERKEZLERI sayiliyordu. Yani boskuk grafi denenmemistir.

BURADA GERCEK TOPOLOJI CIKARILIYOR (gmsh/OCC):
    yuz -> sinir kenarlari            (getBoundary)
    kenar -> onu paylasan yuzler      (getAdjacencies)  <- GERCEK KOMSULUK
    yuz alani / kenar uzunlugu        (occ.getMass)
    yuz tipi                          (getType)

OZELLIKLER (8), her aday icin:
    bg_kanal_alan   adayla eseksenli silindirik yuzlerin toplam alani (kanal duvari)
    bg_kanal_boy    kanal duvari uzunlugu = alan / (2*pi*r)
    bg_agiz_cev     kanal duvarini sinirlayan kenarlarin toplam uzunlugu (AGIZ CEVRIMI)
    bg_dairesel     agiz cevriminin daireden sapmasi: cev^2 / (4*pi*alan)
    bg_kom_yuz      kanal duvariyla KENAR PAYLASAN yuz sayisi (gercek komsuluk derecesi)
    bg_kom_duzlem   bu komsulardan duzlem olanlarin sayisi
    bg_gecis        kanal duvarinin serbest sinir cevrimi sayisi (2 = gecen delik, 1 = kor)
    bg_yaricap      kanal yaricapi (dogrudan B-rep'ten, agdan degil)

Renk (plastik->metal) BILEREK YOK: `c_metal` 2026-07-31'de olculdu ve OLU cikti.

ONCE UCUZ PROB: olcum kumesinde (194 parca) grup-capraz aday duzeyi. Bar +0.01.
"""
import io
import json
import os
import pickle
import sys
import time

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

AD = ["bg_kanal_alan", "bg_kanal_boy", "bg_agiz_cev", "bg_dairesel",
      "bg_kom_yuz", "bg_kom_duzlem", "bg_gecis", "bg_yaricap"]
ONBELLEK = "results/bosluk_grafi.pkl"


def topoloji(step_path):
    """STEP'ten GERCEK yuz-kenar topolojisi. gmsh HER ZAMAN finalize edilir."""
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()
        Y = {}
        for dim, tag in gmsh.model.getEntities(2):
            try:
                tip = gmsh.model.getType(2, tag)
                cm = gmsh.model.occ.getCenterOfMass(2, tag)
                alan = float(gmsh.model.occ.getMass(2, tag))
                sinir = [abs(t) for _, t in gmsh.model.getBoundary([(2, tag)], oriented=False)]
                Y[tag] = {"tip": tip, "cm": np.array(cm, float), "alan": alan,
                          "kenar": sinir, "eksen": None, "r": None}
                if tip == "Cylinder":
                    b = gmsh.model.getParametrizationBounds(2, tag)
                    us = np.linspace(b[0][0], b[1][0], 8)
                    vs = np.linspace(b[0][1], b[1][1], 4)
                    par = np.array([[u, v] for u in us for v in vs]).ravel()
                    P = np.asarray(gmsh.model.getValue(2, tag, par), float).reshape(-1, 3)
                    # eksen = en buyuk tekil vektor
                    Q = P - P.mean(0)
                    _, _, V = np.linalg.svd(Q, full_matrices=False)
                    ax = V[0] / (np.linalg.norm(V[0]) + 1e-9)
                    rel = P - P.mean(0)
                    off = rel - (rel @ ax)[:, None] * ax
                    Y[tag]["eksen"] = ax
                    Y[tag]["r"] = float(np.median(np.linalg.norm(off, axis=1)))
            except Exception:
                continue
        K = {}
        for dim, tag in gmsh.model.getEntities(1):
            try:
                K[tag] = float(gmsh.model.occ.getMass(1, tag))
            except Exception:
                K[tag] = 0.0
        # KENAR -> YUZ komsulugu (GERCEK graf kenarlari)
        kom = {}
        for t, y in Y.items():
            for e in y["kenar"]:
                kom.setdefault(e, []).append(t)
        return Y, K, kom
    finally:
        gmsh.finalize()


def ozellik(P, Pd, Y, K, kom):
    """Her aday icin 8 bosluk-grafi sutunu."""
    F = np.zeros((len(P), len(AD)))
    sil = [(t, y) for t, y in Y.items() if y["tip"] == "Cylinder" and y["eksen"] is not None]
    if not sil:
        return F
    C = np.array([y["cm"] for _, y in sil])
    A = np.array([y["eksen"] for _, y in sil])
    R = np.array([y["r"] for _, y in sil])
    TG = [t for t, _ in sil]
    for i in range(len(P)):
        p = P[i]; d = Pd[i] / (np.linalg.norm(Pd[i]) + 1e-9)
        rel = C - p
        al = (rel * A).sum(1)
        off = np.linalg.norm(rel - al[:, None] * A, axis=1)
        # ESEKSENLI: eksen paralel + eksen cizgisine yakin + makul mesafede
        es = (np.abs(A @ d) >= np.cos(np.radians(20.0))) & (off <= 4.0) & (np.abs(al) <= 25.0)
        if not es.any():
            continue
        idx = np.where(es)[0]
        alan = float(sum(Y[TG[j]]["alan"] for j in idx))
        r = float(np.median(R[idx])) if len(idx) else 0.0
        F[i, 0] = alan
        F[i, 1] = alan / max(2 * np.pi * r, 1e-6)
        kenarlar = set()
        for j in idx:
            kenarlar |= set(Y[TG[j]]["kenar"])
        cev = float(sum(K.get(e, 0.0) for e in kenarlar))
        F[i, 2] = cev
        F[i, 3] = cev ** 2 / max(4 * np.pi * alan, 1e-6)
        # GERCEK KOMSULUK: bu kenarlari paylasan BASKA yuzler
        komsu = set()
        for e in kenarlar:
            for t in kom.get(e, []):
                if t not in [TG[j] for j in idx]:
                    komsu.add(t)
        F[i, 4] = float(len(komsu))
        F[i, 5] = float(sum(1 for t in komsu if Y[t]["tip"] == "Plane"))
        # GECIS/KOR: kanal duvarinin SERBEST kenarlari (yalniz bir yuze ait olanlar)
        serbest = [e for e in kenarlar if len(kom.get(e, [])) <= 1]
        F[i, 6] = float(len(serbest))
        F[i, 7] = r
    return F


def main():
    import gate_tezgah as T
    import olcum_kumesi
    from big_arbiter import eligible
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold
    from gece_kilit import bekci

    bekci("t4 baslangic")
    D = T.yukle()
    olcum_kumesi.rapor_bas(D["rap"])
    stp = {p: s for m, p, jf, s in eligible()}

    CACHE = {}
    if os.path.exists(ONBELLEK):
        with open(ONBELLEK, "rb") as f:
            CACHE = pickle.load(f)
        print(f"onbellekten {len(CACHE)} parca", flush=True)

    RX, RY, RG, RP, BG = [], [], [], [], []
    t0 = time.time(); basarili = 0
    for k, r in enumerate(D["DER"], 1):
        if k % 20 == 0:
            print(f"  {k}/{len(D['DER'])}  {time.time()-t0:.0f}s", flush=True)
            with open(ONBELLEK, "wb") as f:
                pickle.dump(CACHE, f)
            bekci(f"t4 {k}")
        if r["X"] is None or r.get("XR") is None or not len(r["G"]):
            continue
        X58 = np.hstack([r["X"], r["XR"]])
        P = np.asarray(r["P"], float); Pd = np.asarray(r["Pd"], float)
        if r["pid"] not in CACHE:
            try:
                CACHE[r["pid"]] = topoloji(stp[r["pid"]])
            except Exception as e:
                print(f"    {r['pid']}: {type(e).__name__}")
                CACHE[r["pid"]] = None
        veri = CACHE[r["pid"]]
        FB = ozellik(P, Pd, *veri) if veri is not None else np.zeros((len(P), len(AD)))
        basarili += int(veri is not None)
        G = np.asarray(r["G"], float); Gd = np.asarray(r["Gd"], float)
        diff = P[:, None, :] - G[None, :, :]
        al = (diff * Gd[None, :, :]).sum(-1)
        pe = np.linalg.norm(diff - al[..., None] * Gd[None, :, :], axis=-1)
        pe = np.where(np.abs(al) > 40, np.inf, pe)
        tt = max(3.0, 0.06 * float(r["diag"]))
        yy = np.zeros(len(P), int); up, ug = set(), set()
        for d_, a_, b_ in sorted((pe[a, b], a, b) for a in range(len(P))
                                 for b in range(len(G))):
            if not np.isfinite(d_) or d_ > tt or a_ in up or b_ in ug:
                continue
            up.add(a_); ug.add(b_); yy[a_] = 1
        for i in range(len(P)):
            RX.append(X58[i]); BG.append(FB[i]); RY.append(yy[i])
            RG.append(r["geo"]); RP.append(r["pid"])
    with open(ONBELLEK, "wb") as f:
        pickle.dump(CACHE, f)
    RX = np.array(RX); BG = np.array(BG); RY = np.array(RY)
    RG = np.array(RG); RP = np.array(RP)
    print(f"\n{len(RY)} aday | pozitif {RY.mean():.1%} | topoloji cikan parca {basarili}")

    from t1_uretici_disi import auc_mw
    print(f"\n{'sutun':<16}{'AUC':>8}{'TP ort':>11}{'FP ort':>11}{'sifir-disi':>11}")
    for j, a in enumerate(AD):
        v = BG[:, j]
        print(f"{a:<16}{auc_mw(v, RY.astype(bool)):>8.3f}{v[RY==1].mean():>11.3f}"
              f"{v[RY==0].mean():>11.3f}{float((v!=0).mean()):>11.1%}")

    def olc(M):
        o = np.zeros(len(RY))
        for tr, te in GroupKFold(n_splits=5).split(M, RY, RG):
            o[te] = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                           random_state=0).fit(M[tr], RY[tr]).predict_proba(M[te])[:, 1]
        m = np.zeros(len(o), bool)
        for u in np.unique(RP):
            i = RP == u; vv = o[i]
            m[i] = (vv >= 0.5 * max(vv.max(), 1e-9)) & (vv >= 0.25)
        tp = int((RY.astype(bool) & m).sum()); fp = int((~RY.astype(bool) & m).sum())
        fn = int((RY.astype(bool) & ~m).sum())
        p_ = tp / max(tp + fp, 1); r_ = tp / max(tp + fn, 1)
        return 2 * p_ * r_ / max(p_ + r_, 1e-9)
    a58 = olc(RX); a66 = olc(np.hstack([RX, BG]))
    print(f"\nADAY DUZEYI: 58 sutun {a58:.4f} | 58+bosluk-grafi {a66:.4f} | fark {a66-a58:+.4f}")
    gecti = (a66 - a58) >= 0.01
    print(f"KARAR: {'SINYAL VAR -> T5 uctan uca sinava' if gecti else 'SINYAL YOK -> T4 KAPANIR'}")
    np.savez("results/t4_bosluk.npz", BG=BG, RY=RY, RG=RG, RP=RP)
    with io.open("results/t4_bosluk.json", "w", encoding="utf-8") as f:
        json.dump({"aday_58": float(a58), "aday_66": float(a66), "fark": float(a66 - a58),
                   "gecti": bool(gecti), "topoloji_parca": basarili,
                   "auc": {a: float(auc_mw(BG[:, j], RY.astype(bool)))
                           for j, a in enumerate(AD)}}, f, indent=1)
    print("makbuz -> results/t4_bosluk.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
