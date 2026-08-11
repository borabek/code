# -*- coding: utf-8 -*-
"""TEZGAH2: kalan kollari AYNI ADIL TABANA karsi olcen ortak altyapi.

NEDEN: G3'te tabani dagitilan gate'i yukleyerek olctum ve 0.8686 cikti -- cunku o gate'in
egitim verisi olcum kumesinin ~391 parcasini iceriyor (SIZINTI). Duzeltince taban tam
olarak mansetin 0.7584'u oldu. Bu tezgah o duzeltilmis tabani TEK YERDE kurar ki her kol
ayni zeminde olculsun ve hata tekrarlanmasin.

TABAN: eski turetme (_der_tam.pkl) + eski korpus (zengin_parite_w2.npz) ama OLCUM
GRUPLARI CIKARILARAK egitilmis gate -> tespit 0.7584 / robot 0.5893 (manset ile birebir).
"""
import io
import json
import os
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ESKI_NPZ = "results/zengin_parite_w2.npz"
DER_YOL = "results/_der_tam.pkl"


def yukle():
    """(DER, gate, ad) -- adil taban gate'i ve olcum kumesi."""
    import olcum_kumesi
    import wire_gate
    from sklearn.ensemble import RandomForestClassifier
    DER, rap = olcum_kumesi.kume(DER_YOL)
    olcum_kumesi.rapor_bas(rap)
    d = np.load(ESKI_NPZ, allow_pickle=True)
    X = (np.hstack([np.asarray(d["X22"], float), np.asarray(d["XR"], float)])
         if "X22" in d.files else np.asarray(d["X"], float))
    y = np.asarray(d["y"]); pid = np.array([str(x) for x in d["pids"]])
    mfg = np.array([str(x) for x in d["mfg"]])
    with io.open("results/_strict_geometry_keys.json", encoding="utf-8") as f:
        gk = json.load(f)
    tg = {r["geo"] for r in DER}
    keep = ~np.isin(np.array([gk.get(x, "yok:" + x) for x in pid]), list(tg))
    Z = np.zeros((len(X), X.shape[1] * 2))
    for u in np.unique(pid):
        i = np.where(pid == u)[0]
        Z[i] = wire_gate.parca_ici(X[i], "zskor")
    gate = {"clf": RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                          random_state=0).fit(Z[keep], y[keep]),
            "n_feat": Z.shape[1], "donusum": "zskor"}
    try:
        import gate_tezgah as T
        ad = T.yukle()["ad"]
    except Exception:
        ad = None
    return DER, gate, {"Z": Z, "y": y, "pid": pid, "mfg": mfg, "keep": keep, "ad": ad}


def puanla(DER, gate, karar=None, duzelt=True):
    """Urunun karar yolu. `karar(skor, r, X)` verilirse varsayilan maskenin YERINE gecer."""
    import wire_gate
    from sina_kume import esle
    det, rob, gg = [], [], []
    for r in DER:
        P = np.zeros((0, 3)); Pd = np.zeros((0, 3))
        if r["X"] is not None and r.get("XR") is not None:
            X = np.hstack([r["X"], r["XR"]])
            if X.shape[1] * 2 == gate["n_feat"]:
                sk = wire_gate.karar_skoru(gate, X)
                k = (wire_gate.karar_maskesi(sk) if karar is None else karar(sk, r, X))
                if k.any():
                    P = np.asarray(r["P"], float)[k].copy()
                    Pd = np.asarray(r["Pd"], float)[k].copy()
                    if duzelt:
                        c = [{"point": P[i], "direction": Pd[i]} for i in range(len(P))]
                        c = wire_gate.pose_duzelt(X[k], c)
                        c = wire_gate.aci_duzelt(X[k], c)
                        if r.get("UYE"):
                            c = wire_gate.uye_yonu_sec(X[k], c, r["UYE"])
                        P = np.array([x["point"] for x in c], float)
                        Pd = np.array([x["direction"] for x in c], float)
        rj = "cok" if r["n"] >= 8 else "dusuk"
        G = np.asarray(r["G"], float); Gd = np.asarray(r["Gd"], float)
        det.append((rj,) + esle(P, Pd, G, Gd, r["diag"], 0.0, 180.0, True))
        rob.append((rj,) + esle(P, Pd, G, Gd, r["diag"], 2.0, 10.0, False))
        gg.append(r["geo"])
    return det, rob, gg


def ga(taban_rows, aday_rows, gg, n=3000):
    import olcum_kumesi
    from sina_kume import f1w
    fn = lambda rows: f1w([q for _, q in rows]) - f1w([p for p, _ in rows])
    _, lo, hi = olcum_kumesi.grup_bootstrap(list(zip(taban_rows, aday_rows)), gg, fn, n=n)
    return lo, hi


def bas(ad, det, rob, taban=None, gg=None):
    from sina_kume import f1w
    s = f"{ad:<30}{f1w(det):>9.4f}{f1w(rob):>9.4f}"
    if taban is not None and gg is not None:
        lo, hi = ga(taban[0], det, gg)
        s += f"   {f1w(det)-f1w(taban[0]):+.4f}  GA[{lo:+.4f},{hi:+.4f}] " \
             f"{'GERCEK' if (lo > 0 or hi < 0) else 'gurultu'}"
    print(s, flush=True)
    return f1w(det), f1w(rob)
