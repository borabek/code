# -*- coding: utf-8 -*-
"""MADDE 6: ayni FIZIKSEL AGIZA yigilan adaylari bastirmak F1'i artirir mi?

Urun zincirinde iki aday ayni silindire/agza dusuyorsa ikisi de cikiyor; Macar
bire-bir eslestirdigi icin biri zorunlu FP. p5-v2 bunu `agiz_kimlik` uzerinden
bipartite ile cozuyordu; burada AYNI kisiti URUN zincirine tek basina, bir
NMS olarak uyguluyoruz (her agizdan en yuksek gate skorlusu kalir).

Uc yaricap denenir; ayrica agiz kimligi yerine SAF mesafe-NMS'i de olculur.
KANONIK girdiler, MIKRO, D7 (=DEV).
"""
import collections, json, os, pickle, sys
import numpy as np
import makbuz_hash
os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"]="1"; os.environ["WG_TOPO"]="1"; os.environ["WG_ZENGIN"]="1"
sys.path.insert(0, ".")
import wire_gate, kanonik_d7 as K
from p1c_esik import maske
from sina_kume import esle_macar

gate = K.gate_yukle()
cy7 = pickle.load(open("results/_d7_silindirler.pkl", "rb"))
k7 = K.yukle(json.load(open("results/d7_sinav_kumesi.json"))["pidler"])


def agiz_kimlikleri(P, cyl, R):
    """Her adayi en yakin FIZIKSEL AGIZA baglar; yoksa -1 (bastirilmaz).

    Silindir kaydinin anahtarlari INGILIZCE: center/axis/radius/mouth_a/mouth_b.
    Kimlik silindir DEGIL AGIZ duzeyinde: ayni silindirin iki ucu AYRI agizdir,
    ikisine de birer aday dusebilir ve bu kalabalik SAYILMAZ.
    """
    kim = np.full(len(P), -1)
    if not cyl:
        return kim
    A = []
    for c in cyl:
        for u in ("mouth_a", "mouth_b"):
            if c.get(u) is not None:
                A.append(np.asarray(c[u], float))
    if not A:
        return kim
    A = np.asarray(A, float)
    for i, p in enumerate(P):
        d = np.linalg.norm(A - p, axis=1)
        j = int(np.argmin(d))
        if d[j] <= R:
            kim[i] = j
    return kim


def nms_kimlik(P, gs, cyl, R):
    kim = agiz_kimlikleri(P, cyl, R)
    tut = np.ones(len(P), bool)
    for j in set(kim.tolist()) - {-1}:
        idx = np.where(kim == j)[0]
        if len(idx) > 1:
            tut[idx] = False
            tut[idx[int(np.argmax(gs[idx]))]] = True
    return tut


def nms_mesafe(P, gs, r):
    sira = np.argsort(-gs); tut = np.ones(len(P), bool)
    for a in range(len(sira)):
        i = sira[a]
        if not tut[i]:
            continue
        for b in range(a + 1, len(sira)):
            j = sira[b]
            if tut[j] and np.linalg.norm(P[i] - P[j]) < r:
                tut[j] = False
    return tut


kol = collections.defaultdict(list)
for pid, r in k7.items():
    X = K.x58(r); G = np.asarray(r.get("G", []), float)
    if X is None or not len(G):
        continue
    P = np.asarray(r["P"], float); D = np.asarray(r["Pd"], float)
    Gd = np.asarray(r["Gd"], float); dg = r["diag"]
    gs = np.asarray(wire_gate.karar_skoru(gate, X), float)
    m = maske(gs, 0.40, 0.30)
    if not m.any():
        for ad in ("esik (urun)",):
            kol[ad].append((len(G), 0, 0, len(G)))
        continue
    Pm, Dm, gm = P[m], D[m], gs[m]
    cyl = cy7.get(pid)

    def ol(sel):
        tp, fp, fn = esle_macar(Pm[sel], Dm[sel], G, Gd, dg, K.YANAL, K.ACI,
                                False, isaretli=True)[:3]
        return (len(G), tp, fp, fn)

    kol["esik (urun)"].append(ol(np.ones(len(Pm), bool)))
    for R in (2.0, 4.0, 6.0):
        kol[f"agiz-NMS R={R}"].append(ol(nms_kimlik(Pm, gm, cyl, R)))
    for rr in (1.5, 3.0):
        kol[f"mesafe-NMS r={rr}"].append(ol(nms_mesafe(Pm, gm, rr)))

taban = K.mikro(kol["esik (urun)"])
print(f"D7 {len(kol['esik (urun)'])} parca\n")
res = {}
for ad in kol:
    res[ad] = K.mikro(kol[ad])
    d = res[ad] - taban
    print(f"  {ad:<18} MIKRO {res[ad]:.4f}  {d:+.4f}")
en = max((a for a in res if a != "esik (urun)"), key=lambda a: res[a])
print(f"\nEN IYI: {en} {res[en]-taban:+.4f}")
print("KARAR: " + ("madde 6 ACIK" if res[en] - taban >= 0.01 else "madde 6 OLU"))
json.dump({"damga": makbuz_hash.damga(), "mikro": res, "taban": taban,
           "en_iyi": en, "kazanc": res[en] - taban, "n": len(kol["esik (urun)"]),
           "not": "URUN zincirine tek basina NMS. D7=DEV. MIKRO."},
          open("results/kalabalik_bastir.json", "w"), indent=1)
