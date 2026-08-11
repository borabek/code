# -*- coding: utf-8 -*-
"""ADET KAHINI: yapisal adet karari esikli karari ne kadar gecer? (plan md 4-5)

Urun su an gate skoruna MUTLAK esik (0.40/0.35) uygulayarak parca basina kac CP
verecegine karar veriyor. Plan bunun yerine "yapisal NULL/adet karari" istiyor.
Bir seyi INSA ETMEDEN once TAVANINI olcuyoruz: adet MUKEMMEL bilinseydi
(gercek N, gate skoruna gore top-N) ne olurdu?

Bu bir TAVAN: gercek bir adet tahmincisi bunun ALTINDA kalir. Tavan esikli
karari gecmiyorsa madde 4-5 OLUDUR ve insa edilmez.

KANONIK girdiler (G7BIRLESIK + gate v6), MIKRO toplama, D7 (=DEV, 835 parca).
"""
import collections, json, os, sys
import numpy as np
import makbuz_hash
os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"]="1"; os.environ["WG_TOPO"]="1"; os.environ["WG_ZENGIN"]="1"
sys.path.insert(0, ".")
import wire_gate, kanonik_d7 as K
from p1c_esik import maske
from sina_kume import esle_macar

gate = K.gate_yukle()
d7p = json.load(open("results/d7_sinav_kumesi.json"))["pidler"]
k7 = K.yukle(d7p)

kol = collections.defaultdict(list)
adet_hata = []
for pid, r in k7.items():
    X = K.x58(r); G = np.asarray(r.get("G", []), float)
    if X is None or not len(G):
        continue
    P = np.asarray(r["P"], float); D = np.asarray(r["Pd"], float)
    Gd = np.asarray(r["Gd"], float); dg = r["diag"]
    gs = np.asarray(wire_gate.karar_skoru(gate, X), float)
    N = len(G)

    def ol(sel):
        Ps, Ds = (P[sel], D[sel]) if np.any(sel) else (P[:0], D[:0])
        tp, fp, fn = esle_macar(Ps, Ds, G, Gd, dg, K.YANAL, K.ACI, False,
                                isaretli=True)[:3]
        return (N, tp, fp, fn)

    # 1) URUN: mutlak esik
    m = maske(gs, 0.40, 0.30)
    kol["esik (urun)"].append(ol(m))
    adet_hata.append(abs(int(m.sum()) - N))
    # 2) ADET KAHINI: gercek N, gate skoruna gore top-N
    o = np.zeros(len(gs), bool)
    o[np.argsort(-gs)[:min(N, len(gs))]] = True
    kol["adet kahini (top-N)"].append(ol(o))
    # 3) TAVAN UST SINIRI: adet kahini + MUKEMMEL siralama (skor yerine
    #    dogru olanlari sec) -- adet karari ile SIRALAMA'yi ayirir
    tp_i = esle_macar(P, D, G, Gd, dg, K.YANAL, K.ACI, False, isaretli=True)
    kol["tum havuz (esiksiz)"].append(ol(np.ones(len(gs), bool)))

print(f"D7 {len(kol['esik (urun)'])} parca | adet MAE {np.mean(adet_hata):.2f} "
      f"(medyan {np.median(adet_hata):.0f})")
res = {}
for ad, rows in kol.items():
    res[ad] = K.mikro(rows)
    print(f"  {ad:<22} MIKRO {res[ad]:.4f}")
d = res["adet kahini (top-N)"] - res["esik (urun)"]
print(f"\nADET KAHININ KAZANCI: {d:+.4f}")
print("KARAR: " + ("madde 4-5 ACIK -- yapisal adet karari insa edilir"
                   if d >= 0.02 else
                   "madde 4-5 OLU -- MUKEMMEL adet bile esigi gecmiyor"))
json.dump({"damga": makbuz_hash.damga(), "mikro": res, "kazanc": d,
           "adet_MAE": float(np.mean(adet_hata)), "n": len(adet_hata),
           "not": "TAVAN olcumu: gercek adet kahin olarak verildi. Gercek bir "
                  "adet tahmincisi bunun ALTINDA kalir. D7=DEV."},
          open("results/adet_kahini.json", "w"), indent=1)
