# -*- coding: utf-8 -*-
"""HAVUZ GERI CAGIRMA: temsilin gercek olcusu (F1 DEGIL).

Onceki kosuda "aday kahini" diye yazdigim sutun kahin DEGILDI: gate'siz havuzun
F1'iydi ve genis havuz fazla adaylar yuzunden FP cezasi yiyordu -- tam olarak
`gate-once-poz-sonra-tavani-kirpiyor` kaydindaki 2 numarali hata. Bir havuzun
temsil gucu GERI CAGIRMA ile olculur: GT'nin yuzde kaci havuzda ULASILABILIR.
Bire-bir Macar, TESPIT toleransi, aci serbest.
"""
import collections, json, os, pickle, sys
import numpy as np
import makbuz_hash
os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, ".")
from sina_kume import esle_macar

d7 = set(map(str, json.load(open("results/d7_sinav_kumesi.json"))["pidler"]))
out = {}
for ad, f in (("G7 (kanonik)", "results/_der_yeni_G7BIRLESIK.pkl"),
              ("g10 (A3-A4)", "results/_der_yeni_g10.pkl")):
    R = [r for r in pickle.load(open(f, "rb")) if str(r["pid"]) in d7]
    TP = FN = 0; nA = 0; per = collections.defaultdict(lambda: [0, 0])
    for r in R:
        G = np.asarray(r.get("G", []), float)
        if not len(G):
            continue
        P = np.asarray(r["P"], float); D = np.asarray(r["Pd"], float)
        tp, fp, fn = esle_macar(P, D, G, np.asarray(r["Gd"], float), r["diag"],
                                max(3.0, 0.06 * r["diag"]), 180.0, True)[:3]
        TP += tp; FN += fn; nA += len(P)
        a = per[r["mfg"]]; a[0] += tp; a[1] += fn
    rc = TP / max(TP + FN, 1)
    pm = {m: a[0] / max(a[0] + a[1], 1) for m, a in per.items()}
    out[ad] = {"recall": rc, "aday_sayisi": nA, "n_parca": len(R),
               "aday_per_parca": nA / max(len(R), 1), "marka": pm}
    print(f"{ad:<14} havuz recall {rc:.4f} | toplam aday {nA:>6} "
          f"({nA/max(len(R),1):.1f}/parca)")
a, b = out["G7 (kanonik)"], out["g10 (A3-A4)"]
print(f"\nRECALL FARKI {b['recall']-a['recall']:+.4f} | "
      f"ADAY SAYISI {b['aday_per_parca']/max(a['aday_per_parca'],1e-9):.2f}x")
print(f"\n{'marka':<8} {'G7':>8} {'g10':>8} {'fark':>8}")
for m in sorted(a["marka"], key=lambda k: -a["marka"][k]):
    print(f"  {m:<7} {a['marka'][m]:>7.4f} {b['marka'].get(m,0):>8.4f} "
          f"{b['marka'].get(m,0)-a['marka'][m]:>+8.4f}")
json.dump({"damga": makbuz_hash.damga(), "sonuc": out,
           "not": "HAVUZ GERI CAGIRMA (F1 DEGIL). D7 marka-disi, tespit tolerans, "
                  "aci serbest, bire-bir Macar."},
          open("results/havuz_recall_d7.json", "w"), indent=1)
