# -*- coding: utf-8 -*-
"""PARCA DUZEYINDE YONLENDIRICI: B-rep'i HER parcada degil, GEREKEN parcada kullan.

GOZLEM: B-rep havuzu recall'u markalara gore cok farkli aciyor (CWT +0.2984,
KLM +0.2651, EFX +0.2331; ELMEX/C3 +0.0000). Seg havuzunun zaten iyi oldugu
parcalarda B-rep yalnizca CELDIRICI ekliyor -- hibrit gate'in tabanin altinda
kalmasinin sebebi bu olabilir.

Bu sonda once TAVANI olcer: her parcada IKI havuzdan iyisini secen MUKEMMEL
yonlendirici ne verir? Tavan tabanin belirgin ustunde degilse parca-duzeyi
yonlendirici INSA EDILMEZ.

Ayrica yonlendirilebilirlik icin UCUZ sinyaller de olculur (segmentasyonun kendi
guveni, aday sayisi, silindir sayisi) -- tavan varsa hangi sinyalle yakalanir.
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

OZ = "results/_brep_oz"
bmod = pickle.load(open("results/brep_only_gate.pkl", "rb"))
g6 = K.gate_yukle()
te = []
for f in sorted(os.listdir(OZ)):
    if f.startswith("d7_") and f.endswith(".npz"):
        z = np.load(f"{OZ}/{f}")
        te.append({"pid": f[3:-4], "X": np.asarray(z["X"], float), "P": z["P"],
                   "D": z["D"], "kaynak": z["kaynak"]})
kay = K.yukle([d["pid"] for d in te])
for d in te:
    r = kay[d["pid"]]
    d.update({"mfg": r["mfg"], "G": np.asarray(r["G"], float),
              "Gd": np.asarray(r["Gd"], float), "diag": r["diag"]})
print(f"D7 {len(te)} parca", flush=True)


def ciktilar(d, brep, b_esik=0.70):
    ms = d["kaynak"] == 0
    Xs = d["X"][ms]
    if len(Xs) < 2:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    ss = np.asarray(wire_gate.karar_skoru(g6, Xs), float)
    ks = ss >= 0.30
    P = list(d["P"][ms][ks]); D = list(d["D"][ms][ks]); S = list(ss[ks])
    if brep:
        Xb = d["X"][~ms]
        if len(Xb) >= 2:
            sb = np.asarray(wire_gate.karar_skoru(bmod, Xb), float)
            kb = sb >= b_esik
            P += list(d["P"][~ms][kb]); D += list(d["D"][~ms][kb]); S += list(sb[kb])
    P = np.asarray(P, float).reshape(-1, 3); D = np.asarray(D, float).reshape(-1, 3)
    S = np.asarray(S, float)
    if len(P) > 1:
        nm = wire_gate.kalabalik_maskesi(P, S)
        P, D, S = P[nm], D[nm], S[nm]
    return P, D, S


def say(P, D, d):
    return esle_macar(P, D, d["G"], d["Gd"], d["diag"], K.YANAL, K.ACI, False,
                      isaretli=True)[:3]


kol = {"TEZ-SAF": [], "HIBRIT": [], "KAHIN (parca basina iyisi)": []}
sinyal = []
for d in te:
    Ps, Ds, _ = ciktilar(d, False)
    Ph, Dh, _ = ciktilar(d, True)
    a = say(Ps, Ds, d); b = say(Ph, Dh, d)
    f = lambda t: 2*t[0]/max(2*t[0]+t[1]+t[2], 1)          # noqa: E731
    kol["TEZ-SAF"].append(a); kol["HIBRIT"].append(b)
    iyi = a if f(a) >= f(b) else b
    kol["KAHIN (parca basina iyisi)"].append(iyi)
    ms = d["kaynak"] == 0
    Xs = d["X"][ms]
    ss = np.asarray(wire_gate.karar_skoru(g6, Xs), float) if len(Xs) >= 2 else np.zeros(1)
    sinyal.append({"pid": d["pid"], "mfg": d["mfg"], "brep_iyi": int(f(b) > f(a)),
                   "esit": int(abs(f(b)-f(a)) < 1e-9),
                   "n_seg": int(ms.sum()), "n_brep": int((~ms).sum()),
                   "seg_max": float(ss.max()), "seg_ort": float(ss.mean()),
                   "n_gt": int(len(d["G"])), "diag": float(d["diag"])})

for ad, v in kol.items():
    TP = sum(x[0] for x in v); FP = sum(x[1] for x in v); FN = sum(x[2] for x in v)
    print(f"{ad:<28} robot {2*TP/max(2*TP+FP+FN,1):.4f}", flush=True)
n_b = sum(s["brep_iyi"] for s in sinyal); n_e = sum(s["esit"] for s in sinyal)
print(f"\nB-rep'in DAHA IYI oldugu parca: {n_b}/{len(sinyal)} "
      f"(esit {n_e}, tez-saf iyi {len(sinyal)-n_b-n_e})")
print("\nB-rep'in kazandigi parcalar hangi markalarda:")
c = collections.Counter(s["mfg"] for s in sinyal if s["brep_iyi"])
t = collections.Counter(s["mfg"] for s in sinyal)
for m, k in c.most_common():
    print(f"  {m:<8} {k:>3}/{t[m]:<4} (%{100*k/t[m]:.0f})")
print("\nUCUZ SINYALLER (B-rep iyi vs degil, ortalama):")
for ad in ("n_seg", "n_brep", "seg_max", "seg_ort", "n_gt", "diag"):
    a = np.mean([s[ad] for s in sinyal if s["brep_iyi"]])
    b = np.mean([s[ad] for s in sinyal if not s["brep_iyi"]])
    print(f"  {ad:<9} B-rep-iyi {a:>8.3f} | digeri {b:>8.3f} | oran {a/max(b,1e-9):.2f}")
json.dump({"damga": makbuz_hash.damga(),
           "robot": {ad: 2*sum(x[0] for x in v)/max(sum(2*x[0]+x[1]+x[2] for x in v), 1)
                     for ad, v in kol.items()},
           "brep_iyi_parca": n_b, "esit": n_e, "n": len(sinyal), "sinyal": sinyal,
           "not": "Parca duzeyinde yonlendirici TAVANI. D7 marka-disi, MIKRO."},
          open("results/parca_yonlendirici.json", "w"), indent=1)
