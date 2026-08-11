# -*- coding: utf-8 -*-
"""C3: tanimlayicilari ISARETE GORE NORMALLESTIR, gate'i oyle egit.

BULUNAN TUTARSIZLIK: `agiz_tanimlayici` sutunlari adayin OZGUN yonuyle
hesaplanmis (`girme` = -d yonunde serbest yol, `erisim` = +d yonunde). Ama
B2a isaret duzeltmesi (+0.0316) yonu ters cevirebiliyor. Yani gate, yonu ters
olan adaylarda bu iki sutunu TERS anlamda okuyor: onun icin "iceri" olan sey
gercekte "disari".

DUZELTME (ek hesap YOK -- sutun takasi):
  erisim < girme  ise  girme <-> erisim TAKAS EDILIR
  ayrica `ters` bayragi eklenir (10. sutun)
Boylece gate her adayi HEP AYNI sozlesmede gorur: `girme` her zaman govdeye
dogru, `erisim` her zaman disariya.

Kova hedefi: GATE_REDDI (489 GT, %15.8). Tutarli oznitelik siralamayi
duzeltirse bu kova bosalir; kova aritmetigi 0.3070 -> 0.455 tavani veriyor.

Tek degisken: OZNITELIK SOZLESMESI. Havuz, etiket, model ailesi, esik izgarasi
ayni. Esik D6'da secilir.
"""
import collections
import json
import os
import pickle
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import makbuz_hash

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, ".")
import d6_kayit                 # noqa: E402
import kanonik_d7 as K          # noqa: E402
import urun_zinciri             # noqa: E402
import wire_gate                # noqa: E402
from sina_kume import esle_macar  # noqa: E402

OZ = "results/_tam_oz"
TAN = "results/_tan_hizali"
OB = {"d7": "results/_p1_olasilik_d7", "d6": "results/_p1_olasilik_g7",
      "tam": "results/_p1_olasilik_brepegit"}
KAYNAKLAR = (0, 1)
YANAL, ACI, EKSENEL = 2.0, 10.0, 40.0
GIRME, ERISIM = 3, 5
ESIKLER = (0.03, 0.05, 0.08, 0.12, 0.20, 0.30)


def normalle(T):
    """Doner: (normallesmis T, ters bayragi). Sutun TAKASI, yeniden hesap YOK."""
    T = np.asarray(T, float).copy()
    ters = T[:, ERISIM] < T[:, GIRME]
    g = T[ters, GIRME].copy()
    T[ters, GIRME] = T[ters, ERISIM]
    T[ters, ERISIM] = g
    return T, ters.astype(float)


def yukle(on, pid, normal, mesh=False):
    z = np.load(f"{OZ}/{on}_{pid}.npz")
    m = np.isin(np.asarray(z["kaynak"], int), KAYNAKLAR)
    T = np.asarray(np.load(f"{TAN}/{on}_{pid}.npz")["T"], float)[m]
    Xg = np.asarray(z["X"], float)[m]
    ham = T.copy()
    if normal:
        Tn, ters = normalle(T)
        X = np.hstack([Xg, Tn, ters[:, None]])
    else:
        X = np.hstack([Xg, T])
    if len(X) < 2:
        return None
    d = {"X": X, "T": ham, "P": np.asarray(z["P"], float)[m],
         "D": np.asarray(z["D"], float)[m], "y": np.asarray(z["y"], int)[m]}
    if mesh:
        f = f"{OB[on]}/{pid}.npz"
        if not os.path.exists(f):
            return None
        zz = np.load(f)
        d.update({"V": np.ascontiguousarray(zz["V"], np.float64),
                  "F": np.ascontiguousarray(zz["F"], np.int64),
                  "pb": np.asarray(zz["pbs"], float).mean(0)})
    return d


def kume(on, normal, mesh=False):
    d6 = {str(p): r for p, r in
          d6_kayit.yukle(set(d6_kayit.sinav()["pidler"])).items()}
    Rk = K.yukle(None)
    out = []
    for f in sorted(os.listdir(OZ)):
        if not (f.startswith(on + "_") and f.endswith(".npz")):
            continue
        pid = f[len(on) + 1:-4]
        r = Rk.get(pid) or d6.get(pid)
        if r is None or not len(r.get("G", [])):
            continue
        d = yukle(on, pid, normal, mesh)
        if d is None:
            continue
        d.update({"pid": pid, "mfg": r["mfg"], "G": np.asarray(r["G"], float),
                  "Gd": np.asarray(r["Gd"], float), "diag": float(r["diag"])})
        out.append(d)
    return out


def olc(model, veri, e, S, tam=False):
    rob = collections.defaultdict(lambda: [0, 0, 0])
    tes = []
    for d in veri:
        s = np.asarray(model.predict_proba(
            wire_gate.parca_ici(d["X"], "zskor"))[:, 1], float)
        k = s >= e
        if not k.any():
            P, D = d["P"][:0], d["D"][:0]
        else:
            P, D, T, sk = d["P"][k], d["D"][k], d["T"][k], s[k]
            if len(P) > 1:
                nm = wire_gate.kalabalik_maskesi(P, sk)
                P, D, T = P[nm], D[nm], T[nm]
            D = np.where((T[:, ERISIM] < T[:, GIRME])[:, None], -D, D)
        if tam and len(P):
            P, D = urun_zinciri.tam_poz(d["V"], d["F"], d["pb"], P, D,
                                        step_path=S.get(d["pid"]))
        tp, fp, fn = esle_macar(P, D, d["G"], d["Gd"], d["diag"], YANAL, ACI,
                                False, isaretli=True)[:3]
        a = rob[d["mfg"]]
        a[0] += tp; a[1] += fp; a[2] += fn
        tes.append((len(d["G"]),) + esle_macar(
            P, D, d["G"], d["Gd"], d["diag"], max(3.0, 0.06 * d["diag"]),
            180.0, True)[:3])
    pm = {m: 2 * v[0] / max(2 * v[0] + v[1] + v[2], 1) for m, v in rob.items()}
    mi = float(2 * sum(v[0] for v in rob.values()) /
               max(sum(2 * v[0] + v[1] + v[2] for v in rob.values()), 1))
    return {"robot": mi, "tespit": K.mikro(tes),
            "makro": float(np.mean(list(pm.values()))),
            "en_kotu": float(min(pm.values())),
            "TP": int(sum(v[0] for v in rob.values())),
            "FP": int(sum(v[1] for v in rob.values())), "marka": pm}


def main():
    S = K.step_haritasi()
    sonuc = {}
    for ad, normal in (("HAM tanimlayici", False),
                       ("ISARET-NORMAL tanimlayici", True)):
        tr = kume("tam", normal)
        dev = kume("d6", normal, mesh=True)
        te = kume("d7", normal, mesh=True)
        M = np.vstack([wire_gate.parca_ici(d["X"], "zskor") for d in tr])
        Y = np.concatenate([d["y"] for d in tr])
        m = HistGradientBoostingClassifier(
            max_iter=600, learning_rate=0.06, max_leaf_nodes=63,
            l2_regularization=1.0, random_state=0).fit(M, Y)
        en = None
        for e in ESIKLER:
            r = olc(m, dev, e, S)
            if en is None or r["robot"] > en[1]["robot"]:
                en = (e, r)
        e, _ = en
        r7 = olc(m, te, e, S, tam=True)
        sonuc[ad] = dict(r7, esik=e, sutun=int(M.shape[1]))
        print(f"{ad:<28} sutun {M.shape[1]} | esik {e:.2f} -> D7 robot "
              f"**{r7['robot']:.4f}** | tespit {r7['tespit']:.4f} | makro "
              f"{r7['makro']:.4f} | TP {r7['TP']} FP {r7['FP']}", flush=True)
    a = sonuc["HAM tanimlayici"]["robot"]
    b = sonuc["ISARET-NORMAL tanimlayici"]["robot"]
    print(f"\nFARK {b - a:+.4f} | KAPI >= +0.02 (esik gurultusu ~0.015)")
    json.dump({"damga": makbuz_hash.damga(), "sonuc": sonuc, "fark": b - a,
               "not": "Tek degisken: tanimlayici sozlesmesi (ham vs isarete gore "
                      "normallesmis + ters bayragi). Esik D6'da. D7 marka-disi, "
                      "TAM ZINCIR, MIKRO."},
              open("results/c3_isaret_normal.json", "w"), indent=1)
    print("makbuz -> results/c3_isaret_normal.json")


if __name__ == "__main__":
    main()
