# -*- coding: utf-8 -*-
"""IKINCI KADEME SECICI: kisa listeyi PAHALI FIZIKSEL olculerle yeniden puanla.

GEREKCE: bu oturumda ise yarayan iki koldan ikisi de FIZIKSELDI --
B2a isaret duzeltme (+0.0316, saf kural) ve C3 tanimlayici tutarliligi
(FP -%45). Ogrenme mimarisi degistirmek (siralama hedefi, pairwise, kalibrasyon)
HICBIR kolda ise yaramadi. Demek ki eksik olan MODEL degil OLCU.

Kademe 1: mevcut gate (58 + 9 tanimlayici, HGB-derin), esik 0.05 -> kisa liste
Kademe 2: kisa listeye 9 PAHALI olcu (`agiz_derin_tanim`) + kademe-1 skoru
          -> ikinci HGB. Esik D6'da secilir.
Sonra: NMS -> isaret duzeltme -> poz kafasi (urun zinciri).

KIYAS: ayni kisa liste, ikinci kademe KAPALI (yani mevcut 0.3070/0.3090 kolu).
Tek degisken: ikinci kademe.
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
DER = "results/_derin_tan"
OB = {"d7": "results/_p1_olasilik_d7", "d6": "results/_p1_olasilik_g7",
      "tam": "results/_p1_olasilik_brepegit"}
KAYNAKLAR = (0, 1)
ESIK1 = 0.05
YANAL, ACI, EKSENEL = 2.0, 10.0, 40.0
GIRME, ERISIM = 3, 5
ESIKLER2 = (0.005, 0.01, 0.02, 0.04, 0.08, 0.15)


def parca(on, pid, r, gate, mesh=False):
    f2 = f"{DER}/{on}_{pid}.npz"
    if not os.path.exists(f2):
        return None
    z = np.load(f"{OZ}/{on}_{pid}.npz")
    m = np.isin(np.asarray(z["kaynak"], int), KAYNAKLAR)
    T = np.asarray(np.load(f"{TAN}/{on}_{pid}.npz")["T"], float)[m]
    X = np.hstack([np.asarray(z["X"], float)[m], T])
    if len(X) < 2:
        return None
    s = np.asarray(gate.predict_proba(
        wire_gate.parca_ici(X, "zskor"))[:, 1], float)
    zz2 = np.load(f2)
    idx = np.asarray(zz2["idx"], int)
    Z = np.asarray(zz2["Z"], float)
    if not len(idx):
        return None
    # ONBELLEK TAZELIK KONTROLU: kisa liste ayni esikle yeniden uretilir ve
    # ONBELLEKTEKIYLE ayni olmali; degilse sessizce yanlis satira baglanir.
    k = np.where(s >= ESIK1)[0]
    if len(k) != len(idx) or not np.array_equal(k, idx):
        return None
    d = {"pid": pid, "mfg": r["mfg"], "X1": X[idx], "s1": s[idx], "Z": Z,
         "T": T[idx], "P": np.asarray(z["P"], float)[m][idx],
         "D": np.asarray(z["D"], float)[m][idx],
         "y": np.asarray(z["y"], int)[m][idx],
         "G": np.asarray(r["G"], float), "Gd": np.asarray(r["Gd"], float),
         "diag": float(r["diag"])}
    if mesh:
        f = f"{OB[on]}/{pid}.npz"
        if not os.path.exists(f):
            return None
        zz = np.load(f)
        d.update({"V": np.ascontiguousarray(zz["V"], np.float64),
                  "F": np.ascontiguousarray(zz["F"], np.int64),
                  "pb": np.asarray(zz["pbs"], float).mean(0)})
    return d


def kume(on, gate, mesh=False):
    d6 = {str(p): r for p, r in
          d6_kayit.yukle(set(d6_kayit.sinav()["pidler"])).items()}
    Rk = K.yukle(None)
    out, atlanan = [], 0
    for f in sorted(os.listdir(OZ)):
        if not (f.startswith(on + "_") and f.endswith(".npz")):
            continue
        pid = f[len(on) + 1:-4]
        r = Rk.get(pid) or d6.get(pid)
        if r is None or not len(r.get("G", [])):
            continue
        d = parca(on, pid, r, gate, mesh)
        if d is None:
            # BOS PARCA DA SAYILIR. Ilk surumde atlaniyordu ve D7'nin 835
            # parcasindan yalnizca 597'si olculuyordu; atlananlar tam da zor
            # parcalardi (kisa listesi bos) ve hepsi FN. Taban 0.3070 yerine
            # SAHTE 0.3642 cikmisti. Kume filtrelemek metrigi sisirir.
            atlanan += 1
            out.append({"pid": pid, "mfg": r["mfg"], "bos": True,
                        "G": np.asarray(r["G"], float),
                        "Gd": np.asarray(r["Gd"], float),
                        "diag": float(r["diag"])})
            continue
        out.append(d)
    print(f"  {on}: {len(out)} parca (atlanan {atlanan})", flush=True)
    return out


def oz2(d):
    """Ikinci kademe oznitelikleri: pahali olculer + kademe-1 skoru + baglam."""
    n = len(d["s1"])
    s = d["s1"][:, None]
    sira = (np.argsort(np.argsort(-d["s1"])) / max(n - 1, 1))[:, None]
    return np.hstack([d["Z"], s, sira, np.full((n, 1), float(n))])


def olc(veri, s2clf, e2, S, tam=False):
    rob = collections.defaultdict(lambda: [0, 0, 0])
    tes = []
    for d in veri:
        if d.get("bos"):
            tp, fp, fn = 0, 0, len(d["G"])
            a = rob[d["mfg"]]
            a[0] += tp; a[1] += fp; a[2] += fn
            tes.append((len(d["G"]), 0, 0, len(d["G"])))
            continue
        if s2clf is None:
            k = np.ones(len(d["s1"]), bool)
            sk_tam = d["s1"]
        else:
            # IKINCI KADEME SERT FILTRE DEGIL, SKOR BIRLESTIRME:
            # s = s1 * s2. Sert esik kademe-1'in bilgisini atiyordu.
            s2v = s2clf.predict_proba(oz2(d))[:, 1]
            sk_tam = d["s1"] * s2v
            k = sk_tam >= e2
        P, D, T = d["P"][k], d["D"][k], d["T"][k]
        sk = sk_tam[k]
        if len(P) > 1:
            nm = wire_gate.kalabalik_maskesi(P, sk)
            P, D, T = P[nm], D[nm], T[nm]
        if len(P):
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
    gate = pickle.load(open("results/kazanan_hgb_derin.pkl", "rb"))["HGB-derin"]
    S = K.step_haritasi()
    tr = kume("tam", gate)
    dev = kume("d6", gate, mesh=True)
    te = kume("d7", gate, mesh=True)
    trd = [d for d in tr if not d.get("bos")]
    X = np.vstack([oz2(d) for d in trd])
    Y = np.concatenate([d["y"] for d in trd])
    print(f"ikinci kademe egitimi {X.shape} | pozitif {Y.mean():.4f}", flush=True)
    s2 = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.06,
                                        max_leaf_nodes=63,
                                        l2_regularization=1.0,
                                        random_state=0).fit(X, Y)
    pickle.dump(s2, open("results/ikinci_kademe.pkl", "wb"))
    en = None
    for e2 in ESIKLER2:
        r = olc(dev, s2, e2, S)
        print(f"  D6 e2={e2:.2f} robot {r['robot']:.4f} TP {r['TP']} FP {r['FP']}",
              flush=True)
        if en is None or r["robot"] > en[1]["robot"]:
            en = (e2, r)
    e2 = en[0]
    taban = olc(te, None, 0.0, S, tam=True)
    yeni = olc(te, s2, e2, S, tam=True)
    print(f"\nTABAN (tek kademe)  robot {taban['robot']:.4f} | tespit "
          f"{taban['tespit']:.4f} | TP {taban['TP']} FP {taban['FP']}")
    print(f"IKI KADEME (e2={e2}) robot {yeni['robot']:.4f} | tespit "
          f"{yeni['tespit']:.4f} | TP {yeni['TP']} FP {yeni['FP']}")
    print(f"\nFARK {yeni['robot']-taban['robot']:+.4f} | KAPI >= +0.02 "
          f"(esik gurultusu ~0.015)")
    json.dump({"damga": makbuz_hash.damga(), "taban": taban, "iki_kademe": yeni,
               "e2": e2,
               "not": "Ikinci kademe: kisa listeye pahali fiziksel olculer. Esik "
                      "D6'da secildi. D7 marka-disi, TAM ZINCIR, MIKRO."},
              open("results/ikinci_kademe_v2.json", "w"), indent=1)
    print("makbuz -> results/ikinci_kademe.json")


if __name__ == "__main__":
    main()
