# -*- coding: utf-8 -*-
"""C1: KAYNAK-DUYARLI ESIK -- B-rep adaylarindan gelen FP'yi kirp.

HATA BANKASI: 1323 FP'nin **750'si B-rep**, 394'u segmentasyon kaynakli. Kova
aritmetigi: FP yariya inerse F1 0.2773 -> 0.319 (isaret duzeltmeli tabanda daha
yuksek). Iki kaynak ayni esikle degerlendiriliyor; B-rep adaylari SAYICA cok
daha fazla oldugu icin ayni skorda daha cok yanlis uretiyorlar.

Kol: seg adaylari icin `e0`, B-rep adaylari icin `e1` AYRI esik. Izgara D6'da
taranir, D7'de YENIDEN TARANMAZ. Taban = isaret duzeltmeli kol (0.3090).

Kontrol: `e0 == e1` satiri tabanı yeniden uretmeli -- uretmiyorsa olcum bozuktur.
"""
import collections
import json
import os
import pickle
import sys

import numpy as np

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
OB = {"d7": "results/_p1_olasilik_d7", "d6": "results/_p1_olasilik_g7"}
KAYNAKLAR = (0, 1)
YANAL, ACI, EKSENEL = 2.0, 10.0, 40.0
GIRME, ERISIM = 3, 5
IZGARA = [(e0, e1) for e0 in (0.03, 0.05, 0.08)
          for e1 in (0.05, 0.08, 0.12, 0.18, 0.25)]


def yukle(on, pid):
    z = np.load(f"{OZ}/{on}_{pid}.npz")
    kay = np.asarray(z["kaynak"], int)
    m = np.isin(kay, KAYNAKLAR)
    T = np.asarray(np.load(f"{TAN}/{on}_{pid}.npz")["T"], float)
    X = np.hstack([np.asarray(z["X"], float), T])[m]
    if len(X) < 2:
        return None
    f = f"{OB[on]}/{pid}.npz"
    if not os.path.exists(f):
        return None
    zz = np.load(f)
    return {"X": X, "T": T[m], "kay": kay[m],
            "P": np.asarray(z["P"], float)[m],
            "D": np.asarray(z["D"], float)[m],
            "V": np.ascontiguousarray(zz["V"], np.float64),
            "F": np.ascontiguousarray(zz["F"], np.int64),
            "pb": np.asarray(zz["pbs"], float).mean(0)}


def kos(gate, veri, e0, e1, S, tam=False):
    rob = collections.defaultdict(lambda: [0, 0, 0])
    tes = []
    for d in veri:
        s = np.asarray(gate.predict_proba(
            wire_gate.parca_ici(d["X"], "zskor"))[:, 1], float)
        esik = np.where(d["kay"] == 0, e0, e1)
        k = s >= esik
        if not k.any():
            P = d["P"][:0]; D = d["D"][:0]
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
            "FP": int(sum(v[1] for v in rob.values())),
            "TP": int(sum(v[0] for v in rob.values())), "marka": pm}


def kume(on):
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
        d = yukle(on, pid)
        if d is None:
            continue
        d.update({"pid": pid, "mfg": r["mfg"],
                  "G": np.asarray(r["G"], float),
                  "Gd": np.asarray(r["Gd"], float), "diag": float(r["diag"])})
        out.append(d)
    return out


def main():
    gate = pickle.load(open("results/kazanan_hgb_derin.pkl", "rb"))["HGB-derin"]
    S = K.step_haritasi()
    dev, te = kume("d6"), kume("d7")
    print(f"D6 {len(dev)} | D7 {len(te)}\n", flush=True)
    en = None
    for e0, e1 in IZGARA:
        r = kos(gate, dev, e0, e1, S)
        print(f"  D6 e0={e0:.2f} e1={e1:.2f}  robot {r['robot']:.4f} "
              f"FP {r['FP']}", flush=True)
        if en is None or r["robot"] > en[2]["robot"]:
            en = (e0, e1, r)
    e0, e1, _ = en
    taban = kos(gate, te, 0.05, 0.05, S, tam=True)
    yeni = kos(gate, te, e0, e1, S, tam=True)
    print(f"\nTABAN  (e0=e1=0.05) robot {taban['robot']:.4f} | tespit "
          f"{taban['tespit']:.4f} | TP {taban['TP']} FP {taban['FP']}")
    print(f"KAYNAK (e0={e0} e1={e1}) robot {yeni['robot']:.4f} | tespit "
          f"{yeni['tespit']:.4f} | TP {yeni['TP']} FP {yeni['FP']}")
    print(f"\nFARK {yeni['robot']-taban['robot']:+.4f} | KAPI >= +0.02")
    json.dump({"damga": makbuz_hash.damga(), "taban": taban, "kaynak": yeni,
               "secilen": [e0, e1],
               "not": "Kaynak-duyarli esik (seg vs B-rep). Izgara D6'da, D7'de "
                      "yeniden taranmadi. Isaret duzeltmesi IKI kolda da acik. "
                      "D7 marka-disi, TAM ZINCIR."},
              open("results/c1_kaynak_esik.json", "w"), indent=1)
    print("makbuz -> results/c1_kaynak_esik.json")


if __name__ == "__main__":
    main()
