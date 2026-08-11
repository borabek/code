# -*- coding: utf-8 -*-
"""P2 (agiz segmentasyonu) NE KADAR GETIRIR? -- tahmin degil olcum.

Soru: "agiz segmentasyonuna emek vermeye deger mi?"

YONTEM: segmentasyon kalitesi ZATEN parcadan parcaya degisiyor. GT'leri
segmentasyon kalitesine gore ceyreklere ayirip her ceyrekte robot basarisini
olcersek, "kaliteli ceyregin davranisi HERKESE uygulansa ne olurdu" sorusu
P2'nin ust sinirini verir. Bu bir TAHMIN DEGIL, veriden okunan bir farktir.

KALITE OLCUSU (GT'den BAGIMSIZ olmali, yoksa dairesel olur): adayin cevresindeki
segmentasyon guveni -- `p_pos = softmax[CE] + softmax[CT]`'nin aday etrafindaki
`R` mm'lik topta ORTALAMASI ve KESKINLIGI. GT hicbir yerde kullanilmaz.

Ayrica ayni ceyreklerde YANAL hata dagilimi verilir: P2'nin dogrudan hedefi
yanal hatadir.
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
import connector3d            # noqa: E402
import kanonik_d7 as K        # noqa: E402
import wire_gate              # noqa: E402
from p1c_esik import maske    # noqa: E402

YANAL, ACI, EKSENEL = 2.0, 10.0, 40.0
TOP_R = 3.0                   # aday cevresi yaricapi (mm)
CE, CT = int(connector3d.CABLE_ENTRY), int(connector3d.CONTACT)
OB = "results/_p1_olasilik_d7"


def kalite(V, ppos, p):
    """Aday cevresindeki segmentasyon kalitesi: (ortalama guven, keskinlik).

    Keskinlik = topun ICI ile DISI arasindaki guven farki. Yuksek keskinlik =
    ag agzi net ayirmis; dusuk = bulanik yayilmis.
    """
    d = np.linalg.norm(V - p, axis=1)
    ic = d <= TOP_R
    dis = (d > TOP_R) & (d <= 3 * TOP_R)
    if ic.sum() < 3:
        return 0.0, 0.0
    o = float(ppos[ic].mean())
    return o, o - (float(ppos[dis].mean()) if dis.sum() >= 3 else 0.0)


def main():
    gate = K.gate_yukle()
    te = K.yukle(json.load(open("results/d7_sinav_kumesi.json"))["pidler"])
    kayitlar = []
    atlanan = 0
    for pid, r in sorted(te.items()):
        G = np.asarray(r.get("G", []), float)
        M = K.x58(r)
        f = f"{OB}/{pid}.npz"
        if M is None or not len(G) or not os.path.exists(f):
            atlanan += 1
            continue
        z = np.load(f)
        V = np.asarray(z["V"], float)
        pb = np.asarray(z["pbs"], float).mean(0)
        ppos = pb[:, CE] + pb[:, CT]
        P = np.asarray(r["P"], float)
        D = np.asarray(r["Pd"], float)
        Gd = np.asarray(r["Gd"], float)
        s = np.asarray(wire_gate.karar_skoru(gate, M), float)
        k = maske(s, 0.40, 0.30)
        Pk, Dk = (P[k], D[k]) if k.any() else (P[:0], D[:0])
        for j in range(len(G)):
            g, gd = G[j], Gd[j]
            n = np.linalg.norm(gd)
            if n < 1e-9:
                continue
            u = gd / n
            o, kes = kalite(V, ppos, g)      # GT KONUMU yalniz TOP MERKEZI icin;
            # kalite olcusu GT'nin dogru/yanlis olmasina BAKMIYOR, yalnizca agin
            # o BOLGEDE ne kadar net konustugunu olcuyor.
            if not len(Pk):
                kayitlar.append((o, kes, np.inf, np.inf, 0))
                continue
            v = Pk - g[None]
            e = v @ u
            yan = np.linalg.norm(v - e[:, None] * u[None], axis=1)
            i = int(np.argmin(yan))
            Dn = Dk / np.maximum(np.linalg.norm(Dk, axis=1, keepdims=True), 1e-12)
            aci = np.degrees(np.arccos(np.clip(abs(float(Dn[i] @ u)), -1.0, 1.0)))
            tamam = int(((yan <= YANAL) & (np.abs(e) <= EKSENEL) &
                         (np.degrees(np.arccos(np.clip(np.abs(Dn @ u), -1.0, 1.0)))
                          <= ACI)).any())
            kayitlar.append((o, kes, float(yan[i]), float(aci), tamam))
    print(f"D7: {len(kayitlar)} GT | atlanan parca {atlanan}", flush=True)

    A = np.asarray([[a, b] for a, b, _, _, _ in kayitlar], float)
    yanal = np.asarray([c for _, _, c, _, _ in kayitlar], float)
    tam = np.asarray([t for _, _, _, _, t in kayitlar], int)
    for ad, sut in (("ORTALAMA GUVEN", 0), ("KESKINLIK", 1)):
        q = np.percentile(A[:, sut], [25, 50, 75])
        gr = np.digitize(A[:, sut], q)
        print(f"\n--- {ad} ceyrekleri ---")
        print(f"{'ceyrek':<10} {'n':>5} {'robot':>8} {'yanal<=2mm':>11} "
              f"{'medyan yanal':>13}")
        oran = []
        for c in range(4):
            k = gr == c
            if not k.any():
                continue
            y = yanal[k]
            sonlu = y[np.isfinite(y)]
            r = float(tam[k].mean())
            oran.append(r)
            print(f"Q{c+1:<9} {int(k.sum()):>5} {r:>8.4f} "
                  f"{float((y <= YANAL).mean()):>11.4f} "
                  f"{(np.median(sonlu) if len(sonlu) else float('nan')):>13.3f}")
        if len(oran) == 4:
            print(f"  EN IYI ceyrek {oran[3]:.4f} vs EN KOTU {oran[0]:.4f} "
                  f"-> fark {oran[3]-oran[0]:+.4f}")
            print(f"  HERKES en iyi ceyrek gibi olsa robot recall "
                  f"{oran[3]:.4f} (su an {float(tam.mean()):.4f}, "
                  f"pay {oran[3]-float(tam.mean()):+.4f})")
    su_an = float(tam.mean())
    q = np.percentile(A[:, 0], [25, 50, 75])
    gr = np.digitize(A[:, 0], q)
    ust = float(tam[gr == 3].mean())
    f1_su = 2 * su_an / (1 + su_an)
    f1_ust = 2 * ust / (1 + ust)
    print(f"\nROBOT RECALL su an {su_an:.4f} -> tavan (herkes Q4 gibi) {ust:.4f}")
    print(f"KABA F1 KARSILIGI  {f1_su:.4f} -> {f1_ust:.4f}  "
          f"(pay {f1_ust-f1_su:+.4f})")
    json.dump({"damga": makbuz_hash.damga(), "n_gt": len(kayitlar),
               "robot_recall_su_an": su_an, "robot_recall_Q4": ust,
               "kaba_f1_su_an": f1_su, "kaba_f1_Q4": f1_ust,
               "not": "P2'nin UST SINIRI: segmentasyon kalitesi ceyreklerinden "
                      "okunan fark. Kalite olcusu GT'nin dogrulugundan BAGIMSIZ "
                      "(agin o bolgede ne kadar net konustugu). D7 marka-disi."},
              open("results/p2_getirisi.json", "w"), indent=1)
    print("\nmakbuz -> results/p2_getirisi.json")


if __name__ == "__main__":
    main()
