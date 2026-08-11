# -*- coding: utf-8 -*-
"""CALISAN TAVANI 0.80'e: mesh tepelerini FILTRELE + normali yon kaynagi yap.

DURUM (`results/tavan_085.json`): P2+Y3 kademesi 0.7798, P3 (ham mesh tepeleri)
0.8686. 0.80 arasinda kaliyor -> mesh tepeleri SART, ama ham hali parca basina
~7000 aday demek ve DAGITILAMAZ.

BU SONDA IKI SEYI OLCER:
1. Mesh tepelerini segmentasyon guveniyle (`p_pos = CE + CT`) filtrelersek
   tavanin ne kadari HAYATTA KALIR ve aday sayisi NEYE DUSER.
2. Mesh tepesinin YEREL NORMALI yon kaynagi olarak eklenirse YON YOK kovasi
   (%23.0) ne olur.

Cikti bir EGRI: her esik icin (tavan, aday/parca). Dagitilabilirlik icin
aday/parca'nin mevcut urun olcegine (13.4) yakin kalmasi gerekir; 0.80 tavan
100-200 aday/parca ile geliyorsa hala kabul edilebilir (gate onu tarayabilir).

ISARETLI aci. D7 marka-disi. Mukemmel secici tavani.
"""
import collections
import json
import os
import pickle
import sys

import numpy as np

import makbuz_hash

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, ".")
import brep_havuz               # noqa: E402
import connector3d              # noqa: E402
import kanonik_d7 as K          # noqa: E402

YANAL, ACI, EKSENEL = 2.0, 10.0, 40.0
YON_R = 10.0
OB = "results/_p1_olasilik_d7"
SIK_T = (0.05, 0.15, 0.3)
CE, CT = int(connector3d.CABLE_ENTRY), int(connector3d.CONTACT)
ESIKLER = (1.01, 0.50, 0.30, 0.20, 0.10, 0.05, 0.0)   # 1.01 = mesh tepesi YOK


def _birim(V):
    V = np.asarray(V, float).reshape(-1, 3)
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)


def tepe_normalleri(V, F):
    """Alan agirlikli tepe normalleri -- agiz tepesinde disari bakar."""
    N = np.zeros_like(V)
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    return _birim(N)


def eksen_ornek(cyl):
    P, D = [], []
    for c in cyl or []:
        a = np.asarray(c["axis"], float)
        na = np.linalg.norm(a)
        ma, mb = c.get("mouth_a"), c.get("mouth_b")
        if na < 1e-9 or ma is None or mb is None:
            continue
        a = a / na
        ma = np.asarray(ma, float); mb = np.asarray(mb, float)
        for t in SIK_T:
            P.append(ma + t * (mb - ma)); D.append(a)
            P.append(mb + t * (ma - mb)); D.append(-a)
    return (np.asarray(P, float).reshape(-1, 3),
            np.asarray(D, float).reshape(-1, 3))


def main():
    cy = pickle.load(open("results/_d7_silindirler.pkl", "rb"))
    ac = pickle.load(open("results/_d7_acikliklar.pkl", "rb"))
    kay = K.yukle(json.load(open("results/d7_sinav_kumesi.json"))["pidler"])
    say = {e: collections.Counter() for e in ESIKLER}
    aday = {e: 0 for e in ESIKLER}
    n_gt = 0
    n_parca = 0
    for pid, r in sorted(kay.items()):
        G = np.asarray(r.get("G", []), float)
        f = f"{OB}/{pid}.npz"
        if not len(G) or not os.path.exists(f):
            continue
        Gd = np.asarray(r["Gd"], float)
        z = np.load(f)
        V = np.asarray(z["V"], float)
        F = np.asarray(z["F"], np.int64)
        pb = np.asarray(z["pbs"], float).mean(0)
        ppos = pb[:, CE] + pb[:, CT]
        NV = tepe_normalleri(V, F)
        cyl = cy.get(str(pid))
        P0 = np.asarray(r["P"], float).reshape(-1, 3)
        D0 = _birim(r["Pd"]) if len(P0) else np.zeros((0, 3))
        Pb, Db, _ = brep_havuz.birlesik_havuz(P0, D0, cyl, ac.get(str(pid)))
        Pe, De = eksen_ornek(cyl)
        Pbase = np.vstack([Pb, Pe]) if len(Pe) else Pb
        Dbase = _birim(np.vstack([Db, De])) if len(De) else _birim(Db)
        eks = [np.asarray(c["axis"], float) for c in (cyl or [])
               if np.linalg.norm(np.asarray(c["axis"], float)) > 1e-9]
        if eks:
            _e = _birim(eks); eks = np.vstack([_e, -_e])
        else:
            eks = np.zeros((0, 3))
        ana = np.zeros((0, 3))
        if len(V) > 3:
            Q = V - V.mean(0)
            _a = _birim(np.linalg.svd(Q, full_matrices=False)[2])
            ana = np.vstack([_a, -_a])
        n_parca += 1
        n_gt += len(G)
        for e in ESIKLER:
            k = ppos >= e
            Pm = V[k]
            Nm = np.vstack([NV[k], -NV[k]]) if k.any() else np.zeros((0, 3))
            Pp = np.vstack([Pbase, Pm]) if len(Pm) else Pbase
            aday[e] += len(Pp)
            if not len(Pp):
                say[e]["KONUM YOK"] += len(G)
                continue
            for j in range(len(G)):
                nn = np.linalg.norm(Gd[j])
                if nn < 1e-9:
                    say[e]["GT BOZUK"] += 1
                    continue
                u = Gd[j] / nn
                w = Pp - G[j]
                al = w @ u
                yan = np.linalg.norm(w - al[:, None] * u[None], axis=1)
                ok = (yan <= YANAL) & (np.abs(al) <= EKSENEL)
                if not ok.any():
                    say[e]["KONUM YOK"] += 1
                    continue
                idx = np.where(ok)[0]

                def uy(Dv):
                    if Dv is None or not len(Dv):
                        return False
                    a = np.degrees(np.arccos(
                        np.clip(_birim(Dv) @ u, -1.0, 1.0)))
                    return bool((a <= ACI).any())

                nb = len(Dbase)
                if uy(Dbase[idx[idx < nb]]):
                    say[e]["Y0 kendi"] += 1
                    continue
                bul = False
                for i in idx[idx < nb]:
                    d = np.linalg.norm(Pbase - Pbase[i], axis=1)
                    if uy(Dbase[d <= YON_R]):
                        say[e]["Y1 komsu"] += 1
                        bul = True
                        break
                if bul:
                    continue
                # YENI: mesh tepesinin YEREL NORMALI (yalniz secilen tepelerde)
                mi = idx[idx >= nb] - nb
                if len(mi) and len(Nm):
                    yerel = np.vstack([Nm[:len(Nm)//2][mi],
                                       -Nm[:len(Nm)//2][mi]])
                    if uy(yerel):
                        say[e]["Y_normal"] += 1
                        continue
                if len(eks) and uy(eks):
                    say[e]["Y2 silindir ekseni"] += 1
                    continue
                if len(ana) and uy(ana):
                    say[e]["Y3 ana eksen"] += 1
                    continue
                say[e]["YON YOK"] += 1

    print(f"D7 {n_parca} parca / {n_gt} GT\n")
    BASARI = ("Y0 kendi", "Y1 komsu", "Y_normal", "Y2 silindir ekseni",
              "Y3 ana eksen")
    print(f"{'p_pos esigi':<12} {'recall':>8} {'F1 tavani':>10} "
          f"{'aday/parca':>11} {'KONUM YOK':>10} {'YON YOK':>9}")
    out = {}
    for e in ESIKLER:
        c = say[e]
        tam = sum(c[k] for k in BASARI)
        rr = tam / max(n_gt, 1)
        f1 = 2 * rr / (1 + rr)
        ap = aday[e] / max(n_parca, 1)
        out[str(e)] = {"recall": rr, "f1_tavani": f1, "aday_per_parca": ap,
                       "kirilim": dict(c)}
        ad = "mesh YOK" if e > 1 else f"{e:.2f}"
        print(f"{ad:<12} {rr:>8.4f} {f1:>10.4f} {ap:>11.1f} "
              f"{c['KONUM YOK']/max(n_gt,1):>10.3f} "
              f"{c['YON YOK']/max(n_gt,1):>9.3f}")
    uy80 = [(e, out[str(e)]) for e in ESIKLER if out[str(e)]["f1_tavani"] >= 0.80]
    if uy80:
        e, c = min(uy80, key=lambda x: x[1]["aday_per_parca"])
        print(f"\n0.80'i saglayan EN UCUZ esik: p_pos>={e} -> tavan "
              f"{c['f1_tavani']:.4f} | {c['aday_per_parca']:.1f} aday/parca")
    else:
        print(f"\n0.80 SAGLANMADI (en yuksek "
              f"{max(v['f1_tavani'] for v in out.values()):.4f})")
    json.dump({"damga": makbuz_hash.damga(), "n_gt": n_gt, "n_parca": n_parca,
               "sonuc": out,
               "not": "Mesh tepeleri p_pos ile filtrelendi + YEREL NORMAL yon "
                      "kaynagi eklendi. ISARETLI aci. D7 marka-disi. TAVAN."},
              open("results/tavan_080_dagitilabilir.json", "w"), indent=1)
    print("makbuz -> results/tavan_080_dagitilabilir.json")


if __name__ == "__main__":
    main()
