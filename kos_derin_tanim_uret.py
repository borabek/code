# -*- coding: utf-8 -*-
"""Ikinci kademe olculerini KISA LISTE icin uret ve onbellekle.

Kisa liste = birinci kademe gate'in esigi gecen adaylar (~10-30/parca), yani
pahali isin olculeri ancak burada hesaplanabilir. Tum havuzda (100+/parca)
maliyet 5 kat artardi ve gerek yok: elenmis adaya olcu hesaplamak israf.

BIRINCI KADEME: `results/kazanan_hgb_derin.pkl` + ISARET-NORMAL tanimlayici
sozlesmesi (C3) -- yon isaret duzeltmesinden SONRA, yani `D` DISARI bakar.

Yeniden baslatilabilir: her parca kendi npz'sine yazilir, var olan atlanir.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import trimesh

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, ".")
import agiz_derin_tanim as ADT  # noqa: E402
import d6_kayit                 # noqa: E402
import kanonik_d7 as K          # noqa: E402
import wire_gate                # noqa: E402

OZ = "results/_tam_oz"
TAN = "results/_tan_hizali"
CIK = "results/_derin_tan"
OB = {"d7": "results/_p1_olasilik_d7", "d6": "results/_p1_olasilik_g7",
      "tam": "results/_p1_olasilik_brepegit"}
KAYNAKLAR = (0, 1)
ESIK = 0.05
GIRME, ERISIM = 3, 5


def normalle(T):
    T = np.asarray(T, float).copy()
    ters = T[:, ERISIM] < T[:, GIRME]
    g = T[ters, GIRME].copy()
    T[ters, GIRME] = T[ters, ERISIM]
    T[ters, ERISIM] = g
    return T, ters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kume", nargs="+", default=["d7", "d6", "tam"])
    a = ap.parse_args()
    os.makedirs(CIK, exist_ok=True)
    gate = pickle.load(open("results/kazanan_hgb_derin.pkl", "rb"))["HGB-derin"]
    d6 = {str(p): r for p, r in
          d6_kayit.yukle(set(d6_kayit.sinav()["pidler"])).items()}
    Rk = K.yukle(None)
    for on in a.kume:
        pids = [f[len(on) + 1:-4] for f in sorted(os.listdir(OZ))
                if f.startswith(on + "_") and f.endswith(".npz")]
        t0 = time.time()
        n = atlanan = 0
        for i, pid in enumerate(pids, 1):
            hedef = f"{CIK}/{on}_{pid}.npz"
            if os.path.exists(hedef):
                atlanan += 1
                continue
            r = Rk.get(pid) or d6.get(pid)
            f = f"{OB[on]}/{pid}.npz"
            if r is None or not os.path.exists(f):
                atlanan += 1
                continue
            z = np.load(f"{OZ}/{on}_{pid}.npz")
            m = np.isin(np.asarray(z["kaynak"], int), KAYNAKLAR)
            T = np.asarray(np.load(f"{TAN}/{on}_{pid}.npz")["T"], float)[m]
            Xg = np.asarray(z["X"], float)[m]
            # BIRINCI KADEME MODELI HAM tanimlayici sozlesmesinde egitildi
            # (134 sutun = (58+9)*2). C3'un isaret-normal surumu 136 sutunluydu
            # ve AYRI kaydedilmemisti; modelin kendi sozlesmesine uyulur.
            # Isaret duzeltmesi zaten SKORLAMADAN SONRA yone uygulanir.
            _Tn, ters = normalle(T)
            X = np.hstack([Xg, T])
            if len(X) < 2:
                np.savez_compressed(hedef, idx=np.zeros(0, int),
                                    Z=np.zeros((0, len(ADT.AD))))
                n += 1
                continue
            s = np.asarray(gate.predict_proba(
                wire_gate.parca_ici(X, "zskor"))[:, 1], float)
            k = np.where(s >= ESIK)[0]
            if not len(k):
                np.savez_compressed(hedef, idx=np.zeros(0, int),
                                    Z=np.zeros((0, len(ADT.AD))))
                n += 1
                continue
            P = np.asarray(z["P"], float)[m][k]
            D = np.asarray(z["D"], float)[m][k]
            # ISARET DUZELTMESI: derin olculer DISARI bakan yonle anlamlidir
            D = np.where(ters[k][:, None], -D, D)
            zz = np.load(f)
            mesh = trimesh.Trimesh(np.asarray(zz["V"], float),
                                   np.asarray(zz["F"], np.int64), process=False)
            Z = ADT.tanimla(P, D, mesh, float(r["diag"]))
            np.savez_compressed(hedef, idx=k.astype(np.int32),
                                Z=Z.astype(np.float32))
            n += 1
            if n % 50 == 0:
                h = (time.time() - t0) / n
                print(f"  {on} {i}/{len(pids)} yazilan={n} {h:.2f}s/parca "
                      f"kalan ~{h*(len(pids)-i)/60:.0f}dk", flush=True)
        print(f"{on} BITTI: yazilan {n} | atlanan {atlanan}", flush=True)
    print("->", CIK)


if __name__ == "__main__":
    main()
