# -*- coding: utf-8 -*-
"""MESH NORMALI KORPUSU: her adaya TEK yon (kendi yuzey normali)

BULGU (2026-08-12). Mesh tepesinin kendi normali GT yonunu ISARETLI olarak
NIT'te %80, SUPU'da %93.6 tutuyor. Yani yon OGRENILMESI gereken bir karar
degil, GEOMETRIDEN OKUNAN bir buyukluk.

Bugun her adaya `yon_bankasi` ~24 secenek takiyor ve secici 4857 secenek
icinden ~24 dogruyu bulmaya calisiyor (pozitif yogunlugu 1/202). Bu korpus
her adaya TEK secenek birakir -> ~350 secenek, yogunluk 1/15 (13.5 KAT).

NEDEN YENIDEN CIKARIM DEGIL DE SUZGEC: oznitelik sutunlarinin C ve D
bloklari SECENEK YONUNE bagli hesaplanmis durumda. Yeniden cikarim ~4 saat.
Onun yerine her adayin seceneklerinden, yonu mesh normaline EN YAKIN olani
TUTULUR -- oznitelikler o secenek icin zaten dogru hesaplanmistir.

Kullanim:
    NK_ON=d6 NK_CIK=results/_p6_oz_normal python kos_normal_korpus.py
"""
import os
import sys
import time

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, ".")

KAYNAK = os.environ.get("NK_KAYNAK", "results/_p6_oz_tam4")
CIK = os.environ.get("NK_CIK", "results/_p6_oz_normal")
ON = os.environ.get("NK_ON", "d6")
MESH = {"d6": "results/_p1_olasilik",
        "tam": "results/_p1_olasilik_brepegit",
        "d7": "results/_p1_olasilik_d7"}[ON]
SHARD = os.environ.get("NK_SHARD")


def _birim(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def main():
    import trimesh
    os.makedirs(CIK, exist_ok=True)
    fs = sorted(f for f in os.listdir(KAYNAK)
                if f.startswith(ON + "_") and f.endswith(".npz"))
    if SHARD:
        i_, n_ = (int(x) for x in SHARD.split("/"))
        fs = [f for k, f in enumerate(fs) if k % n_ == i_]
        print(f"  PAY {i_}/{n_}", flush=True)
    print(f"{ON}: {len(fs)} parca | {KAYNAK} -> {CIK}", flush=True)
    t0 = time.time()
    yaz = atla = bos = 0
    once = sonra = 0
    for i, f in enumerate(fs, 1):
        hedef = f"{CIK}/{f}"
        if os.path.exists(hedef):
            atla += 1
            continue
        pid = f[len(ON) + 1:-4]
        mf = f"{MESH}/{pid}.npz"
        if not os.path.exists(mf):
            bos += 1
            continue
        z = np.load(f"{KAYNAK}/{f}")
        X, idx, YD = z["X"], np.asarray(z["idx"], int), np.asarray(z["YD"], float)
        P, D, kay = z["P"], z["D"], z["kaynak"]
        if not len(idx):
            bos += 1
            continue
        zz = np.load(mf)
        V = np.ascontiguousarray(zz["V"], np.float64)
        Fc = np.ascontiguousarray(zz["F"], np.int64)
        mesh = trimesh.Trimesh(V, Fc, process=False)
        VN = _birim(np.asarray(mesh.vertex_normals, float))
        Pc = np.asarray(P, float)
        # her ADAYIN normali: en yakin mesh tepesinin normali
        yak = np.argmin(np.linalg.norm(Pc[:, None, :] - V[None, :, :],
                                       axis=-1), axis=1)
        AN = VN[yak]                       # (aday, 3)
        YDn = _birim(YD)
        # her secenek: kendi adayinin normaliyle ISARETLI aci
        cos = (YDn * AN[idx]).sum(1)
        # aday basina EN YAKIN secenegi tut
        tut = np.zeros(len(idx), bool)
        sira = np.lexsort((-cos, idx))
        _, ilk = np.unique(idx[sira], return_index=True)
        tut[sira[ilk]] = True
        once += len(idx)
        sonra += int(tut.sum())
        np.savez_compressed(hedef + ".tmp",
                            X=X[tut], idx=idx[tut], YD=YD[tut],
                            P=P, D=D, kaynak=kay)
        os.replace(hedef + ".tmp.npz", hedef)
        yaz += 1
        if i % 50 == 0:
            print(f"  {i}/{len(fs)} yazilan {yaz} | secenek "
                  f"{once / max(yaz, 1):.0f} -> {sonra / max(yaz, 1):.0f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    print(f"\nBITTI: yazilan {yaz} | atlanan {atla} | bos {bos}")
    if yaz:
        print(f"secenek/parca: {once / yaz:.0f} -> {sonra / yaz:.0f} "
              f"({once / max(sonra, 1):.1f}x azalma)")


if __name__ == "__main__":
    main()
