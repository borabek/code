# -*- coding: utf-8 -*-
"""ADAYIN KENDI NORMALI: yeniden cikarimi hak eden bir tavan var mi?

DURUM. "Mesh normali yonu %80 tutuyor" sondasi, kutudaki 6000 TEPENIN
HERHANGI BIRINI kabul ediyordu -- fazla comert bir olcut. Korpus suzgeci
denendiginde (her adaya bankadaki normale EN YAKIN secenek) yonlu recall
0.8926 -> 0.3679 dustu, NIT'te 0.8429 -> 0.027.

Iki ayri sebep vardi:
  (a) adaylar 6000 tepeden ~490'a SEYRELTILMIS
  (b) "bankadaki normale en yakin secenek" != "normalin kendisi"

BU SONDA (b)'yi kaldirir: her ADAYA KENDI normali verilir ve yonlu recall
olculur. Bu, yeniden cikarimin (yaklasik 4 saat) TAVANIDIR.

  tavan >= ~0.80 ise yeniden cikarim HAK EDILIR
  tavan dusukse mekanizma OLU -- 4 saat harcanmaz

Ozniteliklere ihtiyac yok; yalnizca geometri. D7'ye BAKILMAZ.
"""
import collections
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, ".")
import kanonik_d7 as K   # noqa: E402
import d6_kayit          # noqa: E402

KORPUS = os.environ.get("AN_KORPUS", "results/_p6_oz_tam4")
MESH = os.environ.get("AN_MESH", "results/_p1_olasilik")
ON = os.environ.get("AN_ON", "d6")
YANAL, EKSENEL = 2.0, 40.0


def _birim(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def main():
    import trimesh
    kay = K.yukle()
    kay.update({str(p): r for p, r in d6_kayit.yukle().items()
                if str(p) not in kay})
    fs = sorted(f for f in os.listdir(KORPUS)
                if f.startswith(ON + "_") and f.endswith(".npz"))
    ist = collections.defaultdict(lambda: collections.defaultdict(list))
    t0 = time.time()
    n = 0
    for f in fs:
        pid = f[len(ON) + 1:-4]
        r = kay.get(pid)
        if not r or not len(r.get("G", [])):
            continue
        mf = f"{MESH}/{pid}.npz"
        if not os.path.exists(mf):
            continue
        z = np.load(f"{KORPUS}/{f}")
        P = np.asarray(z["P"], float)
        idx = np.asarray(z["idx"], int)
        YD = _birim(np.asarray(z["YD"], float))
        if not len(P):
            continue
        zz = np.load(mf)
        V = np.ascontiguousarray(zz["V"], np.float64)
        Fc = np.ascontiguousarray(zz["F"], np.int64)
        mesh = trimesh.Trimesh(V, Fc, process=False)
        VN = _birim(np.asarray(mesh.vertex_normals, float))
        yak = np.argmin(np.linalg.norm(P[:, None, :] - V[None, :, :],
                                       axis=-1), axis=1)
        AN = VN[yak]                              # ADAYIN KENDI normali
        G = np.asarray(r["G"], float)
        Gn = _birim(np.asarray(r["Gd"], float))
        v = P[:, None, :] - G[None, :, :]
        al = (v * Gn[None, :, :]).sum(-1)
        yan = np.linalg.norm(v - al[..., None] * Gn[None, :, :], axis=-1)
        konum = (yan <= YANAL) & (np.abs(al) <= EKSENEL)     # (aday, gt)
        aci_n = np.degrees(np.arccos(np.clip(AN @ Gn.T, -1, 1)))
        a = ist[r["mfg"]]
        a["gt"].append(len(G))
        a["konum"].append(int(konum.any(0).sum()))
        a["normal"].append(int((konum & (aci_n <= K.ACI)).any(0).sum()))
        # KIYAS: mevcut BANKA (tum secenekler)
        kb = np.zeros(len(G), bool)
        for j in range(len(G)):
            ad = np.where(konum[:, j])[0]
            if not len(ad):
                continue
            m_ = np.isin(idx, ad)
            if not m_.any():
                continue
            aci = np.degrees(np.arccos(np.clip(YD[m_] @ Gn[j], -1, 1)))
            if (aci <= K.ACI).any():
                kb[j] = True
        a["banka"].append(int(kb.sum()))
        a["aday"].append(len(P))
        n += 1
        if n % 60 == 0:
            print(f"  {n} parca ({time.time() - t0:.0f} s)", flush=True)

    print(f"\n{'marka':<7}{'GT':>7}{'aday/p':>8}{'KONUM':>8}"
          f"{'ADAY NORMALI':>14}{'banka (24 sec)':>16}")
    out = {}
    tg = tn = tb = 0
    for m_ in sorted(ist, key=lambda x: -sum(ist[x]["gt"])):
        a = ist[m_]
        g = max(sum(a["gt"]), 1)
        k = sum(a["konum"]) / g
        nn = sum(a["normal"]) / g
        bb = sum(a["banka"]) / g
        tg += g; tn += sum(a["normal"]); tb += sum(a["banka"])
        out[m_] = {"gt": g, "konum": k, "aday_normali": nn, "banka": bb,
                   "aday_parca": float(np.mean(a["aday"]))}
        print(f"{m_:<7}{g:>7}{np.mean(a['aday']):>8.0f}{k:>8.3f}"
              f"{nn:>14.3f}{bb:>16.3f}")
    print(f"{'TOPLAM':<7}{tg:>7}{'':>8}{'':>8}{tn / tg:>14.3f}{tb / tg:>16.3f}")
    json.dump({"korpus": KORPUS, "marka": out,
               "toplam": {"aday_normali": tn / tg, "banka": tb / tg},
               "not": "ADAYIN KENDI mesh normali vs 24 secenekli yon bankasi. "
                      "Yeniden cikarimin TAVANI. D7'ye BAKILMADI."},
              open("results/aday_normal_tavan.json", "w"), indent=1)
    print("\nmakbuz -> results/aday_normal_tavan.json")
    print("KARAR: aday normali ~banka'ya yakinsa yeniden cikarim HAK EDILIR")
    print("       (24 kat az secenek, ayni recall). Cok dusukse mekanizma OLU.")


if __name__ == "__main__":
    main()
