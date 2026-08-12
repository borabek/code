# -*- coding: utf-8 -*-
"""VII.4 -- URETILEN KONUMDA YON KURTARILABILIYOR MU?

VII.0m kafesin GT olmadan bulunabildigini gosterdi (NIT kapsama 0.676). AMA
o olcum yalnizca KONUMU kontrol ediyordu (yanal 2mm / eksenel 40mm). Robot
metrigi ayrica **ISARETLI ACI <= 10 derece** istiyor.

BU SONDA tam kabul kutusunu uygular. Uretilen her izgara noktasinin
CEVRESINDEKI adaylarin yon-bankasi secenekleri toplanir ve GT, ancak
(konum kutusu) VE (yon kutusu) birlikte saglanirsa kapsanmis sayilir.

Yon secimi KAHINDIR (mevcut secenekler icinden EN IYISI). Yani bu bir TAVAN:
"uretilen konumda dogru yon MEVCUT MU?" sorusunu yanitlar. Mevcut degilse
hicbir secici onu bulamaz -- kol orada olur. Mevcutsa is seciciye kalir.

NEDEN ONEMLI: onceki kafes denemesi (K2.1) tam burada coktu -- yonu
KOPYALAMISTI ve robot metrigi -0.0100 dusmustu (tespit +0.0126 iken).

D7'ye BAKILMAZ.
"""
import collections
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ.setdefault("P6_DIZIN", "results/_p6_oz_tam4")
sys.path.insert(0, ".")
import kanonik_d7 as K             # noqa: E402
from kos_p6_ortak import yukle     # noqa: E402
from sonda_kafes_v2 import kafes_ara   # noqa: E402  (GT'SIZ arama)

KUME = os.environ.get("KY_KUME", "d6")
MARKALAR = set(os.environ.get("KY_MARKA", "NIT,MOR,SUPU,UPUN").split(","))
YANAL, EKSENEL = 2.0, 40.0
YAKIN_R = float(os.environ.get("KY_YAKIN", "2.0"))   # izgara -> aday yaricapi


def main():
    t0 = time.time()
    veri = yukle(KUME, int(os.environ.get("P6_TR", "0")))
    ist = collections.defaultdict(lambda: collections.defaultdict(list))
    n = 0
    for d in veri:
        if d["mfg"] not in MARKALAR:
            continue
        G = np.asarray(d["G"], float)
        if len(G) < 3:
            continue
        Gd = np.asarray(d["Gd"], float)
        Gn = Gd / np.maximum(np.linalg.norm(Gd, axis=1, keepdims=True), 1e-12)
        P = np.asarray(d["P"], float)          # ADAY konumlari
        idx = np.asarray(d["idx"], int)        # secenek -> aday
        YD = np.asarray(d["YD"], float)        # secenek yonleri
        Pu = np.unique(np.round(P, 3), axis=0)
        bulunan = kafes_ara(Pu)
        if not bulunan:
            continue
        uret = np.vstack([b[2] for b in bulunan])

        # 1) YALNIZ KONUM (VII.0m ile ayni olcu)
        v = uret[:, None, :] - G[None, :, :]
        al = (v * Gn[None, :, :]).sum(-1)
        yan = np.linalg.norm(v - al[..., None] * Gn[None, :, :], axis=-1)
        konum_ok = (yan <= YANAL) & (np.abs(al) <= EKSENEL)      # (uret, gt)

        # 2) KONUM + YON: izgara noktasinin cevresindeki adaylarin secenekleri
        d_ua = np.linalg.norm(uret[:, None, :] - P[None, :, :], axis=-1)
        yakin_aday = d_ua <= YAKIN_R                              # (uret, aday)
        tam_ok = np.zeros(len(G), bool)
        for j in range(len(G)):
            u_ler = np.where(konum_ok[:, j])[0]
            if not len(u_ler):
                continue
            adaylar = np.where(yakin_aday[u_ler].any(0))[0]
            if not len(adaylar):
                continue
            sec = np.isin(idx, adaylar)
            if not sec.any():
                continue
            cos = YD[sec] @ Gn[j]
            aci = np.degrees(np.arccos(np.clip(cos, -1, 1)))      # ISARETLI
            if (aci <= K.ACI).any():
                tam_ok[j] = True

        a = ist[d["mfg"]]
        a["gt"].append(len(G))
        a["konum"].append(int(konum_ok.any(0).sum()))
        a["tam"].append(int(tam_ok.sum()))
        a["uret"].append(len(uret))
        n += 1
        if n % 30 == 0:
            print(f"  {n} parca ({time.time() - t0:.0f} s)", flush=True)

    print(f"\n{'marka':<7}{'GT':>7}{'uret/p':>8}{'KONUM':>9}{'KONUM+YON':>11}"
          f"{'yon kaybi':>11}")
    out = {}
    for m_ in sorted(ist, key=lambda x: -sum(ist[x]["gt"])):
        a = ist[m_]
        g = sum(a["gt"])
        k = sum(a["konum"]) / max(g, 1)
        t = sum(a["tam"]) / max(g, 1)
        out[m_] = {"gt": g, "konum": k, "konum_yon": t, "yon_kaybi": k - t,
                   "uret_parca": float(np.mean(a["uret"]))}
        print(f"{m_:<7}{g:>7}{np.mean(a['uret']):>8.0f}{k:>9.3f}{t:>11.3f}"
              f"{k - t:>11.3f}")
    json.dump({"yanal": YANAL, "eksenel": EKSENEL, "aci": K.ACI,
               "yakin_r": YAKIN_R, "marka": out,
               "not": "Uretilen izgara noktalarinda YON, cevredeki adaylarin "
                      "yon-bankasi seceneklerinden KAHIN gibi secilir. TAVAN "
                      "olcumudur: dogru yon MEVCUT MU? D7'ye BAKILMADI."},
              open(f"results/kafes_yon_{KUME}.json", "w"), indent=1)
    print(f"\nmakbuz -> results/kafes_yon_{KUME}.json")
    print("OKUMA: yon kaybi KUCUKSE kol canli (yon uretilen konumda mevcut);")
    print("       BUYUKSE K2.1'in coktugu yere geri donduk demektir.")


if __name__ == "__main__":
    main()
