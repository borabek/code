# -*- coding: utf-8 -*-
"""III. KOL -- YONU OGRENME, HESAPLA (analitik silindir ekseni)

NEDEN SIMDI. Yayilim kolunun darbogazi olculdu: uretilen konumda dogru yon
MEVCUT (kahin 0.527) ama model onu SECEMIYOR (0.077). Yani sorun yon
BILGISININ yoklugu degil, SECIMI. B-rep agzinda yon ise ANALITIKTIR --
silindirin ekseni. Ogrenmeye gerek yok.

ISARET SORUNU. Metrik ISARETLI aci kullaniyor; eksenin isareti ise keyfi
(+u ve -u ayni eksen). Disari bakan yon, silindir MERKEZINDEN AGZA giden
vektorle belirlenir (`mouth_a`/`mouth_b` onbellekte var).

BU SONDA su tavani olcer: her GT icin, konum kutusunda bir B-rep agzi VE o
agzin ANALITIK yonu ISARETLI aci kutusunda mi?

  konum         : agiz, GT'nin konum kutusunda mi (yanal<=2, eksenel<=40)
  konum+eksen   : ustune eksen ISARETSIZ 10 derece icinde mi
  konum+isaretli: ustune eksen ISARETLI 10 derece icinde mi (URUN OLCUSU)

`konum+isaretli` yuksekse yon HESAPLANABILIR ve ogrenilmesi gereksizdir.
ISARETSIZ ile ISARETLI arasindaki fark, isaret belirleme sorununun buyuklugu.

D7'ye BAKILMAZ.
"""
import collections
import json
import os
import pickle
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, ".")
import kanonik_d7 as K   # noqa: E402
import d6_kayit          # noqa: E402

SIL = os.environ.get("AY_SIL", "results/_d6_silindirler.pkl")
MARKALAR = set(os.environ.get("AY_MARKA", "NIT,MOR,SUPU,UPUN").split(","))
YANAL, EKSENEL = 2.0, 40.0


def _birim(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def agizlar(sil):
    """(konum, disari_bakan_yon) ciftleri. Yon = merkezden agza."""
    P, D = [], []
    for c in sil:
        ax = np.asarray(c["axis"], float)
        ax = ax / max(np.linalg.norm(ax), 1e-12)
        merkez = np.asarray(c["center"], float)
        for k in ("mouth_a", "mouth_b"):
            m = c.get(k)
            if m is None:
                continue
            m = np.asarray(m, float)
            v = m - merkez
            if np.linalg.norm(v) < 1e-6:
                continue
            # DISARI BAKAN yon: eksenin, merkezden agza giden bilesenle
            # ayni isaretli hali
            yon = ax if float(v @ ax) > 0 else -ax
            # ISARET SOZLESMESI: GT yonu govdenin ICINE mi DISARI mi bakiyor?
            # `kanonik` blogunda pozitiflerin `disa_bakis` degeri NEGATIFTI,
            # yani GT ICERI bakiyor olabilir. AY_TERS=1 ile sinanir.
            if os.environ.get("AY_TERS", "0") == "1":
                yon = -yon
            P.append(m)
            D.append(yon)
    if not P:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.asarray(P), np.asarray(D)


def main():
    cy = pickle.load(open(SIL, "rb"))
    kay = K.yukle()
    kay.update({str(p): r for p, r in d6_kayit.yukle().items()
                if str(p) not in kay})
    ist = collections.defaultdict(lambda: collections.defaultdict(list))
    n = 0
    for pid, r in kay.items():
        if r.get("mfg") not in MARKALAR:
            continue
        G = np.asarray(r.get("G", []), float)
        if not len(G):
            continue
        sil = cy.get(str(pid))
        if not sil:
            continue
        AP, AD = agizlar(sil)
        if not len(AP):
            continue
        Gn = _birim(np.asarray(r["Gd"], float))
        v = AP[:, None, :] - G[None, :, :]
        al = (v * Gn[None, :, :]).sum(-1)
        yan = np.linalg.norm(v - al[..., None] * Gn[None, :, :], axis=-1)
        konum = (yan <= YANAL) & (np.abs(al) <= EKSENEL)      # (agiz, gt)
        cos = AD @ Gn.T
        aci_i = np.degrees(np.arccos(np.clip(cos, -1, 1)))          # ISARETLI
        aci_s = np.degrees(np.arccos(np.clip(np.abs(cos), -1, 1)))  # isaretsiz
        a = ist[r["mfg"]]
        a["gt"].append(len(G))
        a["konum"].append(int(konum.any(0).sum()))
        a["eksen"].append(int((konum & (aci_s <= K.ACI)).any(0).sum()))
        a["isaretli"].append(int((konum & (aci_i <= K.ACI)).any(0).sum()))
        a["agiz"].append(len(AP))
        n += 1
    print(f"{n} parca | kabul: yanal<={YANAL} eksenel<={EKSENEL} aci<={K.ACI}\n")
    print(f"{'marka':<7}{'GT':>7}{'agiz/p':>8}{'KONUM':>9}{'+eksen':>9}"
          f"{'+ISARETLI':>11}{'isaret kaybi':>14}")
    out = {}
    for m_ in sorted(ist, key=lambda x: -sum(ist[x]["gt"])):
        a = ist[m_]
        g = sum(a["gt"])
        k = sum(a["konum"]) / max(g, 1)
        e = sum(a["eksen"]) / max(g, 1)
        i = sum(a["isaretli"]) / max(g, 1)
        out[m_] = {"gt": g, "konum": k, "eksen": e, "isaretli": i,
                   "isaret_kaybi": e - i,
                   "agiz_parca": float(np.mean(a["agiz"]))}
        print(f"{m_:<7}{g:>7}{np.mean(a['agiz']):>8.0f}{k:>9.3f}{e:>9.3f}"
              f"{i:>11.3f}{e - i:>14.3f}")
    json.dump({"yanal": YANAL, "eksenel": EKSENEL, "aci": K.ACI,
               "marka": out,
               "not": "B-rep agzi + ANALITIK eksen yonu (merkezden agza "
                      "isaretlenmis). 'isaret kaybi' = isaretsiz - isaretli. "
                      "D7'ye BAKILMADI."},
              open("results/analitik_yon.json", "w"), indent=1)
    print("\nmakbuz -> results/analitik_yon.json")
    print("OKUMA: +ISARETLI yuksekse yon HESAPLANABILIR, ogrenilmesi gereksiz.")


if __name__ == "__main__":
    main()
