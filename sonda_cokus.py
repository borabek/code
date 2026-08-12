# -*- coding: utf-8 -*-
"""S7 COKUS TESHISI: coken marka ile calisan marka arasindaki fark NE?

BULGU (2026-08-12). Secici verimliligi (gerceklesen F1 / havuz tavani) iki
kutuplu: UPUN %65.5, SUPU %47.5 -- ama MOR %5.7, NIT %0.5. Arada bir sey yok.
NIT'te havuz cevabi TASIYOR (tavan 0.9147) ama secici bulamiyor.

ILK HIPOTEZ (YOGUNLUK) HEMEN CURUYOR: NIT yogun (24.4 CP/parca) ama MOR
3.4 CP/parca ve UPUN 3.2 -- ikisi neredeyse ayni, biri cokuyor obru calisiyor.

BU SONDA su eksenlerde marka basina olcum yapar:
  * CP yogunlugu, aday/parca, secenek/parca
  * DOGRU secenegin parca ICINDEKI SKOR SIRASI (yuzdelik) -- asil soru bu:
    model dogru secenegi YUKARI koyuyor mu, yoksa gomuyor mu?
  * ilk-10 / ilk-50 icinde bulunan GT orani
  * pozitif ve negatif skorlarin AYRILIGI (medyan fark)
  * skorun parca ici DAGILIM GENISLIGI (goreli kural buna duyarli)

Skorlar OOF: her marka disarida birakilarak egitilir (gorulmemis marka kosulu).

Kullanim:  P6_DIZIN=results/_p6_oz_tam4 python sonda_cokus.py
"""
import collections
import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
os.environ.setdefault("P6_DIZIN", "results/_p6_oz_tam4")
sys.path.insert(0, ".")
import kanonik_d7 as K             # noqa: E402
import p6_karar                    # noqa: E402
from kos_p6_ortak import yukle     # noqa: E402

KUME = os.environ.get("CK_KUME", "d6")
KAT_MIN = int(os.environ.get("CK_KAT_MIN", "40"))
ITER = int(os.environ.get("P6_ITER", "200"))
NEG_KAT = int(os.environ.get("P6_NEG_KAT", "6"))
EKSEN_TOL = 40.0


def temel(d):
    return np.hstack([p6_karar.donustur(d["X"]),
                      p6_karar.kaynak_blok(d["kaynak"][d["idx"]])])


def dogru_maske(d):
    """Her secenek icin: bir GT'yi kabul kutusunda tutuyor mu?"""
    P = d["P"][d["idx"]]
    YD = d["YD"]
    G, Gd = np.asarray(d["G"], float), np.asarray(d["Gd"], float)
    if not len(P) or not len(G):
        return np.zeros(len(P), bool), np.zeros((0,), bool)
    v = P[:, None, :] - G[None, :, :]
    al = (v * Gd[None, :, :]).sum(-1)
    pe = np.linalg.norm(v - al[..., None] * Gd[None, :, :], axis=-1)
    an = np.degrees(np.arccos(np.clip(YD @ Gd.T, -1, 1)))
    kabul = (pe <= K.YANAL) & (np.abs(al) <= EKSEN_TOL) & (an <= K.ACI)
    return kabul.any(1), kabul.any(0)      # (secenek dogru mu, GT ulasildi mi)


def main():
    t0 = time.time()
    veri = [d for d in yukle(KUME, int(os.environ.get("P6_TR", "0")))]
    for d in veri:
        d["y"] = np.asarray(d["y"], int)
    marka = collections.Counter(d["mfg"] for d in veri)
    katlar = [m for m, n in marka.items() if n >= KAT_MIN]
    print(f"{len(veri)} parca | katlar {katlar} ({time.time() - t0:.0f} s)",
          flush=True)

    oof = [None] * len(veri)
    for b in katlar:
        ic = [i for i, d in enumerate(veri) if d["mfg"] != b]
        dis = [i for i, d in enumerate(veri) if d["mfg"] == b]
        n_satir = sum(len(veri[i]["y"]) for i in ic)
        ilk = temel(veri[ic[0]])
        M = np.empty((n_satir, ilk.shape[1]), np.float32)
        M[:len(ilk)] = ilk
        o = len(ilk)
        for i in ic[1:]:
            b_ = temel(veri[i])
            M[o:o + len(b_)] = b_
            o += len(b_)
        Y = np.concatenate([veri[i]["y"] for i in ic])
        rng = np.random.default_rng(0)
        poz = np.where(Y == 1)[0]
        neg = np.where(Y == 0)[0]
        sec = np.concatenate([poz, rng.choice(
            neg, min(len(neg), NEG_KAT * max(len(poz), 1)), replace=False)])
        m = HistGradientBoostingClassifier(
            max_iter=ITER, learning_rate=0.06, max_leaf_nodes=63,
            l2_regularization=1.0, random_state=0).fit(M[sec], Y[sec])
        for i in dis:
            oof[i] = m.predict_proba(temel(veri[i]).astype(np.float32))[:, 1]
        print(f"  OOF {b} ({time.time() - t0:.0f} s)", flush=True)

    ist = collections.defaultdict(lambda: collections.defaultdict(list))
    for d, s in zip(veri, oof):
        if s is None:
            continue
        mfg = d["mfg"]
        dg, ulasilan = dogru_maske(d)
        n = len(s)
        a = ist[mfg]
        a["parca"].append(1)
        a["cp"].append(len(d["G"]))
        a["aday"].append(len(np.unique(d["idx"])))
        a["secenek"].append(n)
        a["havuzda"].append(float(ulasilan.mean()) if len(ulasilan) else 0.0)
        if not n or not dg.any():
            continue
        # DOGRU secenegin parca ICINDEKI sira yuzdeligi (0 = en tepe)
        sira = np.argsort(np.argsort(-s))
        en_iyi_dogru = int(sira[dg].min())
        a["en_iyi_sira"].append(en_iyi_dogru)
        a["en_iyi_yuzdelik"].append(en_iyi_dogru / max(n - 1, 1))
        a["ilk10"].append(1.0 if en_iyi_dogru < 10 else 0.0)
        a["ilk50"].append(1.0 if en_iyi_dogru < 50 else 0.0)
        a["poz_med"].append(float(np.median(s[dg])))
        a["neg_med"].append(float(np.median(s[~dg])))
        a["skor_maks"].append(float(s.max()))
        a["skor_araligi"].append(float(s.max() - np.median(s)))

    print(f"\n{'marka':<7}{'parca':>6}{'CP/p':>7}{'aday/p':>8}{'sec/p':>8}"
          f"{'havuzda':>9}{'ilk10':>7}{'ilk50':>7}{'sira%':>8}"
          f"{'poz-neg':>9}{'skor_ar':>8}")
    out = {}
    for mfg in sorted(ist, key=lambda m: -np.mean(ist[m]["ilk10"] or [0])):
        a = ist[mfg]
        if not a["en_iyi_sira"]:
            continue
        r = {"parca": len(a["parca"]), "cp_parca": float(np.mean(a["cp"])),
             "aday_parca": float(np.mean(a["aday"])),
             "secenek_parca": float(np.mean(a["secenek"])),
             "havuzda": float(np.mean(a["havuzda"])),
             "ilk10": float(np.mean(a["ilk10"])),
             "ilk50": float(np.mean(a["ilk50"])),
             "sira_yuzdelik": float(np.mean(a["en_iyi_yuzdelik"])),
             "poz_neg_fark": float(np.mean(a["poz_med"]) -
                                   np.mean(a["neg_med"])),
             "skor_araligi": float(np.mean(a["skor_araligi"]))}
        out[mfg] = r
        print(f"{mfg:<7}{r['parca']:>6}{r['cp_parca']:>7.1f}"
              f"{r['aday_parca']:>8.0f}{r['secenek_parca']:>8.0f}"
              f"{r['havuzda']:>9.3f}{r['ilk10']:>7.3f}{r['ilk50']:>7.3f}"
              f"{r['sira_yuzdelik']:>8.3f}{r['poz_neg_fark']:>9.3f}"
              f"{r['skor_araligi']:>8.3f}")

    json.dump({"dizin": os.environ["P6_DIZIN"], "kume": KUME, "marka": out,
               "not": "S7 cokus teshisi. `havuzda` = GT'nin kabul kutusunda "
                      "en az bir secenegi olma orani (havuz tavani). "
                      "`ilk10/ilk50` = DOGRU secenegin parca icinde ilk 10/50'ye "
                      "girme orani. `sira%` = dogru secenegin ortalama sira "
                      "yuzdeligi (0 = tepe). D7'ye BAKILMADI."},
              open(f"results/cokus_teshisi_{KUME}.json", "w"), indent=1)
    print(f"\nmakbuz -> results/cokus_teshisi_{KUME}.json")


if __name__ == "__main__":
    main()
