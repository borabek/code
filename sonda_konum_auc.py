# -*- coding: utf-8 -*-
"""KONUM duzeyi AYIRT EDICILIK: model NIT'te konum siralayabiliyor mu?

SUPHE. NIT'te ~260 konum var, 24'u dogru. RASTGELE ilk-24 secsek
24/260 = 0.092 cikardi. Olculen ilk-k orani 0.090.
Yani model NIT'te konum siralamasinda TAM RASTGELE olabilir.

Eger dogruysa: secenek duzeyindeki AUC 0.8854, konumu degil YONU ayirt
etmekten geliyordur. O zaman NIT'in duvari ne yon ne siralama kurali;
"hangi acikliklar kablo girisi" sorusunu HIC cevaplayamiyoruz demektir --
temsil darbogazi.

OLCULEN (marka disarida, gorulmemis marka kosulu):
  auc_secenek : secenek duzeyi (bilinen: NIT 0.8854)
  auc_konum   : KONUM duzeyi -- konum skoru = seceneklerinin max'i
  auc_yon     : YALNIZ dogru konumlar icinde, dogru yonu ayirma AUC'si
  rastgele_k  : k / n_konum  (ilk-k'nin rastgele beklentisi)
  ustk_konum  : gerceklesen konum ilk-k orani

OKUMA:
  auc_konum ~ 0.5  ve  ustk_konum ~ rastgele_k  -> konum bilgisi YOK
  auc_konum yuksek ama ustk dusuk               -> siralama var, kural kotu

D7'ye BAKILMAZ.
"""
import collections
import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import makbuz_hash

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
os.environ.setdefault("P6_DIZIN", "results/_p6_oz_tam4")
sys.path.insert(0, ".")
import p6_karar                    # noqa: E402
from kos_p6_ortak import yukle     # noqa: E402

KUME = os.environ.get("KA_KUME", "d6")
KAT_MIN = int(os.environ.get("KA_KAT_MIN", "40"))
ITER = int(os.environ.get("P6_ITER", "200"))
NEG_KAT = int(os.environ.get("P6_NEG_KAT", "6"))


def temel(d):
    return np.hstack([p6_karar.donustur(d["X"]),
                      p6_karar.kaynak_blok(d["kaynak"][d["idx"]])]).astype(
                          np.float32)


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    poz, neg = s[y == 1], s[y == 0]
    if not len(poz) or not len(neg):
        return np.nan
    h = np.concatenate([poz, neg])
    r = np.argsort(np.argsort(h)) + 1.0
    return float((r[:len(poz)].sum() - len(poz) * (len(poz) + 1) / 2.0)
                 / (len(poz) * len(neg)))


def konum_topla(s, y, idx):
    """her benzersiz konum icin (max skor, dogru mu)."""
    sira = np.argsort(idx, kind="stable")
    idx_s, s_s, y_s = idx[sira], s[sira], y[sira]
    sinir = np.flatnonzero(np.diff(idx_s)) + 1
    ss = np.split(s_s, sinir)
    yy = np.split(y_s, sinir)
    return (np.asarray([q.max() for q in ss]),
            np.asarray([int(q.max()) for q in yy]))


def main():
    t0 = time.time()
    veri = yukle(KUME, int(os.environ.get("P6_TR", "0")))
    for d in veri:
        d["y"] = np.asarray(d["y"], int)
        d["_M"] = temel(d)
    marka = collections.Counter(d["mfg"] for d in veri)
    katlar = [m for m, n in marka.items() if n >= KAT_MIN]
    print(f"{len(veri)} parca | katlar {katlar}", flush=True)

    oof = [None] * len(veri)
    for b in katlar:
        ic = [i for i, d in enumerate(veri) if d["mfg"] != b]
        dis = [i for i, d in enumerate(veri) if d["mfg"] == b]
        n_s = sum(len(veri[i]["y"]) for i in ic)
        M = np.empty((n_s, veri[0]["_M"].shape[1]), np.float32)
        o = 0
        for i in ic:
            m_ = veri[i]["_M"]
            M[o:o + len(m_)] = m_
            o += len(m_)
        Y = np.concatenate([veri[i]["y"] for i in ic])
        rng = np.random.default_rng(0)
        poz, neg = np.where(Y == 1)[0], np.where(Y == 0)[0]
        sec = np.concatenate([poz, rng.choice(
            neg, min(len(neg), NEG_KAT * max(len(poz), 1)), replace=False)])
        m = HistGradientBoostingClassifier(
            max_iter=ITER, learning_rate=0.06, max_leaf_nodes=63,
            l2_regularization=1.0, random_state=0).fit(M[sec], Y[sec])
        del M
        for i in dis:
            oof[i] = m.predict_proba(veri[i]["_M"])[:, 1]
        print(f"  OOF {b} ({time.time() - t0:.0f} s)", flush=True)

    ist = collections.defaultdict(lambda: collections.defaultdict(list))
    for d, s in zip(veri, oof):
        if s is None or d["y"].sum() == 0:
            continue
        s = np.asarray(s, float)
        y = d["y"]
        idx = np.asarray(d["idx"], int)
        ks, ky = konum_topla(s, y, idx)
        if ky.sum() == 0 or ky.sum() == len(ky):
            continue
        k = int(ky.sum())
        sira = np.argsort(-ks)
        a = ist[d["mfg"]]
        a["gt"].append(int(len(d["G"])))
        a["n_konum"].append(len(ks))
        a["auc_secenek"].append(auc(s, y))
        a["auc_konum"].append(auc(ks, ky))
        a["rastgele_k"].append(k / len(ks))
        a["ustk_konum"].append(float(ky[sira[:k]].sum()) / k)
        # YALNIZ dogru konumlar icinde yon ayrimi
        dk = np.isin(idx, np.unique(idx)[ky.astype(bool)])
        if dk.any() and 0 < y[dk].sum() < dk.sum():
            a["auc_yon"].append(auc(s[dk], y[dk]))

    print(f"\n{'marka':<7}{'GT':>6}{'n_konum':>9}{'auc_secenek':>13}"
          f"{'auc_KONUM':>12}{'auc_yon':>10}{'ustk_konum':>12}"
          f"{'rastgele_k':>12}")
    out = {}
    for m_ in sorted(ist, key=lambda x: -sum(ist[x]["gt"])):
        a = ist[m_]
        r = {"gt": int(sum(a["gt"])),
             "n_konum": float(np.median(a["n_konum"])),
             "auc_secenek": float(np.nanmean(a["auc_secenek"])),
             "auc_konum": float(np.nanmean(a["auc_konum"])),
             "auc_yon": float(np.nanmean(a["auc_yon"])) if a["auc_yon"]
             else float("nan"),
             "ustk_konum": float(np.mean(a["ustk_konum"])),
             "rastgele_k": float(np.mean(a["rastgele_k"]))}
        out[m_] = r
        print(f"{m_:<7}{r['gt']:>6}{r['n_konum']:>9.0f}"
              f"{r['auc_secenek']:>13.4f}{r['auc_konum']:>12.4f}"
              f"{r['auc_yon']:>10.4f}{r['ustk_konum']:>12.4f}"
              f"{r['rastgele_k']:>12.4f}")
    print("\nOKUMA:")
    print("  auc_KONUM ~ 0.5 ve ustk ~ rastgele -> KONUM bilgisi YOK")
    print("  auc_KONUM yuksek ama ustk dusuk    -> siralama var, kural kotu")
    json.dump({"damga": makbuz_hash.damga(), "kume": KUME, "marka": out,
               "not": "KONUM duzeyi ayirt edicilik. Secenek AUC'sinin yonden "
                      "mi konumdan mi geldigini ayirir. D7'ye BAKILMADI."},
              open(f"results/konum_auc_{KUME}.json", "w"), indent=1)
    print(f"makbuz -> results/konum_auc_{KUME}.json")


if __name__ == "__main__":
    main()
