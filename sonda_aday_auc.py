# -*- coding: utf-8 -*-
"""ADAY DUZEYINDE AYIRT EDICILIK: 0.75 ne kadar uzakta, TEK sayiyla

NEDEN. Otopsi mesh TEPELERINDE olctu: NIT'te segmentasyon AUC 0.6985.
Ama siralama TEPELERDE degil ADAYLAR uzerinde oluyor. Bugun tam bu farktan
bir kez yanildim (mesh normali tepelerde 0.800, secili adaylarda 0.334).
Bu yuzden ayni sey aday duzeyinde AYRICA olculur.

OLCULEN (marka disarida, gorulmemis marka kosulu):
  auc_secici : egitilmis secicinin aday duzeyinde AUC'si
  sira_ilk   : ilk dogru adayin ortanca sirasi
  sira_son   : SON dogru adayin ortanca sirasi   <- yogun parcada belirleyici
  n_aday     : parca basina ortanca secenek sayisi
  ustk_oran  : ilk-k (k = gercek CP sayisi) icinde dogru orani

VE TERSINDEN: ustk_oran'i 0.75'e cikarmak icin gereken AUC.
  Kaba bagintili tahmin: bir pozitifin ustundeki negatif orani (1 - AUC).
  Ilk-k'da kalabilmesi icin (1-AUC) * n_aday < k olmali, yani
      gereken_auc ~ 1 - k / n_aday
Bu bir ust sinir degil, BUYUKLUK MERTEBESI okumasidir; raporda oyle yazilir.

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

KUME = os.environ.get("AA_KUME", "d6")
KAT_MIN = int(os.environ.get("AA_KAT_MIN", "40"))
ITER = int(os.environ.get("P6_ITER", "200"))
NEG_KAT = int(os.environ.get("P6_NEG_KAT", "6"))


def temel(d):
    return np.hstack([p6_karar.donustur(d["X"]),
                      p6_karar.kaynak_blok(d["kaynak"][d["idx"]])]).astype(
                          np.float32)


def auc(s, y):
    poz, neg = s[y == 1], s[y == 0]
    if not len(poz) or not len(neg):
        return np.nan
    h = np.concatenate([poz, neg])
    r = np.argsort(np.argsort(h)) + 1.0
    return float((r[:len(poz)].sum() - len(poz) * (len(poz) + 1) / 2.0)
                 / (len(poz) * len(neg)))


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
        print(f"  {b} ({time.time() - t0:.0f} s)", flush=True)

    ist = collections.defaultdict(lambda: collections.defaultdict(list))
    for d, s in zip(veri, oof):
        if s is None:
            continue
        y = d["y"]
        if y.sum() == 0:
            continue
        s = np.asarray(s, float)
        sira = np.argsort(-s)
        yer = np.where(y[sira] == 1)[0]          # dogrularin 0-tabanli sirasi
        k = int(len(d["G"]))
        a = ist[d["mfg"]]
        a["gt"].append(k)
        a["auc"].append(auc(s, y))
        a["n"].append(len(s))
        a["ilk"].append(float(yer[0] + 1))
        a["son"].append(float(yer[-1] + 1))
        a["ustk"].append(float((yer < max(k, 1)).sum()) / max(k, 1))

    print(f"\n{'marka':<7}{'GT':>6}{'n_aday':>9}{'auc_secici':>12}"
          f"{'sira_ilk':>10}{'sira_son':>10}{'ustk':>8}{'gereken_auc':>13}")
    out = {}
    for m_ in sorted(ist, key=lambda x: -sum(ist[x]["gt"])):
        a = ist[m_]
        n_ = float(np.median(a["n"]))
        k_ = float(np.median(a["gt"]))
        r = {"gt": int(sum(a["gt"])),
             "n_aday": n_,
             "auc": float(np.nanmean(a["auc"])),
             "sira_ilk": float(np.median(a["ilk"])),
             "sira_son": float(np.median(a["son"])),
             "ustk": float(np.mean(a["ustk"])),
             "gereken_auc": 1.0 - k_ / max(n_, 1.0)}
        out[m_] = r
        print(f"{m_:<7}{r['gt']:>6}{n_:>9.0f}{r['auc']:>12.4f}"
              f"{r['sira_ilk']:>10.0f}{r['sira_son']:>10.0f}"
              f"{r['ustk']:>8.3f}{r['gereken_auc']:>13.4f}")
    print("\nOKUMA: 'gereken_auc' = ilk-k'nin dogrularla dolmasi icin gereken")
    print("       kaba AUC (buyukluk mertebesi). auc_secici ile arasindaki")
    print("       fark, 0.75'e giden mesafenin TEK sayilik ifadesidir.")
    json.dump({"damga": makbuz_hash.damga(), "kume": KUME, "marka": out,
               "not": "ADAY duzeyinde ayirt edicilik (mesh tepesi DEGIL). "
                      "Gorulmemis marka katlari. D7'ye BAKILMADI."},
              open(f"results/aday_auc_{KUME}.json", "w"), indent=1)
    print(f"makbuz -> results/aday_auc_{KUME}.json")


if __name__ == "__main__":
    main()
