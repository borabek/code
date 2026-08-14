# -*- coding: utf-8 -*-
"""YENI-8 -- KONUM TOPLAMA: 24 yon secenegi = 24 PIYANGO BILETI

TESHIS (results/aday_auc_d6.json). Aday duzeyinde secicinin AUC'si iyi
(NIT 0.8854, UPUN 0.9877). Bagliyan sey AUC degil, ADAY/GT orani:
  NIT  6322 secenek / 24 GT   -> gereken AUC 0.9968
  MOR 11555 secenek /  9 GT   -> gereken AUC 0.9997
Ama bu 6322 sayisi KONUM sayisi degil: her konum MAX_SEC=24 yon secenegi
uretiyor, yani ~260 konum x 24 yon.

MEKANIZMA. Bir konumu "en yuksek skorlu secenegiyle" siralamak, her YANLIS
konuma 24 piyango bileti vermektir. Yanlis bir konumun 24 denemeden birinde
yuksek skor kapma olasiligi, dogru konumun tek gercek sinyalini bastirir.
Bu klasik coklu-karsilastirma sismesidir ve MAX_SEC 12->24'un tavani acip
gerceklesen F1'i acmamasini da aciklar.

COZUM. Konum skorunu MAX ile degil, sisme yapmayan bir toplamayla kur.
Yanlis konumda skorlar rastgele dagilir (yuksek max, dusuk ortanca);
dogru konumda BIRCOK secenek makul skor alir (yuksek ortanca).

KOLLAR (konum siralamasi; yon her zaman o konumun en iyi secenegi):
  max        : bugunku
  ortanca    : medyan -- piyangoyu tamamen sondurur
  ort        : ortalama
  q75        : 75. yuzdelik -- max ile ortanca arasi
  say        : skoru 0.5 ustunde olan secenek SAYISI
  maxxort    : max * ortalama (gucbirligi)
  ust2       : en yuksek IKI secenegin ortalamasi

OLCULEN: konum duzeyinde ilk-k orani ve UCTAN UCA robot F1 (esik kurali).
KAPI: mikro robot F1'de +0.01.
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
import kanonik_d7 as K             # noqa: E402
import p6_karar                    # noqa: E402
from kos_p6_ortak import yukle     # noqa: E402
from sina_kume import esle_macar   # noqa: E402

KUME = os.environ.get("KT_KUME", "d6")
KAT_MIN = int(os.environ.get("KT_KAT_MIN", "40"))
ITER = int(os.environ.get("P6_ITER", "200"))
NEG_KAT = int(os.environ.get("P6_NEG_KAT", "6"))
NMS = 5.0
KOLLAR = ("max", "ortanca", "ort", "q75", "say", "maxxort", "ust2")
ESIKLER = (0.20, 0.40, 0.60, 0.80, 0.90, 0.95)


def temel(d):
    return np.hstack([p6_karar.donustur(d["X"]),
                      p6_karar.kaynak_blok(d["kaynak"][d["idx"]])]).astype(
                          np.float32)


def konum_skoru(s, idx, kol):
    """her BENZERSIZ konum icin (skor, o konumun en iyi secenek indisi)."""
    sira = np.argsort(idx, kind="stable")
    idx_s, s_s = idx[sira], s[sira]
    sinir = np.flatnonzero(np.diff(idx_s)) + 1
    parcalar = np.split(np.arange(len(idx_s)), sinir)
    konum, skor, en_iyi = [], [], []
    for p in parcalar:
        q = s_s[p]
        if kol == "max":
            v = q.max()
        elif kol == "ortanca":
            v = np.median(q)
        elif kol == "ort":
            v = q.mean()
        elif kol == "q75":
            v = np.percentile(q, 75)
        elif kol == "say":
            v = float((q >= 0.5).sum()) / len(q)
        elif kol == "maxxort":
            v = q.max() * q.mean()
        elif kol == "ust2":
            v = np.sort(q)[-2:].mean()
        else:
            raise ValueError(kol)
        konum.append(idx_s[p[0]])
        skor.append(float(v))
        en_iyi.append(int(sira[p[int(np.argmax(q))]]))
    return (np.asarray(konum, int), np.asarray(skor, float),
            np.asarray(en_iyi, int))


def konum_dogru(y, idx):
    """her BENZERSIZ konum (konum_skoru ile AYNI sirada) dogru mu.

    Dongusuz: konum_skoru gibi idx'e gore siralayip parcalara boler.
    Naif hali (her konum icin `y[idx == ki]`) 6322 secenek x 260 konum x
    7 kol x 468 parca = milyarlarca islem ederdi; ayrica kola bagli
    olmadigi icin parca basina BIR kez hesaplanir.
    """
    sira = np.argsort(idx, kind="stable")
    idx_s, y_s = idx[sira], y[sira]
    sinir = np.flatnonzero(np.diff(idx_s)) + 1
    return np.asarray([bool(q.max()) if len(q) else False
                       for q in np.split(y_s, sinir)], bool)


def sec_esik(P, YD, idx, ks, en_iyi, esik):
    """konum skoru esigi + NMS; yon = o konumun en iyi secenegi."""
    tut = np.where(ks >= esik)[0]
    if not len(tut):
        tut = np.array([int(np.argmax(ks))])
    tut = tut[np.argsort(-ks[tut])]
    sp, sd = [], []
    for j in tut:
        p = P[j]
        if sp and min(np.linalg.norm(np.asarray(sp) - p, axis=1)) < NMS:
            continue
        sp.append(p)
        sd.append(YD[en_iyi[j]])
    return (np.asarray(sp).reshape(-1, 3), np.asarray(sd).reshape(-1, 3))


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

    # esik, KAT-DISI secilir: her kol icin egitim markalarindan degil,
    # burada tek kume oldugu icin TUM kollara AYNI esik listesi uygulanip
    # en iyisi AYRI raporlanir (kol karsilastirmasi esikten bagimsiz olsun).
    agg = {k: {e: collections.defaultdict(collections.Counter)
               for e in ESIKLER} for k in KOLLAR}
    ustk = collections.defaultdict(lambda: collections.defaultdict(list))
    n = 0
    for d, s in zip(veri, oof):
        if s is None:
            continue
        G = np.asarray(d["G"], float)
        Gd = np.asarray(d["Gd"], float)
        P = np.asarray(d["P"], float)
        idx = np.asarray(d["idx"], int)
        YD = np.asarray(d["YD"], float)
        s = np.asarray(s, float)
        y = d["y"]
        dogru = konum_dogru(y, idx)
        for kol in KOLLAR:
            kon, ks, en_iyi = konum_skoru(s, idx, kol)
            # OLCEK DUZELTMESI. Ilk kosuda esik izgarasi (0.20-0.95) TUM
            # kollara ham olcekte uygulandi. Ortanca/ortalama skorlarin
            # olcegi cok daha sikisik oldugu icin ayni esik bambaska bir
            # yerden kesiyordu -- kollar kiyaslanamazdi (max 0.3012 vs
            # ort 0.0790). Parca ici YUZDELIK siralamaya cevirince butun
            # kollar ayni olcege gelir ve esik anlamli olur.
            if os.environ.get("KT_NORM", "1") == "1" and len(ks) > 1:
                ks = np.argsort(np.argsort(ks)) / (len(ks) - 1.0)
            assert len(dogru) == len(kon), "konum sirasi tutmuyor"
            # konum duzeyinde ilk-k dogruluk orani
            sira = np.argsort(-ks)
            k = max(len(G), 1)
            ustk[d["mfg"]][kol].append(float(dogru[sira[:k]].sum()) / k)
            for e in ESIKLER:
                Ps, Ds = sec_esik(P[kon], YD, idx, ks, en_iyi, e)
                tp, fp, fn = esle_macar(Ps, Ds, G, Gd, d["diag"], K.YANAL,
                                        K.ACI, False, isaretli=True)[:3]
                q = agg[kol][e][d["mfg"]]
                q["tp"] += tp; q["fp"] += fp; q["fn"] += fn
        n += 1
        if n % 80 == 0:
            print(f"  {n} parca ({time.time() - t0:.0f} s)", flush=True)

    def f1(c):
        return 2 * c["tp"] / max(2 * c["tp"] + c["fp"] + c["fn"], 1)

    print(f"\n{n} parca | KONUM duzeyi ilk-k dogruluk orani")
    print(f"{'marka':<7}" + "".join(f"{k:>10}" for k in KOLLAR))
    for m_ in sorted(ustk):
        print(f"{m_:<7}" + "".join(
            f"{np.mean(ustk[m_][k]):>10.3f}" for k in KOLLAR))

    print(f"\nUCTAN UCA mikro robot F1 (esik taramasi)")
    print(f"{'esik':<7}" + "".join(f"{k:>10}" for k in KOLLAR))
    en = {}
    for e in ESIKLER:
        satir = {}
        for kol in KOLLAR:
            T = collections.Counter()
            for q in agg[kol][e].values():
                T += q
            satir[kol] = f1(T)
            en[kol] = max(en.get(kol, 0.0), satir[kol])
        print(f"{e:<7.2f}" + "".join(f"{satir[k]:>10.4f}" for k in KOLLAR))
    print(f"\n{'EN IYI':<7}" + "".join(f"{en[k]:>10.4f}" for k in KOLLAR))
    print("\n=== max'A GORE ===")
    for kol in KOLLAR[1:]:
        f = en[kol] - en["max"]
        print(f"  {kol:<10}{en[kol]:.4f}   {f:+.4f}"
              + ("  <- KAPI GECTI" if f >= 0.01 else ""))
    json.dump({"damga": makbuz_hash.damga(), "kume": KUME, "en_iyi": en,
               "ustk": {m_: {k: float(np.mean(ustk[m_][k])) for k in KOLLAR}
                        for m_ in ustk},
               "not": "KONUM skoru toplama kurali. max = bugunku (her yanlis "
                      "konuma 24 piyango bileti). D7'ye BAKILMADI."},
              open(f"results/konum_toplama_{KUME}.json", "w"), indent=1)
    print(f"makbuz -> results/konum_toplama_{KUME}.json")


if __name__ == "__main__":
    main()
