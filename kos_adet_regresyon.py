# -*- coding: utf-8 -*-
"""VII.2d + II.2 -- ADET TAHMINI (ogrenmeli) ve ADET-KISITLI SECIM, UCTAN UCA

NEDEN. Ust-k kahin deneyi olctu: dogru CP sayisini bilmek UPUN'da F1'i
0.5359 -> 0.6892 yapiyor (**+0.1533**), SUPU'da +0.0314. Bugune kadar
olctugum en buyuk tek kazanc. Adet GEOMETRIDEN okunamadi (kipsel yaricapli
silindir sayimi curudu: hata 25-84). Ama OGRENMELI olarak HIC denenmedi.

BU BETIK iki seyi birden yapar:
  1. Parca ozniteliklerinden CP SAYISINI tahmin eden bir regresyon egitir
     (marka disarida = gorulmemis marka kosulu).
  2. Tahmini adetle ILK-k secimi yapip UCTAN UCA robot F1 olcer, mevcut
     esik kuralina karsi.

KURESEL UYGULANMAZ: ust-k deneyi NIT'te adet vermenin FP patlattigini
gosterdi. Bu yuzden UC kol birden olculur:
     esik           : bugunku kural
     ilk-k_tahmin   : her parcada tahmini adet kadar
     REJIM          : guven yuksekse ilk-k, dusukse esik

D7'ye BAKILMAZ.
"""
import collections
import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor

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

KUME = os.environ.get("AR_KUME", "d6")
KAT_MIN = int(os.environ.get("AR_KAT_MIN", "40"))
ITER = int(os.environ.get("P6_ITER", "200"))
NEG_KAT = int(os.environ.get("P6_NEG_KAT", "6"))
NMS = 5.0
KURAL = ("goreli", 0.85, 0.20)


def temel(d):
    return np.hstack([p6_karar.donustur(d["X"]),
                      p6_karar.kaynak_blok(d["kaynak"][d["idx"]])]).astype(
                          np.float32)


def parca_oz(d, s):
    """ADET tahmini icin PARCA duzeyi oznitelikler (GT KULLANILMAZ)."""
    s = np.asarray(s, float)
    idx = np.asarray(d["idx"], int)
    P = np.asarray(d["P"], float)
    n_aday = len(np.unique(idx))
    kut = P.max(0) - P.min(0) if len(P) else np.zeros(3)
    q = np.percentile(s, [50, 75, 90, 95, 99]) if len(s) else np.zeros(5)
    return np.array([
        len(s), n_aday, float(d["diag"]),
        kut[0], kut[1], kut[2], float(np.prod(np.sort(kut)[-2:])),
        s.mean(), s.std(), s.max(), *q,
        float((s >= 0.5).sum()), float((s >= 0.8).sum()),
        float((s >= 0.9).sum()), float((s >= 0.95).sum()),
        float((s >= 0.5).sum()) / max(n_aday, 1),
    ], float)


def sec_ustk(d, s, k):
    P = d["P"][d["idx"]]
    YD = d["YD"]
    sira = np.argsort(-np.asarray(s))
    sp, sd = [], []
    for j in sira:
        if len(sp) >= k:
            break
        p = P[j]
        if sp and min(np.linalg.norm(np.asarray(sp) - p, axis=1)) < NMS:
            continue
        sp.append(p)
        sd.append(YD[j])
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
    adet_tah = [None] * len(veri)
    for b in katlar:
        ic = [i for i, d in enumerate(veri) if d["mfg"] != b]
        dis = [i for i, d in enumerate(veri) if d["mfg"] == b]
        # --- 1) secenek skorlayici
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
        for i in ic + dis:
            veri[i]["_s"] = m.predict_proba(veri[i]["_M"])[:, 1]
        for i in dis:
            oof[i] = veri[i]["_s"]
        # --- 2) ADET regresyonu (log uzayinda)
        XA = np.vstack([parca_oz(veri[i], veri[i]["_s"]) for i in ic])
        YA = np.log1p([len(veri[i]["G"]) for i in ic])
        rg = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=0).fit(XA, YA)
        for i in dis:
            adet_tah[i] = float(np.expm1(rg.predict(
                parca_oz(veri[i], veri[i]["_s"])[None])[0]))
        print(f"  {b} bitti ({time.time() - t0:.0f} s)", flush=True)

    # --- degerlendirme
    agg = collections.defaultdict(lambda: collections.Counter())
    hata = collections.defaultdict(list)
    for d, s, kt in zip(veri, oof, adet_tah):
        if s is None:
            continue
        G = np.asarray(d["G"], float)
        Gd = np.asarray(d["Gd"], float)
        dg = d["diag"]
        a = agg[d["mfg"]]
        a["gt"] += len(G)
        k_ger = len(G)
        k_tah = max(1, int(round(kt)))
        hata[d["mfg"]].append(abs(k_tah - k_ger))
        # esik
        P, D = p6_karar.sec(d["P"], d["idx"], d["YD"], s, KURAL, nms_mm=NMS)
        tp, fp, fn = esle_macar(P, D, G, Gd, dg, K.YANAL, K.ACI, False,
                                isaretli=True)[:3]
        a["e_tp"] += tp; a["e_fp"] += fp; a["e_fn"] += fn
        # ilk-k tahmin
        P2, D2 = sec_ustk(d, s, k_tah)
        tp2, fp2, fn2 = esle_macar(P2, D2, G, Gd, dg, K.YANAL, K.ACI, False,
                                   isaretli=True)[:3]
        a["k_tp"] += tp2; a["k_fp"] += fp2; a["k_fn"] += fn2
        # REJIM: skor ayrimi yuksekse ilk-k, dusukse esik
        # REJIM KURALI: adet tahmini KUCUKSE ilk-k'ya guven.
        # Gerekce olculdu: adet hatasi UPUN 0.0, SUPU/MOR 1.0, NIT 18.0 --
        # yani tahmin YOGUN parcada cokuyor. Skor yayilimina bakan ilk kural
        # bunu yakalayamiyordu (+0.0042). Esik cevreden ayarlanir.
        if k_tah <= int(os.environ.get("AR_MAKS_K", "8")):
            a["r_tp"] += tp2; a["r_fp"] += fp2; a["r_fn"] += fn2
        else:
            a["r_tp"] += tp; a["r_fp"] += fp; a["r_fn"] += fn

    def f1(t, f, n):
        return 2 * t / max(2 * t + f + n, 1)

    print(f"\n{'marka':<7}{'GT':>7}{'adet |hata|':>13}{'ESIK':>9}"
          f"{'ILK-k':>9}{'REJIM':>9}")
    out = {}
    T = collections.Counter()
    for m_ in sorted(agg, key=lambda x: -agg[x]["gt"]):
        a = agg[m_]
        for k_ in a:
            T[k_] += a[k_]
        e = f1(a["e_tp"], a["e_fp"], a["e_fn"])
        kk = f1(a["k_tp"], a["k_fp"], a["k_fn"])
        rr = f1(a["r_tp"], a["r_fp"], a["r_fn"])
        h = float(np.median(hata[m_]))
        out[m_] = {"gt": a["gt"], "adet_ortanca_hata": h,
                   "esik": e, "ilk_k": kk, "rejim": rr}
        print(f"{m_:<7}{a['gt']:>7}{h:>13.1f}{e:>9.4f}{kk:>9.4f}{rr:>9.4f}")
    e = f1(T["e_tp"], T["e_fp"], T["e_fn"])
    kk = f1(T["k_tp"], T["k_fp"], T["k_fn"])
    rr = f1(T["r_tp"], T["r_fp"], T["r_fn"])
    print(f"{'TOPLAM':<7}{T['gt']:>7}{'':>13}{e:>9.4f}{kk:>9.4f}{rr:>9.4f}")
    print(f"\nILK-k  - ESIK = {kk - e:+.4f}")
    print(f"REJIM  - ESIK = {rr - e:+.4f}")
    json.dump({"damga": makbuz_hash.damga(), "kume": KUME, "marka": out,
               "toplam": {"esik": e, "ilk_k": kk, "rejim": rr},
               "not": "Adet OGRENMELI tahmin (parca oznitelikleri, marka "
                      "disarida). Uctan uca robot F1. D7'ye BAKILMADI."},
              open(f"results/adet_regresyon_{KUME}.json", "w"), indent=1)
    print(f"makbuz -> results/adet_regresyon_{KUME}.json")


if __name__ == "__main__":
    main()
