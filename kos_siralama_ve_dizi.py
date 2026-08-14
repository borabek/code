# -*- coding: utf-8 -*-
"""L1 + L2 — PARCA ICI SIRALAMA ve DIZI UYELIGI

TESHIS (results/konum_auc_d6.json). Baglayan sey PARCA ICI KONUM SIRALAMASI:
NIT'te yon AUC 0.8899 ama konum AUC 0.7053 (gereken 0.944). Havuz kucultme
kapandi (en iyi 1.33x), HPO tukendi (yalniz neg=12, +0.0088/+0.0054).
Geriye iki sey kaliyor: YENI BILGI ve YENI HEDEF FONKSIYONU.

--- L1: HEDEF FONKSIYONU (yeni bilgi gerektirmez) ---

L1a YEREL NEGATIF. Bugun negatifler TUM egitim parcalarindan KURESEL
cekiliyor; model "genel olarak CP neye benzer" ogreniyor. Oysa is "BU
parcanin icinde hangisi daha iyi". Negatifleri pozitifle AYNI parcadan
cekmek sinirin parca-ici karsitliga oturmasini saglar.

L1b LAMBDA AGIRLIGI. Her pozitif icin, AYNI parcada onu gecen negatif
sayisi = o pozitifin parca-ici AUC'ye verdigi zarar. Agirliklar oradan
kurulur (LambdaMART'in agirlik duzeyindeki karsiligi).
DIKKAT: agirliklar KAT-DISI skordan hesaplanir. Focal denemesinde
ornek-ici olasilik kullanmak modeli sifirlamisti (p~1 -> agirlik~0).

--- L2: YENI BILGI (dizi uyeligi) ---

Klemens girisleri DUZENLI BIR DIZI olusturur; montaj delikleri, test
noktalari, ray yuvalari olusturmaz. "Bu aciklik, >=3 benzer aciklikatan
olusan duzenli dizinin uyesi mi" sorusu konum duzeyinde ayirt edicidir.

BU, DUSEN KAFES KOLU DEGILDIR. O kol konum URETIYORDU ve yon kopyalamada
coktu (uctan uca -0.0645). Bu kol mevcut adaylara OZNITELIK verir; o
cokus mekanizmasi burada gecerli degil.

KOLLAR: taban / yerel_neg / lambda / dizi / dizi+yerel_neg
KAPI: uctan uca mikro robot F1'de +0.01. Gecen kol `tam`'da dogrulanir.
D7'ye BAKILMAZ.
"""
import collections
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier

import makbuz_hash

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
os.environ.setdefault("P6_DIZIN", "results/_p6_oz_tam4")
sys.path.insert(0, ".")
import kanonik_d7 as K             # noqa: E402
import kanonik_hizalama as KH      # noqa: E402
import p6_karar                    # noqa: E402
from kos_p6_ortak import yukle     # noqa: E402
from sina_kume import esle_macar   # noqa: E402

KUME = os.environ.get("SD_KUME", "d6")
KAT_MIN = int(os.environ.get("SD_KAT_MIN", "40"))
MESH = {"d6": "results/_p1_olasilik",
        "tam": "results/_p1_olasilik_brepegit"}[KUME]
NEG_KAT = int(os.environ.get("SD_NEG", "12"))     # tur-2'nin kazanani
ITER, LR, YAPRAK, L2R = 200, 0.06, 63, 1.0
NMS = 5.0
KURAL = ("goreli", 0.85, 0.20)
KOM = 12          # dizi aramasinda bakilacak komsu sayisi
ADIM_K = 3        # +-3 adim
DIZI_TOL = 1.0    # mm


def temel(d):
    return np.hstack([p6_karar.donustur(d["X"]),
                      p6_karar.kaynak_blok(d["kaynak"][d["idx"]])]).astype(
                          np.float32)


def kanonik_blok(d):
    mf = f"{MESH}/{d['pid']}.npz"
    V = (np.asarray(np.load(mf)["V"], float)
         if os.path.exists(mf) else np.asarray(d["P"], float))
    return KH.oznitelik(d["P"][d["idx"]], d["YD"], V).astype(np.float32)


def dizi_oz(Pk_tekil):
    """KONUM basina dizi uyeligi (GT'siz). doner: (n_konum, 5)

    Her konum p icin, en yakin KOM komsusunun her biri bir ADIM adayidir.
    p + k*adim (k = -3..3) noktalarinda baska konum var mi diye bakilir;
    en cok uye toplayan adim kazanir.
    """
    n = len(Pk_tekil)
    out = np.zeros((n, 5), np.float32)
    if n < 3:
        return out
    agac = cKDTree(Pk_tekil)
    kk = min(KOM + 1, n)
    _, kom = agac.query(Pk_tekil, k=kk)
    for i in range(n):
        p = Pk_tekil[i]
        en_uye, en_adim, en_kalinti, en_yer, yon_say = 1, 0.0, 0.0, 0.0, 0
        for j in kom[i][1:]:
            adim = Pk_tekil[j] - p
            s = float(np.linalg.norm(adim))
            if s < 1e-6:
                continue
            hedef = p[None, :] + np.arange(-ADIM_K, ADIM_K + 1)[:, None] * \
                adim[None, :]
            uz, _ = agac.query(hedef)
            var = uz <= DIZI_TOL
            u = int(var.sum())
            if u >= 3:
                yon_say += 1
            if u > en_uye:
                en_uye = u
                en_adim = s
                en_kalinti = float(uz[var].mean()) if var.any() else 0.0
                # p'nin dizi icindeki normalize konumu (uc mu, orta mi)
                yer = np.where(var)[0]
                en_yer = float(abs(ADIM_K - yer.mean()) / max(ADIM_K, 1))
        out[i] = (en_uye, en_adim, en_kalinti, en_yer, yon_say)
    return out


def f1(c):
    return 2 * c["tp"] / max(2 * c["tp"] + c["fp"] + c["fn"], 1)


def yap():
    return HistGradientBoostingClassifier(
        max_iter=ITER, learning_rate=LR, max_leaf_nodes=YAPRAK,
        l2_regularization=L2R, random_state=0)


def main():
    t0 = time.time()
    veri = yukle(KUME, int(os.environ.get("P6_TR", "0")))
    for d in veri:
        d["y"] = np.asarray(d["y"], int)
        base = np.hstack([temel(d), kanonik_blok(d)])
        d["_M"] = base
        idx = np.asarray(d["idx"], int)
        tek, ters = np.unique(idx, return_inverse=True)
        dz = dizi_oz(np.asarray(d["P"], float)[tek])
        d["_D"] = np.hstack([base, dz[ters]]).astype(np.float32)
    marka = collections.Counter(d["mfg"] for d in veri)
    katlar = [m for m, n in marka.items() if n >= KAT_MIN]
    print(f"{len(veri)} parca | katlar {katlar} | neg={NEG_KAT} "
          f"({time.time() - t0:.0f} s)", flush=True)

    KOLLAR = ("taban", "yerel_neg", "lambda", "dizi", "dizi_yerel")
    agg = {k: collections.Counter() for k in KOLLAR}
    for b in katlar:
        ic = [i for i, d in enumerate(veri) if d["mfg"] != b]
        dis = [i for i, d in enumerate(veri) if d["mfg"] == b]
        for kol in KOLLAR:
            alan = "_D" if kol.startswith("dizi") else "_M"
            n_s = sum(len(veri[i]["y"]) for i in ic)
            M = np.empty((n_s, veri[ic[0]][alan].shape[1]), np.float32)
            PA = np.empty(n_s, np.int32)          # parca kimligi
            o = 0
            for pi, i in enumerate(ic):
                m_ = veri[i][alan]
                M[o:o + len(m_)] = m_
                PA[o:o + len(m_)] = pi
                o += len(m_)
            Y = np.concatenate([veri[i]["y"] for i in ic])
            rng = np.random.default_rng(0)
            poz = np.where(Y == 1)[0]
            if kol in ("yerel_neg", "dizi_yerel"):
                # PARCA-YEREL NEGATIF: her parcada, o parcanin pozitif
                # sayisinin NEG_KAT kati negatif -- KENDI parcasindan.
                sec = [poz]
                for pi in np.unique(PA[poz]):
                    yer = np.where(PA == pi)[0]
                    ng = yer[Y[yer] == 0]
                    n_al = min(len(ng), NEG_KAT * int((Y[yer] == 1).sum()))
                    if n_al:
                        sec.append(rng.choice(ng, n_al, replace=False))
                sec = np.concatenate(sec)
            else:
                neg = np.where(Y == 0)[0]
                sec = np.concatenate([poz, rng.choice(
                    neg, min(len(neg), NEG_KAT * max(len(poz), 1)),
                    replace=False)])
            w = np.ones(len(sec), np.float32)
            if kol == "lambda":
                # KAT-DISI skor (2 katli ic bolme) -> parca ici sira hatasi
                p_oof = np.zeros(len(sec))
                yari = rng.permutation(len(sec))
                for h in (yari[:len(yari) // 2], yari[len(yari) // 2:]):
                    dg = np.setdiff1d(np.arange(len(sec)), h)
                    p_oof[h] = yap().fit(M[sec][dg], Y[sec][dg]) \
                        .predict_proba(M[sec][h])[:, 1]
                for pi in np.unique(PA[sec]):
                    yer = np.where(PA[sec] == pi)[0]
                    yp = yer[Y[sec][yer] == 1]
                    yn = yer[Y[sec][yer] == 0]
                    if not len(yp) or not len(yn):
                        continue
                    # her pozitifi kac negatif geciyor
                    for q in yp:
                        w[q] = 1.0 + (p_oof[yn] > p_oof[q]).sum()
                    # her negatif kac pozitifi geciyor
                    for q in yn:
                        w[q] = 1.0 + (p_oof[yp] < p_oof[q]).sum()
                w = w / max(w.mean(), 1e-9)
            m = yap().fit(M[sec], Y[sec], sample_weight=w)
            del M, PA
            for i in dis:
                d = veri[i]
                s = m.predict_proba(d[alan])[:, 1]
                P, D = p6_karar.sec(d["P"], d["idx"], d["YD"], s, KURAL,
                                    nms_mm=NMS)
                tp, fp, fn = esle_macar(P, D, d["G"], d["Gd"], d["diag"],
                                        K.YANAL, K.ACI, False,
                                        isaretli=True)[:3]
                c = agg[kol]
                c["tp"] += tp; c["fp"] += fp; c["fn"] += fn
        print(f"  {b} bitti ({time.time() - t0:.0f} s)", flush=True)

    son = {k: f1(agg[k]) for k in KOLLAR}
    print(f"\n=== TABAN {son['taban']:.4f} ===")
    for k in KOLLAR[1:]:
        fark = son[k] - son["taban"]
        print(f"  {k:<12}{son[k]:.4f}   {fark:+.4f}"
              + ("  <- KAPI GECTI" if fark >= 0.01 else ""))
    json.dump({"damga": makbuz_hash.damga(), "kume": KUME, "neg": NEG_KAT,
               "toplam": son,
               "not": "L1 parca-ici siralama (yerel negatif / lambda "
                      "agirligi) + L2 dizi uyeligi oznitelikleri. Taban = "
                      "temel + kanonik + neg12. D7'ye BAKILMADI."},
              open(f"results/siralama_dizi_{KUME}.json", "w"), indent=1)
    print(f"makbuz -> results/siralama_dizi_{KUME}.json")


if __name__ == "__main__":
    main()
