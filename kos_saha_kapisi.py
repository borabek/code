# -*- coding: utf-8 -*-
"""SAHA GUVEN KAPISI: "robot hangi isaretlere KENDI BASINA guvenebilir?"

SORU. Urun her parcada bir dizi CP isaretliyor ama hepsi ayni kalitede degil.
Robotun otonom davranabilmesi icin "bu isaret %X dogrudur" diyebilmek gerekir.
Bugun GLB'deki kirmizi/turuncu ayrimi `robot_conf_auto=0.5` + 3 oy gibi KEYFI
bir esikle yapiliyor -- olculmus bir kesinlige BAGLI DEGIL.

BU BETIK. `tam` MARKA KATLARINDA (her kat: bir marka disarida, gorulmemis
marka kosulu) her SECILEN tahmin icin (skor, dogru mu) kaydeder ve
kesinlik-kapsama egrisini cikarir. Boylece "ONAYLI" katmani icin esik, olculmus
kesinlige gore secilir.

NEDEN D7 DEGIL. D7 SINAV kumesidir ve butcesi 2 okumadir. Isletme esigi ayarlamak
bir okumayi HARCAR ve dahasi esigi sinava UYDURMAK olur. Kat olcumu ayni soruyu
(gorulmemis marka) sinavi harcamadan yanitlar.

DURUSTLUK NOTU. Buradaki kesinlik, SECILEN tahminler icindir. Kapsama = ONAYLI
isaretlerin GT'ye orani (yani isin ne kadari otonom yapilabilir). Kesinligi
yukseltmek kapsamayi DUSURUR; ikisi ayni anda buyumez.

Cikti: results/saha_kapisi_tam.json + ekrana tablo.
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
os.environ.setdefault("P6_DIZIN", "results/_p6_oz_tam3")
sys.path.insert(0, ".")
import kanonik_d7 as K             # noqa: E402
import p6_karar                    # noqa: E402
from kos_p6_ortak import yukle     # noqa: E402
from sina_kume import esle_macar   # noqa: E402

NMS = 5.0
ITER = int(os.environ.get("P6_ITER", "200"))
NEG_KAT = int(os.environ.get("P6_NEG_KAT", "6"))
KAT_MIN = int(os.environ.get("P6_KAT_MIN", "200"))
# Genis bir kural: ONAYLI katmani zaten skor esigiyle daraltilacak, bu yuzden
# secim kurali GENIS tutulur (yuksek recall) ve daraltmayi esik yapar.
KURAL = ("goreli", 0.50, 0.05)
HEDEFLER = (0.70, 0.80, 0.90, 0.95)


def temel(d):
    return np.hstack([p6_karar.donustur(d["X"]),
                      p6_karar.kaynak_blok(d["kaynak"][d["idx"]])])


def main():
    t0 = time.time()
    veri = []
    for kume in os.environ.get("P6_KUME", "tam,d6").split(","):
        for d in yukle(kume.strip(), int(os.environ.get("P6_TR", "0"))):
            d["_kume"] = kume.strip()
            veri.append(d)
    for d in veri:
        d["y"] = np.asarray(d["y"], int)
    marka = collections.Counter(d["mfg"] for d in veri)
    katlar = [m for m, n in marka.items() if n >= KAT_MIN]
    print(f"{len(veri)} parca | katlar {katlar} ({time.time() - t0:.0f} s)",
          flush=True)

    kayit = []          # (skor, dogru_mu)
    gt_top = 0
    for b in katlar:
        ic = [i for i, d in enumerate(veri) if d["mfg"] != b]
        dis = [i for i, d in enumerate(veri) if d["mfg"] == b]
        M = np.vstack([temel(veri[i]) for i in ic]).astype(np.float32)
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
            d = veri[i]
            s = m.predict_proba(temel(d).astype(np.float32))[:, 1]
            P, D, _, sk = p6_karar.sec_ayrintili(
                d["P"], d["idx"], d["YD"], s, KURAL, nms_mm=NMS)
            gt_top += len(d["G"])
            if not len(P):
                continue
            tp, fp, fn, bilgi = esle_macar(P, D, d["G"], d["Gd"], d["diag"],
                                           K.YANAL, K.ACI, False, isaretli=True)
            dogru = np.zeros(len(P), bool)
            for e in bilgi["eslesme"]:
                dogru[e[0]] = True
            kayit.extend(zip(np.asarray(sk, float).tolist(), dogru.tolist()))
        print(f"  {b:<6} biriken tahmin {len(kayit)} ({time.time() - t0:.0f} s)",
              flush=True)

    if not kayit:
        sys.exit("hic tahmin yok")
    sk = np.array([a for a, _ in kayit])
    dg = np.array([b for _, b in kayit], bool)
    sira = np.argsort(-sk)
    dg_s = dg[sira]
    sk_s = sk[sira]
    kum_tp = np.cumsum(dg_s)
    n = np.arange(1, len(dg_s) + 1)
    kesinlik = kum_tp / n
    kapsama = kum_tp / max(gt_top, 1)

    print(f"\ntoplam tahmin {len(sk)} | toplam GT {gt_top} | "
          f"ham kesinlik {dg.mean():.4f}")
    print(f"\n{'hedef':<8}{'esik':>8}{'ONAYLI':>9}{'kesinlik':>10}"
          f"{'kapsama':>9}")
    oneri = {}
    for h in HEDEFLER:
        uy = np.where(kesinlik >= h)[0]
        # en BUYUK k: esik dustukce kesinlik dusuyor; hedefi saglayan en genis
        # onek aranir (en az 20 tahmin olsun ki gurultu olmasin)
        uy = uy[uy >= 19]
        if not len(uy):
            print(f"{h:<8.2f}{'-':>8}{'-':>9}{'ULASILMIYOR':>10}{'-':>9}")
            oneri[str(h)] = None
            continue
        k = int(uy.max())
        oneri[str(h)] = {"esik": float(sk_s[k]), "n": int(k + 1),
                         "kesinlik": float(kesinlik[k]),
                         "kapsama": float(kapsama[k])}
        print(f"{h:<8.2f}{sk_s[k]:>8.3f}{k + 1:>9d}{kesinlik[k]:>10.4f}"
              f"{kapsama[k]:>9.4f}")

    json.dump({"damga": makbuz_hash.damga(), "dizin": os.environ["P6_DIZIN"],
               "katlar": katlar, "n_parca": len(veri), "n_tahmin": len(sk),
               "gt_toplam": int(gt_top), "ham_kesinlik": float(dg.mean()),
               "kural": list(KURAL), "nms": NMS, "oneri": oneri,
               "not": "GORULMEMIS MARKA katlarinda (LOMO) kesinlik-kapsama. "
                      "ONAYLI katmani esigi buradan secilir. D7'ye BAKILMADI. "
                      "Kapsama = ONAYLI isaret / toplam GT."},
              open("results/saha_kapisi_tam.json", "w"), indent=1)
    print("\nmakbuz -> results/saha_kapisi_tam.json")


if __name__ == "__main__":
    main()
