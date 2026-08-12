# -*- coding: utf-8 -*-
"""YENI-1 -- KAYIP FONKSIYONU: focal / sinif-dengeli, S7 teshisine dogrudan nisan

NEDEN. S7 cokusun sebebini olctu: POZ-NEG SKOR AYRIMI. UPUN'da 0.847,
NIT'te 0.050 -- yani NIT'te dogru secenek yanlistan yalnizca 0.05 daha
yuksek puan aliyor. Bugune kadar denenen her sey OZNITELIK ekliyordu; kayip
fonksiyonuna HIC dokunulmadi. Oysa ayrimi dogrudan sekillendiren sey odur.

DORT KOL (hepsi AYNI oznitelikler, AYNI katlar, AYNI kural aramasi):
  taban        : bugunku HGB (alt-orneklenmis, duz log-loss)
  agirlik      : alt-ornekleme YOK, POZITIFE agirlik (class-balanced)
  focal        : kolay negatifleri bastiran odak kaybi (gamma)
  agirlik+odak : ikisi birden

HGB'de dogrudan focal yok; `sample_weight` ile YAKLASILIR:
  w_i = (1-p_i)^gamma   -- p_i bir ON GECISTEN gelen olasilik
Yani once bir taban model, sonra onun kolay buldugu orneklerin agirligi
dusurulerek IKINCI model. Bu, focal kaybin pratik karsiligidir.

KAPI: +0.01 (mikro robot F1, `tam` marka katlari mantigi).
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

KUME = os.environ.get("KD_KUME", "d6")
KAT_MIN = int(os.environ.get("KD_KAT_MIN", "40"))
ITER = int(os.environ.get("P6_ITER", "200"))
NEG_KAT = int(os.environ.get("P6_NEG_KAT", "6"))
GAMMA = float(os.environ.get("KD_GAMMA", "2.0"))
NMS = 5.0
KURALLAR = ([("mutlak", e) for e in (0.20, 0.40, 0.60, 0.80, 0.90, 0.95)] +
            [("goreli", o, t) for o in (0.50, 0.70, 0.85, 0.95)
             for t in (0.05, 0.20, 0.40)])


def temel(d):
    return np.hstack([p6_karar.donustur(d["X"]),
                      p6_karar.kaynak_blok(d["kaynak"][d["idx"]])]).astype(
                          np.float32)


def yap(**kw):
    return HistGradientBoostingClassifier(
        max_iter=ITER, learning_rate=0.06, max_leaf_nodes=63,
        l2_regularization=1.0, random_state=0, **kw)


def puanla(veri, skor, kural):
    per = collections.defaultdict(lambda: [0, 0, 0])
    for d, s in zip(veri, skor):
        P, D = p6_karar.sec(d["P"], d["idx"], d["YD"], s, kural, nms_mm=NMS)
        a, b, c = esle_macar(P, D, d["G"], d["Gd"], d["diag"], K.YANAL, K.ACI,
                             False, isaretli=True)[:3]
        q = per[d["mfg"]]
        q[0] += a; q[1] += b; q[2] += c
    T = [sum(q[i] for q in per.values()) for i in range(3)]
    pm = {m: 2 * q[0] / max(2 * q[0] + q[1] + q[2], 1) for m, q in per.items()}
    return {"robot": 2 * T[0] / max(2 * T[0] + T[1] + T[2], 1),
            "makro": float(np.mean(list(pm.values()))) if pm else 0.0,
            "TP": T[0], "FP": T[1], "FN": T[2]}


def main():
    t0 = time.time()
    veri = yukle(KUME, int(os.environ.get("P6_TR", "0")))
    for d in veri:
        d["y"] = np.asarray(d["y"], int)
        d["_M"] = temel(d)
    marka = collections.Counter(d["mfg"] for d in veri)
    katlar = [m for m, n in marka.items() if n >= KAT_MIN]
    print(f"{len(veri)} parca | katlar {katlar} | gamma={GAMMA}", flush=True)

    KOLLAR = ("taban", "agirlik", "focal", "agirlik_focal")
    top = {k: collections.Counter() for k in KOLLAR}
    kat_sonuc = {}
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
        alt = np.concatenate([poz, rng.choice(
            neg, min(len(neg), NEG_KAT * max(len(poz), 1)), replace=False)])

        modeller = {}
        # 1) TABAN: alt-ornekleme + duz kayip
        modeller["taban"] = yap().fit(M[alt], Y[alt])
        # 2) AGIRLIK: TUM veri, pozitife sinif agirligi
        w = np.where(Y == 1, len(neg) / max(len(poz), 1), 1.0)
        modeller["agirlik"] = yap().fit(M, Y, sample_weight=w)
        # 3) FOCAL -- agirliklar KAT-DISI olasiliktan.
        # ILK DENEMEM BOZUKTU: agirliklari taban modelin KENDI EGITIM
        # verisindeki olasiligindan hesaplamistim. Model o ornekleri
        # ezberledigi icin p~1 cikiyor, (1-p)^gamma sifira iniyor ve geriye
        # yalniz gurultu kaliyor -- olculdu: UPUN'da 0.6341 -> 0.0127.
        # Dogrusu: 2 katli ic bolme ile KAT-DISI olasilik.
        def oof_p(Mx, Yx):
            pr = np.zeros(len(Yx))
            yari = np.random.default_rng(0).permutation(len(Yx))
            for h in (yari[:len(yari) // 2], yari[len(yari) // 2:]):
                digeri = np.setdiff1d(np.arange(len(Yx)), h)
                mm = yap().fit(Mx[digeri], Yx[digeri])
                pr[h] = mm.predict_proba(Mx[h])[:, 1]
            return pr

        p0 = oof_p(M[alt], Y[alt])
        pt = np.where(Y[alt] == 1, p0, 1.0 - p0)
        wf = np.power(np.clip(1.0 - pt, 1e-6, 1.0), GAMMA)
        wf = wf / max(wf.mean(), 1e-9)          # olcegi koru
        modeller["focal"] = yap().fit(M[alt], Y[alt], sample_weight=wf)
        # 4) AGIRLIK + FOCAL (ayni kat-disi olasilik, alt-orneklem uzerinde)
        wa = np.where(Y[alt] == 1, len(neg) / max(len(poz), 1), 1.0) * wf
        wa = wa / max(wa.mean(), 1e-9)
        modeller["agirlik_focal"] = yap().fit(M[alt], Y[alt], sample_weight=wa)
        del M

        kat_sonuc[b] = {}
        for ad, m in modeller.items():
            s_ic = [m.predict_proba(veri[i]["_M"])[:, 1] for i in ic]
            s_dis = [m.predict_proba(veri[i]["_M"])[:, 1] for i in dis]
            ar = np.random.default_rng(0).choice(
                len(ic), min(120, len(ic)), replace=False)
            AR = [veri[ic[j]] for j in ar]
            AS = [s_ic[j] for j in ar]
            kural = max(KURALLAR, key=lambda k: puanla(AR, AS, k)["makro"])
            r = puanla([veri[i] for i in dis], s_dis, kural)
            for k_ in ("TP", "FP", "FN"):
                top[ad][k_] += r[k_]
            kat_sonuc[b][ad] = r["robot"]
        print(f"  {b:<6} " + " | ".join(
            f"{a} {kat_sonuc[b][a]:.4f}" for a in KOLLAR)
            + f"  ({time.time() - t0:.0f} s)", flush=True)

    son = {}
    for ad in KOLLAR:
        c = top[ad]
        son[ad] = 2 * c["TP"] / max(2 * c["TP"] + c["FP"] + c["FN"], 1)
    print(f"\n=== KAYIP DENEMESI (MIKRO) ===")
    for ad in KOLLAR:
        fark = son[ad] - son["taban"]
        print(f"  {ad:<15}{son[ad]:.4f}   {fark:+.4f}"
              + ("  <- KAPI GECTI" if fark >= 0.01 else ""))
    json.dump({"damga": makbuz_hash.damga(), "kume": KUME, "gamma": GAMMA,
               "toplam": son, "kat": kat_sonuc,
               "not": "Kayip fonksiyonu kollari: alt-ornekleme / sinif "
                      "agirligi / focal / ikisi. AYNI oznitelik, kat, kural "
                      "aramasi. D7'ye BAKILMADI."},
              open(f"results/kayip_denemesi_{KUME}.json", "w"), indent=1)
    print(f"makbuz -> results/kayip_denemesi_{KUME}.json")


if __name__ == "__main__":
    main()
