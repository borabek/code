# -*- coding: utf-8 -*-
"""P6 IKI KADEMELI: birinci gecisin tohumlarindan PERIYODIK YAPI ozniteligi.

GEREKCE. D6-ici LOMO'da olculdu: yon secimi cozuldu (kahin farki +0.0057) ama
recall 0.177 / havuz tavani 0.525. Aday-basina bilgi tukendi. Kullanilmamis bilgi
PARCA DUZEYINDE ve olculdu: D6'da >=6 CP'li 1096 parcada GT'lerin **%90.8'i**
parcanin en sik OTELEME VEKTORUYLE baska bir GT'ye ulasiyor.

KADEMELER
  1. P6 ortak siralayici (92 sutun) -> skor
  2. yuksek skorlu secim = TOHUM -> `kafes` oznitelikleri (8 sutun)
  3. ikinci siralayici (100 sutun) -> nihai skor -> secim

SIZINTIYA KARSI IKI ONLEM
  * Tohumlar HER ZAMAN tahminden gelir, GT'den ASLA.
  * Egitim korpusunda birinci kademe skorlari MARKA-KATLI (out-of-fold) uretilir.
    Aksi halde tohumlar kendi egitim verisinde asiri iyi olur, ikinci kademe
    gercekte olmayan bir tohum kalitesine gore ogrenir ve sinavda coker.

AYAR VE OLCUM AYRIMI
  * Butun kural/esik secimi `tam` korpusunun MARKA KATLARINDA yapilir.
  * D6 (SUPU/UPUN/MOR/NIT/UTL/S+S/SE/ONV) TEMIZ OKUMADIR -- ayar icin
    KULLANILMAZ.
  * D7 sinavdir ve bu betik ONA HIC BAKMAZ.
"""
import collections
import json
import os
import pickle
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import makbuz_hash

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
os.environ.setdefault("P6_DIZIN", "results/_p6_oz_m60")
sys.path.insert(0, ".")
import kafes                       # noqa: E402
import kanonik_d7 as K             # noqa: E402
import p6_karar                    # noqa: E402
import urun_genis                  # noqa: E402
import yon_bankasi as YB           # noqa: E402
from kos_p6_ortak import yukle     # noqa: E402
from sina_kume import esle_macar   # noqa: E402

AB = p6_karar.AB
C0 = AB
KURALLAR = ([("mutlak", e) for e in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)] +
            [("goreli", o, t) for o in (0.30, 0.50, 0.70) for t in (0.05, 0.20)])
NMSLER = (2.5, 3.5, 5.0)
TOHUM_KURAL = ("mutlak", 0.60)     # tohum ESIGI ayrica taranmaz: yuksek tutulur
TOHUM_NMS = 5.0
NEG_KAT = int(os.environ.get("P6_NEG_KAT", "8"))
KOLLAR = ("TABAN", "P6", "P6_KAFES")


def yap(tohum=0):
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
        l2_regularization=1.0, random_state=tohum)


def kendi(d):
    return np.where(d["X"][:, C0] == 1.0)[0]


def alt_ornekle(M, Y, kat=NEG_KAT, tohum=0):
    """Tum pozitifler + `kat` katı negatif. 3M satirlik egitimi kaldirilabilir
    kilar; esik zaten sonradan taraniyor, taban oran degismesi zararsiz."""
    if kat <= 0:
        return M, Y
    rng = np.random.default_rng(tohum)
    poz = np.where(Y == 1)[0]
    neg = np.where(Y == 0)[0]
    n = min(len(neg), kat * max(len(poz), 1))
    sec = np.concatenate([poz, rng.choice(neg, n, replace=False)])
    rng.shuffle(sec)
    return M[sec], Y[sec]


def tohumla(d, s):
    """Birinci kademe skorlarindan TOHUM konum/yonleri."""
    return p6_karar.sec_ayrintili(d["P"], d["idx"], d["YD"], s, TOHUM_KURAL,
                                  nms_mm=TOHUM_NMS)[:2]


def kafes_bloku(d, s):
    Pt, Dt = tohumla(d, s)
    return kafes.oznitelik(d["P"][d["idx"]], d["YD"], Pt, Dt)


def puanla(d, s, kural, nms, kol):
    if kol == "TABAN":
        k = kendi(d)
        sk = s[k]
        m = p6_karar.kabul_maskesi(sk, kural)
        if not m.any():
            return np.zeros((0, 3)), np.zeros((0, 3))
        P, D = d["P"][m], d["D"][m]
        T = d["X"][k][m][:, AB + len(YB.OZ_AD):]
        n = np.ones(len(P), bool)
        if nms > 0 and len(P) > 1:
            import wire_gate
            n = wire_gate.kalabalik_maskesi(P, sk[m])
        return P[n], urun_genis.isaret_duzelt(D[n], T[n])
    return p6_karar.sec(d["P"], d["idx"], d["YD"], s, kural, nms_mm=nms)


def olc(veri, skor, kural, nms, kol):
    tp = fp = fn = 0
    tes = []
    for d, s in zip(veri, skor):
        P, D = puanla(d, s, kural, nms, kol)
        a, b, c = esle_macar(P, D, d["G"], d["Gd"], d["diag"], K.YANAL, K.ACI,
                             False, isaretli=True)[:3]
        tp += a; fp += b; fn += c
        tes.append((len(d["G"]),) + esle_macar(
            P, D, d["G"], d["Gd"], d["diag"], max(3.0, 0.06 * d["diag"]),
            180.0, True)[:3])
    return {"robot": 2 * tp / max(2 * tp + fp + fn, 1), "tespit": K.mikro(tes),
            "TP": int(tp), "FP": int(fp), "FN": int(fn),
            "recall": tp / max(tp + fn, 1), "kesinlik": tp / max(tp + fp, 1)}


def oz(d, kol, kafes_blok=None):
    if kol == "TABAN":
        return p6_karar.donustur(d["X"][kendi(d)][:, :AB], "hepsi")
    X = p6_karar.donustur(d["X"])
    if kol == "P6_KAFES":
        return np.hstack([X, kafes_blok])
    return X


def egit(tr, kol, kafes_bloklar=None):
    M = np.vstack([oz(d, kol, None if kafes_bloklar is None else kafes_bloklar[i])
                   for i, d in enumerate(tr)]).astype(np.float32)
    Y = np.concatenate([d["y"][kendi(d)] if kol == "TABAN" else d["y"]
                        for d in tr])
    M, Y = alt_ornekle(M, Y)
    return yap().fit(M, Y)


def skorla(m, veri, kol, kafes_bloklar=None):
    out = []
    for i, d in enumerate(veri):
        X = oz(d, kol, None if kafes_bloklar is None else kafes_bloklar[i])
        p = m.predict_proba(X.astype(np.float32))[:, 1]
        if kol == "TABAN":
            s = np.zeros(len(d["X"]))
            s[kendi(d)] = p
        else:
            s = p
        out.append(np.asarray(s, float))
    return out


def main():
    t0 = time.time()
    tr = yukle("tam", int(os.environ.get("P6_TR", "0")))
    for d in tr:
        d["y"] = np.asarray(d["y"], int)
    marka = collections.Counter(d["mfg"] for d in tr)
    katlar = [m for m, n in marka.items() if n >= 60]
    print(f"tam {len(tr)} parca | markalar {dict(marka)}", flush=True)
    print(f"marka katlari (n>=60): {katlar}  ({time.time() - t0:.0f} s)",
          flush=True)

    # --- 1) BIRINCI KADEME: marka-katli OOF skorlari -----------------------
    oof = [None] * len(tr)
    for b in katlar:
        ic = [i for i, d in enumerate(tr) if d["mfg"] != b]
        dis = [i for i, d in enumerate(tr) if d["mfg"] == b]
        m1 = egit([tr[i] for i in ic], "P6")
        for i, s in zip(dis, skorla(m1, [tr[i] for i in dis], "P6")):
            oof[i] = s
        print(f"  OOF {b}: {len(dis)} parca ({time.time() - t0:.0f} s)",
              flush=True)
    kucuk = [i for i, s in enumerate(oof) if s is None]
    if kucuk:                       # kat olusturamayan kucuk markalar
        m1 = egit([tr[i] for i in range(len(tr)) if i not in set(kucuk)], "P6")
        for i, s in zip(kucuk, skorla(m1, [tr[i] for i in kucuk], "P6")):
            oof[i] = s
        print(f"  OOF kucuk markalar: {len(kucuk)} parca", flush=True)

    kafes_tr = [kafes_bloku(d, s) for d, s in zip(tr, oof)]
    kv = np.vstack(kafes_tr)
    print(f"kafes blogu {kv.shape} | kafes bulunan secenek orani "
          f"{kv[:, 0].mean():.3f} ({time.time() - t0:.0f} s)", flush=True)

    # --- 2) KAT ICINDE kural secimi + kol kiyasi ---------------------------
    top = {k: collections.Counter() for k in KOLLAR}
    ayrinti = {}
    for b in katlar:
        ic = [i for i, d in enumerate(tr) if d["mfg"] != b]
        dis = [i for i, d in enumerate(tr) if d["mfg"] == b]
        TR = [tr[i] for i in ic]
        TE = [tr[i] for i in dis]
        ayrinti[b] = {}
        for kol in KOLLAR:
            kb_tr = [kafes_tr[i] for i in ic] if kol == "P6_KAFES" else None
            kb_te = [kafes_tr[i] for i in dis] if kol == "P6_KAFES" else None
            m = egit(TR, kol, kb_tr)
            s_tr = skorla(m, TR, kol, kb_tr)
            s_te = skorla(m, TE, kol, kb_te)
            en = max(((r, n) for r in KURALLAR for n in NMSLER),
                     key=lambda x: olc(TR, s_tr, x[0], x[1], kol)["robot"])
            r = olc(TE, s_te, en[0], en[1], kol)
            for k in ("TP", "FP", "FN"):
                top[kol][k] += r[k]
            ayrinti[b][kol] = dict(r, kural=list(en[0]), nms=en[1])
        a = ayrinti[b]
        print(f"  {b:<6} n={len(TE):<4} TABAN {a['TABAN']['robot']:.4f} | "
              f"P6 {a['P6']['robot']:.4f} | P6+KAFES {a['P6_KAFES']['robot']:.4f}"
              f"   ({time.time() - t0:.0f} s)", flush=True)

    print(f"\n{'kol':<12} {'robot':>8} {'recall':>8} {'kesinlik':>9} "
          f"{'TP':>7} {'FP':>7} {'FN':>7}")
    son = {}
    for kol in KOLLAR:
        c = top[kol]
        f1 = 2 * c["TP"] / max(2 * c["TP"] + c["FP"] + c["FN"], 1)
        rc = c["TP"] / max(c["TP"] + c["FN"], 1)
        pr = c["TP"] / max(c["TP"] + c["FP"], 1)
        son[kol] = {"robot": f1, "recall": rc, "kesinlik": pr, **dict(c)}
        print(f"{kol:<12} {f1:>8.4f} {rc:>8.4f} {pr:>9.4f} {c['TP']:>7} "
              f"{c['FP']:>7} {c['FN']:>7}")
    print(f"\nP6      - TABAN = {son['P6']['robot'] - son['TABAN']['robot']:+.4f}")
    print(f"P6KAFES - P6    = {son['P6_KAFES']['robot'] - son['P6']['robot']:+.4f}")

    # --- 3) NIHAI MODELLER (tum tam) ---------------------------------------
    en_kol = max(KOLLAR, key=lambda k: son[k]["robot"])
    kural_sayim = collections.Counter(
        (tuple(ayrinti[b][en_kol]["kural"]), ayrinti[b][en_kol]["nms"])
        for b in katlar)
    kural, nms = kural_sayim.most_common(1)[0][0]
    print(f"\nSECILEN kol {en_kol} | kural {kural} | nms {nms} "
          f"(marka katlarinda en sik)")
    m1 = egit(tr, "P6")
    paket = {"kademe1": m1, "kol": en_kol, "kural": list(kural), "nms": nms,
             "zskor": "ab", "AB": AB, "tohum_kural": list(TOHUM_KURAL),
             "tohum_nms": TOHUM_NMS}
    if en_kol == "P6_KAFES":
        paket["kademe2"] = egit(tr, "P6_KAFES", kafes_tr)
    with open("results/p6_kademe2_model.pkl", "wb") as f:
        pickle.dump(paket, f)
    json.dump({"damga": makbuz_hash.damga(), "toplam": son, "marka": ayrinti,
               "n_egitim": len(tr), "katlar": katlar, "secilen": en_kol,
               "kural": list(kural), "nms": nms, "dizin": os.environ["P6_DIZIN"],
               "not": "tam korpusunun MARKA KATLARINDA kural secimi + kol "
                      "kiyasi. Tohumlar OUT-OF-FOLD skorlardan. D6 ve D7'ye "
                      "BAKILMADI."},
              open("results/p6_kademe2_tam.json", "w"), indent=1)
    print(f"makbuz -> results/p6_kademe2_tam.json  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
