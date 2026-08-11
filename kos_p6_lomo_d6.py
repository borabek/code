# -*- coding: utf-8 -*-
"""P6 ERKEN OKUMA: D6-ICI LOMO (marka-birak-disarida) + ESIK x NMS taramasi.

NEDEN: `tam` korpusunun secenek oznitelikleri hala cikiyor. Ama D6'nin 8 markasi
elde ve LOMO ile ADIL bir kiyas kurulabilir -- her kivrimda ayni parcalar, ayni
egitim buyuklugu, TEK DEGISKEN oznitelik+secim kurali.

KOLLAR
  TABAN     aday basina TEK satir (kendi yonu), A+B (67 sutun), esik + NMS +
            isaret duzeltme                      -- dagitilan kuralin ta kendisi
  P6        aday basina TUM yon secenekleri, A+B+C+D (92 sutun), ortak skor +
            acgozlu secim + konum NMS
  P6_KAHIN  P6 secimi, ama SECILEN adaylarin yonu KAHIN'den. Tavan degil TESHIS:
            "yonu mu kaciriyoruz, konumu mu?"

ILK KOSU BULGUSU (2026-08-11):
  TABAN 0.2093 -> P6 0.2649 (+0.0556), yon kahini yalniz +0.0064 EKLIYOR.
  Yani secilen adaylarda yon secimi ZATEN neredeyse mukemmel; bankanin actigi
  +0.2392'lik tavan HIC SECILMEYEN adaylarda duruyor. Recall %17 / kesinlik %58
  -- yani cok AZ tahmin uretiyoruz.
  Olculdu: D6 GT'lerinin %16.1'inin en yakin komsusu 5mm'den yakin. NMS yaricapi
  5mm ise KOMSU KONTAKLARI birbirini bastiriyor ve recall'a tavan koyuyor.
  Bu yuzden esik ve NMS BIRLIKTE taranir; ikisi de EGITIM markalarinda secilir.

Esik/NMS HER KOL ve HER KIVRIM icin EGITIM markalarinda secilir; disarida
birakilan markada TARANMAZ. Aksi halde sayi sisik olur.

Bu bir ON OKUMADIR: 468 parca / 8 marka kucuk bir taban ve `tam` korpusunun
buyuklugunu temsil etmez. Karar `tam` ile egitilmis modelde verilir.
"""
import collections
import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import makbuz_hash

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, ".")
import kanonik_d7 as K             # noqa: E402
import p6_karar                    # noqa: E402
import urun_genis                  # noqa: E402
import wire_gate                   # noqa: E402
import yon_bankasi as YB           # noqa: E402
from kos_p6_ortak import yukle     # noqa: E402
from sina_kume import esle_macar   # noqa: E402

AB = p6_karar.AB
C0 = AB                            # C blogunun ilk sutunu = k_kendi
# KARAR KURALLARI. Ilk taramada MUTLAK esik neredeyse her kivrimda IZGARANIN EN
# UST degerini (0.50) sectti -- yani optimum SINIRDAYDI ve gercek optimumu
# bulamiyordum. Izgara yukari acildi ve urunun kendi GORELI kurali
# (`p1c_esik.maske`: parca-ici en yuksek skorun `oran` kati VE `taban` ustu) de
# aday kurallar arasina alindi.
KURALLAR = ([("mutlak", e) for e in
             (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)] +
            [("goreli", o, t) for o in (0.30, 0.50, 0.70)
             for t in (0.05, 0.15, 0.30)])
NMSLER = (0.0, 1.5, 2.5, 3.5, 5.0)
KOLLAR = ("TABAN", "P6", "P6_KAHIN")


def yap():
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
        l2_regularization=1.0, random_state=0)


def kendi(d):
    """Aday basina TEK satir: kendi yon secenegi (C blogunun k_kendi=1 satiri)."""
    return np.where(d["X"][:, C0] == 1.0)[0]


def _nms(P, s, mm):
    """Skor sirali konum bastirma. mm<=0 ise bastirma YOK."""
    if mm <= 0 or len(P) < 2:
        return np.ones(len(P), bool)
    tut = np.zeros(len(P), bool)
    alinan = []
    for i in np.argsort(-s):
        if alinan and float(np.min(np.linalg.norm(
                np.asarray(alinan) - P[i], axis=1))) < mm:
            continue
        alinan.append(P[i])
        tut[i] = True
    return tut


def yon_kahini(d, ai):
    """Secilen her aday icin KENDI bankasindan en iyi yon (KAHIN, teshis icin)."""
    if not len(ai):
        return np.zeros((0, 3))
    G = np.asarray(d["G"], float)
    Gn = YB.birim(d["Gd"])
    out = []
    for i in ai:
        Y = d["YD"][np.where(d["idx"] == i)[0]]
        w = d["P"][i][None] - G
        al = (w * Gn).sum(-1)
        yan = np.linalg.norm(w - al[:, None] * Gn, axis=-1)
        uy = (yan <= K.YANAL) & (np.abs(al) <= 40.0)
        if not uy.any():
            out.append(Y[0])
            continue
        an = np.degrees(np.arccos(np.clip(YB.birim(Y) @ Gn[uy].T, -1.0, 1.0)))
        out.append(Y[int(np.unravel_index(np.argmin(an), an.shape)[0])])
    return np.asarray(out, float).reshape(-1, 3)


def puanla(d, s, esik, nms, kol):
    if kol == "TABAN":
        k = kendi(d)
        sk = s[k]
        m = p6_karar.kabul_maskesi(sk, esik)
        if not m.any():
            return np.zeros((0, 3)), np.zeros((0, 3))
        P, D = d["P"][m], d["D"][m]
        T = d["X"][k][m][:, AB + len(YB.OZ_AD):]
        n = _nms(P, sk[m], nms)
        return P[n], urun_genis.isaret_duzelt(D[n], T[n])
    P, D, ai, _sc = p6_karar.sec_ayrintili(d["P"], d["idx"], d["YD"], s, esik,
                                           nms_mm=nms)
    if kol == "P6_KAHIN":
        return P, yon_kahini(d, ai)
    return P, D


def olc(veri, skor, esik, nms, kol):
    tp = fp = fn = 0
    tes = []
    for d, s in zip(veri, skor):
        P, D = puanla(d, s, esik, nms, kol)
        a, b, c = esle_macar(P, D, d["G"], d["Gd"], d["diag"], K.YANAL, K.ACI,
                             False, isaretli=True)[:3]
        tp += a; fp += b; fn += c
        tes.append((len(d["G"]),) + esle_macar(
            P, D, d["G"], d["Gd"], d["diag"], max(3.0, 0.06 * d["diag"]),
            180.0, True)[:3])
    return {"robot": 2 * tp / max(2 * tp + fp + fn, 1), "tespit": K.mikro(tes),
            "TP": tp, "FP": fp, "FN": fn}


def egit(tr, taban):
    if taban:
        M = np.vstack([p6_karar.donustur(d["X"][kendi(d)][:, :AB], "hepsi")
                       for d in tr])
        Y = np.concatenate([d["y"][kendi(d)] for d in tr])
    else:
        M = np.vstack([p6_karar.donustur(d["X"]) for d in tr])
        Y = np.concatenate([d["y"] for d in tr])
    return yap().fit(M, Y)


def skorla(m, veri, taban):
    out = []
    for d in veri:
        if taban:
            k = kendi(d)
            s = np.zeros(len(d["X"]))
            s[k] = m.predict_proba(
                p6_karar.donustur(d["X"][k][:, :AB], "hepsi"))[:, 1]
        else:
            s = m.predict_proba(p6_karar.donustur(d["X"]))[:, 1]
        out.append(np.asarray(s, float))
    return out


def main():
    dev = yukle("d6", int(os.environ.get("P6_DEV", "0")))
    for d in dev:
        d["y"] = np.asarray(d["y"], int)
    marka = collections.Counter(d["mfg"] for d in dev)
    kivrimlar = [m for m, n in marka.items() if n >= 8]
    print(f"D6 {len(dev)} parca | LOMO kivrimi {kivrimlar}", flush=True)

    top = {k: collections.Counter() for k in KOLLAR}
    ayrinti = {}
    for b in kivrimlar:
        tr = [d for d in dev if d["mfg"] != b]
        te = [d for d in dev if d["mfg"] == b]
        modeller = {False: egit(tr, False), True: egit(tr, True)}
        ayrinti[b] = {}
        for kol in KOLLAR:
            tb = (kol == "TABAN")
            m = modeller[tb]
            s_tr, s_te = skorla(m, tr, tb), skorla(m, te, tb)
            en = max(((e, n) for e in KURALLAR for n in NMSLER),
                     key=lambda en: olc(tr, s_tr, en[0], en[1], kol)["robot"])
            r = olc(te, s_te, en[0], en[1], kol)
            for k in ("TP", "FP", "FN"):
                top[kol][k] += r[k]
            ayrinti[b][kol] = dict(r, kural=list(en[0]), nms=en[1])
        t, p, o = (ayrinti[b]["TABAN"], ayrinti[b]["P6"], ayrinti[b]["P6_KAHIN"])
        print(f"  {b:<6} n={len(te):<4} TABAN {t['robot']:.4f} -> "
              f"P6 {p['robot']:.4f} ({p['kural']}, nms {p['nms']})  "
              f"({p['robot'] - t['robot']:+.4f})  [yon kahini {o['robot']:.4f}]",
              flush=True)

    print(f"\n{'kol':<10} {'robot':>8} {'TP':>6} {'FP':>6} {'FN':>6} {'recall':>8} {'kesinlik':>9}")
    son = {}
    for kol in KOLLAR:
        c = top[kol]
        f1 = 2 * c["TP"] / max(2 * c["TP"] + c["FP"] + c["FN"], 1)
        rc = c["TP"] / max(c["TP"] + c["FN"], 1)
        pr = c["TP"] / max(c["TP"] + c["FP"], 1)
        son[kol] = {"robot": f1, "recall": rc, "kesinlik": pr, **dict(c)}
        print(f"{kol:<10} {f1:>8.4f} {c['TP']:>6} {c['FP']:>6} {c['FN']:>6} "
              f"{rc:>8.4f} {pr:>9.4f}")
    print(f"\nYON SECIM KAYBI (P6_KAHIN - P6): "
          f"{son['P6_KAHIN']['robot'] - son['P6']['robot']:+.4f}")
    d = son["P6"]["robot"] - son["TABAN"]["robot"]
    art = sum(1 for b in kivrimlar
              if ayrinti[b]["P6"]["robot"] > ayrinti[b]["TABAN"]["robot"])
    print(f"P6 - TABAN = {d:+.4f} | {art}/{len(kivrimlar)} markada ARTI")
    json.dump({"damga": makbuz_hash.damga(), "toplam": son, "marka": ayrinti,
               "n_parca": len(dev), "kivrim": kivrimlar,
               "kurallar": [list(k) for k in KURALLAR], "nmsler": list(NMSLER),
               "not": "D6-ICI LOMO on okumasi. Esik VE NMS her kol/kivrim icin "
                      "EGITIM markalarinda secildi. Onbellek havuzu -- urunun "
                      "canli havuzu DEGIL."},
              open("results/p6_lomo_d6.json", "w"), indent=1)
    print("makbuz -> results/p6_lomo_d6.json")


if __name__ == "__main__":
    main()
