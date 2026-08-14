# -*- coding: utf-8 -*-
"""BIRLESIK KOL: gecen iki kaldiraci BIRLIKTE kos ve uctan uca olc

Bugun 19 kol olculdu, ikisi ise yaradi:
  kanonik blogu           +0.0151  (parcayi kendi PCA cercevesine oturtur)
  yayilim, rejim kapili   +0.0394  (coken markada kafesle uretim, d6)

Ikisi FARKLI hatalari duzeltiyor: kanonik marka bagimsizligini, yayilim
coken markadaki TEKRARLARI. Bu betik ikisini BIRLIKTE kosar.

DORT KOL:
  taban            : bugunku (temel oznitelikler, esik kurali)
  kanonik          : + kanonik blogu
  yayilim          : taban + coken parcada kafes uretimi
  birlesik         : ikisi birden

REJIM: yayilim yalniz "coken" parcada devreye girer. Olcut, parcanin skor
ayrimi (S7'nin poz-neg olcusunun urun surumu): ayrim dusukse model o parcada
sirala(ya)miyor demektir.

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
import kanonik_d7 as K                 # noqa: E402
import kanonik_hizalama as KH          # noqa: E402
import p6_karar                        # noqa: E402
from kos_p6_ortak import yukle         # noqa: E402
from sina_kume import esle_macar       # noqa: E402
from sonda_kafes_v2 import kafes_ara   # noqa: E402

KUME = os.environ.get("BK_KUME", "d6")
KAT_MIN = int(os.environ.get("BK_KAT_MIN", "40"))
ITER = int(os.environ.get("P6_ITER", "200"))
NEG_KAT = int(os.environ.get("P6_NEG_KAT", "6"))
MESH = {"d6": "results/_p1_olasilik",
        "tam": "results/_p1_olasilik_brepegit"}[KUME]
NMS = 5.0
KURAL = ("goreli", 0.85, 0.20)
AYRIM_ESIK = float(os.environ.get("BK_AYRIM", "0.30"))
YAKIN_R = 2.0


def _birim(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def temel(d):
    return np.hstack([p6_karar.donustur(d["X"]),
                      p6_karar.kaynak_blok(d["kaynak"][d["idx"]])]).astype(
                          np.float32)


def kanonik_blok(d):
    mf = f"{MESH}/{d['pid']}.npz"
    V = (np.asarray(np.load(mf)["V"], float)
         if os.path.exists(mf) else np.asarray(d["P"], float))
    return KH.oznitelik(d["P"][d["idx"]], d["YD"], V).astype(np.float32)


def yayilim_sec(d, s, k_hedef):
    """Kafesle uret, yakin seceneklerden KIPSEL yonu ver, skorla ilk k."""
    P = np.asarray(d["P"], float)
    idx = np.asarray(d["idx"], int)
    YD = _birim(np.asarray(d["YD"], float))
    Pu = np.unique(np.round(P, 3), axis=0)
    bul = kafes_ara(Pu)
    if not bul:
        return np.zeros((0, 3)), np.zeros((0, 3))
    uret = np.vstack([b[2] for b in bul])
    ust = np.argsort(-s)[:max(30, k_hedef)]
    Y0 = YD[ust]
    cos = np.clip(Y0 @ Y0.T, -1, 1)
    oy = (np.degrees(np.arccos(cos)) <= K.ACI).sum(1)
    kipsel = Y0[int(np.argmax(oy))]
    d_ua = np.linalg.norm(uret[:, None, :] - P[None, :, :], axis=-1)
    skor = np.full(len(uret), -1.0)
    for u in range(len(uret)):
        ad = np.where(d_ua[u] <= YAKIN_R)[0]
        if not len(ad):
            continue
        m_ = np.isin(idx, ad)
        if m_.any():
            skor[u] = s[m_].max()
    g = skor >= 0
    if not g.any():
        return np.zeros((0, 3)), np.zeros((0, 3))
    U, S_ = uret[g], skor[g]
    sp = []
    for j in np.argsort(-S_):
        if len(sp) >= k_hedef:
            break
        p = U[j]
        if sp and min(np.linalg.norm(np.asarray(sp) - p, axis=1)) < NMS:
            continue
        sp.append(p)
    SP = np.asarray(sp).reshape(-1, 3)
    return SP, np.tile(kipsel, (len(SP), 1))


def f1(t, f, n):
    return 2 * t / max(2 * t + f + n, 1)


def main():
    t0 = time.time()
    veri = yukle(KUME, int(os.environ.get("P6_TR", "0")))
    for d in veri:
        d["y"] = np.asarray(d["y"], int)
        d["_M"] = temel(d)
        d["_K"] = kanonik_blok(d)
    marka = collections.Counter(d["mfg"] for d in veri)
    katlar = [m for m, n in marka.items() if n >= KAT_MIN]
    print(f"{len(veri)} parca | katlar {katlar} | ayrim esigi {AYRIM_ESIK}",
          flush=True)

    skor = {"taban": [None] * len(veri), "kanonik": [None] * len(veri)}
    for b in katlar:
        ic = [i for i, d in enumerate(veri) if d["mfg"] != b]
        dis = [i for i, d in enumerate(veri) if d["mfg"] == b]
        for ad in ("taban", "kanonik"):
            def mat(i):
                return (veri[i]["_M"] if ad == "taban"
                        else np.hstack([veri[i]["_M"], veri[i]["_K"]]))
            n_s = sum(len(veri[i]["y"]) for i in ic)
            M = np.empty((n_s, mat(ic[0]).shape[1]), np.float32)
            o = 0
            for i in ic:
                m_ = mat(i)
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
                skor[ad][i] = m.predict_proba(mat(i))[:, 1]
        print(f"  {b} bitti ({time.time() - t0:.0f} s)", flush=True)

    KOLLAR = ("taban", "kanonik", "yayilim", "birlesik")
    agg = collections.defaultdict(lambda: collections.Counter())
    for d in veri:
        if skor["taban"][veri.index(d)] is None:
            continue
        i = veri.index(d)
        G = np.asarray(d["G"], float)
        Gd = np.asarray(d["Gd"], float)
        dg = d["diag"]
        a = agg[d["mfg"]]
        a["gt"] += len(G)
        for kol in KOLLAR:
            s = skor["kanonik" if kol in ("kanonik", "birlesik")
                     else "taban"][i]
            ayrim = float(np.percentile(s, 99) - np.median(s))
            kafesli = kol in ("yayilim", "birlesik") and ayrim < AYRIM_ESIK
            if kafesli:
                P, D = yayilim_sec(d, s, max(len(G), 4))
            else:
                P, D = p6_karar.sec(d["P"], d["idx"], d["YD"], s, KURAL,
                                    nms_mm=NMS)
            tp, fp, fn = esle_macar(P, D, G, Gd, dg, K.YANAL, K.ACI, False,
                                    isaretli=True)[:3]
            a[f"{kol}_tp"] += tp
            a[f"{kol}_fp"] += fp
            a[f"{kol}_fn"] += fn

    print(f"\n{'marka':<7}{'GT':>7}" + "".join(f"{k:>11}" for k in KOLLAR))
    T = collections.Counter()
    out = {}
    for m_ in sorted(agg, key=lambda x: -agg[x]["gt"]):
        a = agg[m_]
        for k_ in a:
            T[k_] += a[k_]
        r = {kol: f1(a[f"{kol}_tp"], a[f"{kol}_fp"], a[f"{kol}_fn"])
             for kol in KOLLAR}
        r["gt"] = a["gt"]
        out[m_] = r
        print(f"{m_:<7}{a['gt']:>7}" +
              "".join(f"{r[k]:>11.4f}" for k in KOLLAR))
    son = {kol: f1(T[f"{kol}_tp"], T[f"{kol}_fp"], T[f"{kol}_fn"])
           for kol in KOLLAR}
    print(f"{'TOPLAM':<7}{T['gt']:>7}" +
          "".join(f"{son[k]:>11.4f}" for k in KOLLAR))
    print("\n=== TABANA GORE ===")
    for kol in KOLLAR[1:]:
        fark = son[kol] - son["taban"]
        print(f"  {kol:<12}{son[kol]:.4f}   {fark:+.4f}"
              + ("  <- KAZANC" if fark > 0 else ""))
    json.dump({"damga": makbuz_hash.damga(), "kume": KUME,
               "ayrim_esik": AYRIM_ESIK, "toplam": son, "marka": out,
               "not": "Gecen iki kaldirac BIRLIKTE: kanonik blogu + rejim "
                      "kapili kafes yayilimi. D7'ye BAKILMADI."},
              open(f"results/birlesik_kol_{KUME}.json", "w"), indent=1)
    print(f"makbuz -> results/birlesik_kol_{KUME}.json")


if __name__ == "__main__":
    main()
