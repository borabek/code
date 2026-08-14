# -*- coding: utf-8 -*-
"""L3/L4/L5 SURUCUSU — yeni konum bilgisi, UCTAN UCA

Taban = temel + kanonik + neg12 (bugun dogrulanan yigin, d6 0.3135).
Kollar: ayna / vida / temas / hepsi
KAPI: uctan uca mikro robot F1'de +0.01. Gecen kol `tam`'da dogrulanir.
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
import connector3d                 # noqa: E402
import kanonik_d7 as K             # noqa: E402
import kanonik_hizalama as KH      # noqa: E402
import p6_karar                    # noqa: E402
import yeni_konum_oz as YKO        # noqa: E402
from kos_p6_ortak import yukle     # noqa: E402
from sina_kume import esle_macar   # noqa: E402

KUME = os.environ.get("YK_KUME", "d6")
KAT_MIN = int(os.environ.get("YK_KAT_MIN", "40"))
MESH = {"d6": "results/_p1_olasilik",
        "tam": "results/_p1_olasilik_brepegit"}[KUME]
NEG_KAT = int(os.environ.get("YK_NEG", "12"))
ITER, LR, YAPRAK, L2R = 200, 0.06, 63, 1.0
NMS = 5.0
KURAL = ("goreli", 0.85, 0.20)
CT = int(connector3d.CONTACT)
if os.environ.get("YK_MOD") == "2":
    # IKINCI TUR. Birinci turda ayna esi +0.0075 ile en guclu yeni sinyal
    # oldu ama +0.01 kapisini GECEMEDI. Kapiyi indirmek sisirmedir; mesru
    # olan, kapiyi degistirmeden BIRLESIM denemektir. Ayna (oznitelik) ile
    # yerel negatif (ornekleme) FARKLI mekanizmalar -- toplamlari kapiyi
    # gecebilir. `temas` de eklenir (+0.0043, ucuncu bagimsiz mekanizma).
    KOLLAR = ("taban", "ayna", "ayna_yerelneg", "ayna_temas",
              "ayna_temas_yerelneg")
else:
    KOLLAR = ("taban", "ayna", "vida", "temas", "hepsi")
DILIM = {"ayna": (0, 5), "vida": (5, 9), "temas": (9, 13),
         "ayna_yerelneg": (0, 5)}


def temel(d):
    return np.hstack([p6_karar.donustur(d["X"]),
                      p6_karar.kaynak_blok(d["kaynak"][d["idx"]])]).astype(
                          np.float32)


def f1(c):
    return 2 * c["tp"] / max(2 * c["tp"] + c["fp"] + c["fn"], 1)


def yap():
    return HistGradientBoostingClassifier(
        max_iter=ITER, learning_rate=LR, max_leaf_nodes=YAPRAK,
        l2_regularization=L2R, random_state=0)


def main():
    t0 = time.time()
    import trimesh
    veri = yukle(KUME, int(os.environ.get("P6_TR", "0")))
    n_ok = 0
    for _i, d in enumerate(veri):
        d["y"] = np.asarray(d["y"], int)
        idx = np.asarray(d["idx"], int)
        P = np.asarray(d["P"], float)
        YD = np.asarray(d["YD"], float)
        tek, ters = np.unique(idx, return_inverse=True)
        Pk = P[tek]
        Dk = YKO.konum_yonu(idx, YD, tek)
        mf = f"{MESH}/{d['pid']}.npz"
        if os.path.exists(mf):
            z = np.load(mf)
            V = np.asarray(z["V"], float)
            F = np.asarray(z["F"], int)
            pb = np.mean([np.asarray(q, float) for q in z["pbs"]], axis=0)
            ag = trimesh.Trimesh(vertices=V, faces=F, process=False)
            isinci = trimesh.ray.ray_triangle.RayMeshIntersector(ag)
            tm = YKO.isin_temas(Pk, Dk, isinci, V, pb[:, CT])
            n_ok += 1
        else:
            tm = np.zeros((len(Pk), 4), np.float32)
        blok = np.hstack([YKO.ayna_esi(Pk, Dk), YKO.vida_cifti(Pk, Dk), tm])
        d["_M"] = temel(d)
        d["_K"] = KH.oznitelik(P[idx], YD,
                               V if os.path.exists(mf) else P).astype(
                                   np.float32)
        d["_B"] = blok[ters].astype(np.float32)
        # `veri.index(d)` KULLANILMAZ: O(n^2) ve sozluk icinde numpy dizisi
        # oldugu icin karsilastirmasi da pahalidir.
        if (_i + 1) % 100 == 0:
            print(f"  oznitelik {_i + 1}/{len(veri)} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    marka = collections.Counter(d["mfg"] for d in veri)
    # KAT TURU (2026-08-13). Varsayilan MARKA-DISI katlar = GORULMEMIS
    # marka kosulu. `YK_RASTGELE_KAT=1` ile RASTGELE 3 kat kullanilir =
    # marka-KARISIK, yani TANIDIK marka kosulu.
    # NEDEN: bu kollar (ayna esi, yerel negatif, dizi uyeligi, isin-temas)
    # gorulmemis markada olculdu ve kapiyi gecemedi. Tanidik markada temsil
    # problemi cok daha kucuk oldugu icin AYNI kollar tutabilir; bunu
    # olcmeden "olu" saymak, kapatma hukmunu sondanin kosuluna kurban
    # etmek olur.
    if os.environ.get("YK_RASTGELE_KAT") == "1":
        _rng = np.random.default_rng(1)
        _pay = _rng.permutation(len(veri)) % 3
        for _i, _d in enumerate(veri):
            _d["mfg"] = f"kat{_pay[_i]}"
        marka = collections.Counter(d["mfg"] for d in veri)
        katlar = [f"kat{i}" for i in range(3)]
        print("  KATLAR RASTGELE (marka-karisik = TANIDIK marka kosulu)",
              flush=True)
    else:
        katlar = [m for m, n in marka.items() if n >= KAT_MIN]
    print(f"{len(veri)} parca | mesh {n_ok} | katlar {katlar} | "
          f"neg={NEG_KAT} ({time.time() - t0:.0f} s)", flush=True)

    def mat(d, kol):
        if kol == "taban":
            return np.hstack([d["_M"], d["_K"]])
        if kol == "hepsi":
            return np.hstack([d["_M"], d["_K"], d["_B"]])
        if kol in ("ayna_temas", "ayna_temas_yerelneg"):
            return np.hstack([d["_M"], d["_K"], d["_B"][:, 0:5],
                              d["_B"][:, 9:13]])
        a, b = DILIM[kol]
        return np.hstack([d["_M"], d["_K"], d["_B"][:, a:b]])

    agg = {k: collections.Counter() for k in KOLLAR}
    for b in katlar:
        ic = [i for i, d in enumerate(veri) if d["mfg"] != b]
        dis = [i for i, d in enumerate(veri) if d["mfg"] == b]
        for kol in KOLLAR:
            n_s = sum(len(veri[i]["y"]) for i in ic)
            M = np.empty((n_s, mat(veri[ic[0]], kol).shape[1]), np.float32)
            PA = np.empty(n_s, np.int32)
            o = 0
            for pi, i in enumerate(ic):
                m_ = mat(veri[i], kol)
                M[o:o + len(m_)] = m_
                PA[o:o + len(m_)] = pi
                o += len(m_)
            Y = np.concatenate([veri[i]["y"] for i in ic])
            rng = np.random.default_rng(0)
            poz, neg = np.where(Y == 1)[0], np.where(Y == 0)[0]
            if kol.endswith("yerelneg"):
                # PARCA-YEREL NEGATIF (L1a, tek basina +0.0034): negatifler
                # pozitifle AYNI parcadan. Toplam boyut kuresel ornekleme
                # ile AYNI kalir, yani kiyas tek degiskenli.
                pl = [poz]
                for pj in np.unique(PA[poz]):
                    yer = np.where(PA == pj)[0]
                    ng = yer[Y[yer] == 0]
                    n_al = min(len(ng), NEG_KAT * int((Y[yer] == 1).sum()))
                    if n_al:
                        pl.append(rng.choice(ng, n_al, replace=False))
                sec = np.concatenate(pl)
            else:
                sec = np.concatenate([poz, rng.choice(
                    neg, min(len(neg), NEG_KAT * max(len(poz), 1)),
                    replace=False)])
            m = yap().fit(M[sec], Y[sec])
            del M, PA
            for i in dis:
                d = veri[i]
                s = m.predict_proba(mat(d, kol))[:, 1]
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
        print(f"  {k:<8}{son[k]:.4f}   {fark:+.4f}"
              + ("  <- KAPI GECTI" if fark >= 0.01 else ""))
    json.dump({"damga": makbuz_hash.damga(), "kume": KUME, "neg": NEG_KAT,
               "toplam": son,
               "not": "L3 ayna esi + L4 vida cifti + L5 isin-temas. Taban = "
                      "temel + kanonik + neg12. Hepsi GT'siz. "
                      "D7'ye BAKILMADI."},
              open(f"results/yeni_konum_{KUME}.json", "w"), indent=1)
    print(f"makbuz -> results/yeni_konum_{KUME}.json")


if __name__ == "__main__":
    main()
