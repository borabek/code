# -*- coding: utf-8 -*-
"""C2 -- KONUM HAVUZUNU 5x KUCULT: hedef ILK KEZ tam sayiyla belli

TESHIS (results/konum_auc_d6.json). Ilk-k'nin dolmasi icin gereken KONUM
AUC'si = 1 - k/n_konum:
    NIT : 1 - 24/432 = 0.944 gerek,  0.705 var
    MOR : 1 -  9/497 = 0.982 gerek,  0.900 var
Ayni teshis, secenek duzeyinde YANILTICIYDI: NIT'te secenek AUC 0.8854'un
cogu YON becerisinden geliyor (yon AUC 0.8899), konum becerisi 0.7053.

IKI YOL VAR. AUC'yi yukseltmek (temsil problemi, zor) ya da HAVUZU
KUCULTMEK. Ikincisi sayilarla soyle: NIT 432 -> 80 inerse gereken AUC
0.944 -> 0.70, yani ELIMIZDEKI DEGERE iner. MOR 497 -> 90 aynisini yapar.
Iki markada da ayni kat: 5x.

BU SONDA, GT KULLANMAYAN ucuz suzgeclerin recall/kucultme takasini olcer.
Once TAVAN olculur, sonra insa edilir (bugun mesh normalinde 4 saat bu
sayede kurtarildi).

SUZGECLER (hepsi GT'siz, hepsi tek tek VE birlikte):
  seg      : konumun segmentasyon olasiligi (o konumdaki en yakin tepe)
  ickonum  : konumun icbukeylik olcusu -- delik ICBUKEYDIR
  derin    : konumdan govde icine serbest yol (delik DERINDIR)
  cap      : aciklik capi kablo araliginda mi (1.5 - 12 mm)
  dis      : konum dis yuzeyde mi (govde icinde gomulu degil)

CIKTI: her suzgec ve esik icin (konum recall, kalan konum sayisi,
gereken AUC). "gereken AUC 0.705'in ALTINA inerken recall >= 0.95"
olan bir ayar VARSA kol canlidir.

D7'ye BAKILMAZ.
"""
import collections
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
os.environ.setdefault("P6_DIZIN", "results/_p6_oz_tam4")
sys.path.insert(0, ".")
import connector3d                 # noqa: E402
import isin_ekseni as IE           # noqa: E402
import kanonik_d7 as K             # noqa: E402
from kos_p6_ortak import yukle     # noqa: E402

KUME = os.environ.get("KH_KUME", "d6")
MESH = {"d6": "results/_p1_olasilik",
        "tam": "results/_p1_olasilik_brepegit"}[KUME]
CE = int(connector3d.CABLE_ENTRY)
CT = int(connector3d.CONTACT)
EKSENEL = 40.0
N_ISIN = 24          # ickonum/derinlik icin yon ornegi
SUZ = ("seg", "ickonum", "derin", "cap", "dis")


def _birim(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def fibonacci(n):
    i = np.arange(n) + 0.5
    fi = np.arccos(1 - 2 * i / n)
    te = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(te) * np.sin(fi), np.sin(te) * np.sin(fi),
                 np.cos(fi)]


def konum_oz(kon_P, V, N, agac, p_seg, isinci):
    """her konum icin GT'siz betimleyiciler."""
    n = len(kon_P)
    out = {k: np.zeros(n) for k in SUZ}
    yon = fibonacci(N_ISIN)
    for i, p in enumerate(kon_P):
        kom = agac.query_ball_point(p, 6.0)
        if not kom:
            out["dis"][i] = 1.0
            continue
        out["seg"][i] = float(p_seg[agac.query(p)[1]])
        # ICBUKEYLIK: komsu normallerin p'ye dogru bakma orani
        v = _birim(V[kom] - p[None, :])
        out["ickonum"][i] = float((np.sum(v * N[kom], axis=1) > 0).mean())
        # DERINLIK: govde icine en uzun serbest yol
        d = IE.ilk_carpma(isinci, np.tile(p, (N_ISIN, 1)), yon, cap=40.0)
        out["derin"][i] = float(np.max(d))
        # CAP: en yakin yuzeye uzaklik x2 (aciklik genisligi vekili)
        out["cap"][i] = 2.0 * float(agac.query(p)[0])
        # DIS: 6mm topta komsu var mi ve yuzeye yakin mi
        out["dis"][i] = 1.0 if agac.query(p)[0] <= 2.0 else 0.0
    return out


def main():
    t0 = time.time()
    import pickle
    import trimesh
    veri = yukle(KUME, int(os.environ.get("P6_TR", "0")))
    print(f"{len(veri)} parca", flush=True)
    ist = collections.defaultdict(lambda: collections.defaultdict(list))
    # ONBELLEK: isin atma parca basina ~3 s suruyor (tum kume ~25 dk).
    # Betimleyiciler GT'siz ve modelden bagimsiz oldugu icin bir kez
    # hesaplanip saklanir; esik/formul duzeltmeleri saniyeler icinde
    # tekrar okunur.
    ONB = f"results/_konum_havuzu_onbellek_{KUME}.pkl"
    if os.path.exists(ONB) and os.environ.get("KH_ONBELLEK", "1") == "1":
        ist = pickle.load(open(ONB, "rb"))
        n = sum(len(a["n"]) for a in ist.values())
        print(f"onbellekten okundu: {n} parca", flush=True)
        return _rapor(ist, n)
    n = 0
    for d in veri:
        mf = f"{MESH}/{d['pid']}.npz"
        if not os.path.exists(mf):
            continue
        G = np.asarray(d["G"], float)
        if len(G) < 2:
            continue
        z = np.load(mf)
        V, F = np.asarray(z["V"], float), np.asarray(z["F"], int)
        pb = np.mean([np.asarray(q, float) for q in z["pbs"]], axis=0)
        p_seg = pb[:, CE] + pb[:, CT]
        ag = trimesh.Trimesh(vertices=V, faces=F, process=False)
        N = np.asarray(ag.vertex_normals, float)
        isinci = trimesh.ray.ray_triangle.RayMeshIntersector(ag)
        agac = cKDTree(V)

        P = np.asarray(d["P"], float)
        idx = np.unique(np.asarray(d["idx"], int))
        kon_P = P[idx]
        Gn = _birim(np.asarray(d["Gd"], float))
        # konum DOGRU mu: CARPIM kutusu (yanal 2mm / eksenel 40mm)
        dg = np.zeros(len(kon_P), bool)
        for j in range(len(G)):
            v = kon_P - G[j][None, :]
            al = v @ Gn[j]
            yan = np.linalg.norm(v - al[:, None] * Gn[j][None, :], axis=1)
            dg |= (yan <= K.YANAL) & (np.abs(al) <= EKSENEL)
        if dg.sum() == 0:
            continue
        oz = konum_oz(kon_P, V, N, agac, p_seg, isinci)
        a = ist[d["mfg"]]
        a["gt"].append(len(G))
        a["n"].append(len(kon_P))
        a["dogru"].append(int(dg.sum()))
        for k_ in SUZ:
            a[f"v_{k_}"].append(oz[k_])
            a[f"d_{k_}"].append(dg)
        n += 1
        if n % 40 == 0:
            print(f"  {n} parca ({time.time() - t0:.0f} s)", flush=True)
    pickle.dump(dict(ist), open(ONB, "wb"))
    return _rapor(ist, n)


def _rapor(ist, n):
    print(f"\n{n} parca | GT'siz konum suzgecleri: recall / kucultme takasi")
    out = {}
    for m_ in sorted(ist, key=lambda x: -sum(ist[x]["gt"])):
        a = ist[m_]
        k_ort = float(np.mean(a["dogru"]))
        n_ort = float(np.mean(a["n"]))
        print(f"\n--- {m_}  (GT {sum(a['gt'])} | konum {n_ort:.0f} | "
              f"dogru konum {k_ort:.1f} | gereken AUC "
              f"{1 - k_ort / n_ort:.4f})")
        print(f"  {'suzgec':<10}{'yuzdelik':>10}{'recall':>9}{'kalan':>8}"
              f"{'kucultme':>10}{'gereken AUC':>13}")
        out[m_] = {}
        for k_ in SUZ:
            hepsi_v = np.concatenate(a[f"v_{k_}"])
            for q in (10, 25, 50, 75):
                esik = np.percentile(hepsi_v, q)
                tut_n = tut_d = top_n = top_d = 0
                for v_, d_ in zip(a[f"v_{k_}"], a[f"d_{k_}"]):
                    t_ = v_ >= esik
                    tut_n += int(t_.sum()); tut_d += int((t_ & d_).sum())
                    top_n += len(v_); top_d += int(d_.sum())
                if not tut_n or not top_d:
                    continue
                # DENOMINATOR DUZELTMESI. Ilk kosuda `kal` ve `ger` marka
                # parca sayisina degil GLOBAL parca sayisina (n) bolunuyordu;
                # marka basina rapor edilen kucultme ve gereken AUC bu yuzden
                # olceksizdi (NIT'te recall 0.996 + 9.4x kucultme oldugu halde
                # gereken AUC 0.9339 -> 0.9269 gorunuyordu, oysa dusmeliydi).
                n_marka = max(len(a["n"]), 1)
                rec = tut_d / top_d
                kal = tut_n / n_marka
                ger = 1 - tut_d / max(tut_n, 1)
                print(f"  {k_:<10}{q:>10}{rec:>9.3f}{kal:>8.0f}"
                      f"{n_ort / max(kal, 1e-9):>10.2f}x{ger:>12.4f}")
                out[m_][f"{k_}_{q}"] = {"recall": rec, "kalan": kal,
                                        "gereken_auc": ger}
    json.dump({"kume": KUME, "marka": out,
               "not": "GT'siz konum suzgecleri: recall / havuz kucultme "
                      "takasi. Hedef: gereken AUC 0.705 altina inerken "
                      "recall >= 0.95. D7'ye BAKILMADI."},
              open(f"results/konum_havuzu_{KUME}.json", "w"), indent=1)
    print(f"\nmakbuz -> results/konum_havuzu_{KUME}.json")
    print("OKUMA: recall >= 0.95 iken 'gereken AUC' 0.705'in ALTINA inen bir")
    print("       satir varsa kol CANLI; yoksa havuz kucultme bu yoldan olmaz")


if __name__ == "__main__":
    main()
