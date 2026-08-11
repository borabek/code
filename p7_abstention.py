# -*- coding: utf-8 -*-
"""P7: 0.75 GELMEDEN URUN DEGERI -- AUTO kipinde kesinlik >=0.95, kalani REVIEW.

MANTIK: robot F1 0.28 iken sistemi "her CP'yi otonom isle" diye kullanmak yanlis islem
uretir. Ama gate skoru bir GUVEN olcusudur: yuksek skorlu adaylarin kesinligi cok daha
yuksek olabilir. O halde iki kipli bir urun mumkundur:
  AUTO   : kesinlik >=0.95 olan skor bandi -- robot dogrudan islesin
  REVIEW : geri kalani -- operator onaylasin

OLCULEN: her AUTO esiginde (1) kesinlik, (2) KAPSAMA (GT'nin yuzde kaci AUTO'ya dustu),
(3) operatore kalan is. Iki metrikte AYRI: tespit ve robot-hazir.

REVIEW YONLENDIRMESI ayrica olculur: cok-CP parcalari ve silindiri olmayan parcalar
(SE/NIT morfolojisi) zaten cokuyor -- onlari parca duzeyinde REVIEW'a yollamak
AUTO'nun kesinligini yukseltir mi?
"""
import collections
import glob
import io
import json
import os
import pickle
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"; os.environ["WG_TOPO"] = "1"; os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import d6_kayit

MAKBUZ = "results/p7_abstention.json"
HEDEF_KESINLIK = 0.95


def main():
    import protokol
    protokol.tez_dogrula()
    import brep_snap
    import p3c_eksen_secici as P3C
    import wire_gate
    from p1c_esik import maske
    from sina_kume import esle_macar

    sv = d6_kayit.sinav()
    kayit = d6_kayit.yukle(set(sv["pidler"]))
    with open("results/wire_gate_v5.pkl", "rb") as f:
        gate = pickle.load(f)
    with open("results/_d6_silindirler.pkl", "rb") as f:
        cyl = pickle.load(f)
    with open("results/p3c_eksen_secici.pkl", "rb") as f:
        sec = pickle.load(f)["clf"]

    # Her aday icin: (gate skoru, tespit_dogru_mu, robot_dogru_mu, rejim, silindir_var_mi)
    kayitlar = []
    gt_top = 0
    for pid, r in kayit.items():
        G = np.asarray(r["G"], float); Gd = np.asarray(r["Gd"], float)
        gt_top += len(G)
        rj = "cok" if r["n"] >= 8 else "dusuk"
        pak = P3C.parca_adaylari(r, gate, cyl)
        if pak is None:
            continue
        P, D, Sk, komsu = pak
        cy = cyl.get(pid) or []
        sil = sum(1 for c in cy if brep_snap.R_MIN <= c["radius"] <= brep_snap.R_MAX)
        P2 = P.copy(); D2 = D.copy()
        for i in range(len(P)):
            opt = P3C.secenekler(cy, P[i], D[i], r["diag"], float(Sk[i]), komsu, len(P))
            if len(opt) > 1:
                s = sec.predict_proba(np.asarray([o[2] for o in opt], float))[:, 1]
                j = int(np.argmax(s))
                if j != 0:
                    P2[i] = opt[j][0]; D2[i] = opt[j][1]
        P, D = P2, D2
        _t, _f, _n, bt = esle_macar(P, D, G, Gd, r["diag"], 0.0, 180.0, True)
        _t2, _f2, _n2, br = esle_macar(P, D, G, Gd, r["diag"], 2.0, 10.0, False,
                                       isaretli=True)
        t_ok = {e[0] for e in bt["eslesme"]}
        r_ok = {e[0] for e in br["eslesme"]}
        for i in range(len(P)):
            kayitlar.append((float(Sk[i]), int(i in t_ok), int(i in r_ok), rj,
                             int(sil > 0), pid))
    A = np.asarray([(k[0], k[1], k[2], k[4]) for k in kayitlar], float)
    RJ = np.asarray([k[3] for k in kayitlar])
    print(f"D6: {len(kayit)} parca | {gt_top} GT CP | {len(A)} gate-gecen aday\n")

    print(f"{'esik':<7}{'aday':>7}{'kesinlik_T':>12}{'kesinlik_R':>12}"
          f"{'kapsama_T':>11}{'kapsama_R':>11}")
    tablo = []
    for e in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        m = A[:, 0] >= e
        n = int(m.sum())
        if n == 0:
            continue
        kt = float(A[m, 1].mean()); kr = float(A[m, 2].mean())
        ct = float(A[m, 1].sum() / max(gt_top, 1)); cr = float(A[m, 2].sum() / max(gt_top, 1))
        print(f"{e:<7.2f}{n:>7}{kt:>12.4f}{kr:>12.4f}{ct:>11.4f}{cr:>11.4f}")
        tablo.append({"esik": e, "aday": n, "kesinlik_tespit": kt, "kesinlik_robot": kr,
                      "kapsama_tespit": ct, "kapsama_robot": cr})

    uy_t = [t for t in tablo if t["kesinlik_tespit"] >= HEDEF_KESINLIK]
    uy_r = [t for t in tablo if t["kesinlik_robot"] >= HEDEF_KESINLIK]
    print(f"\nAUTO kipi (kesinlik >= {HEDEF_KESINLIK}):")
    if uy_t:
        b = max(uy_t, key=lambda t: t["kapsama_tespit"])
        print(f"  TESPIT: esik {b['esik']:.2f} -> kesinlik {b['kesinlik_tespit']:.3f}, "
              f"GT'nin %{100*b['kapsama_tespit']:.1f}'i AUTO")
    else:
        en = max(tablo, key=lambda t: t["kesinlik_tespit"])
        print(f"  TESPIT: {HEDEF_KESINLIK} ULASILAMIYOR -- en yuksek kesinlik "
              f"{en['kesinlik_tespit']:.3f} (esik {en['esik']:.2f})")
    if uy_r:
        b = max(uy_r, key=lambda t: t["kapsama_robot"])
        print(f"  ROBOT : esik {b['esik']:.2f} -> kesinlik {b['kesinlik_robot']:.3f}, "
              f"GT'nin %{100*b['kapsama_robot']:.1f}'i AUTO")
    else:
        en = max(tablo, key=lambda t: t["kesinlik_robot"])
        print(f"  ROBOT : {HEDEF_KESINLIK} ULASILAMIYOR -- en yuksek kesinlik "
              f"{en['kesinlik_robot']:.3f} (esik {en['esik']:.2f})")

    # PARCA DUZEYI YONLENDIRME: cok-CP ve silindirsiz parcalari REVIEW'a yolla
    print(f"\nPARCA DUZEYI REVIEW YONLENDIRMESI (esik 0.50):")
    m0 = A[:, 0] >= 0.50
    for ad, msk in (("hepsi", m0),
                    ("yalniz dusuk-CP", m0 & (RJ == "dusuk")),
                    ("dusuk-CP + silindiri VAR", m0 & (RJ == "dusuk") & (A[:, 3] > 0))):
        if msk.sum() == 0:
            continue
        print(f"  {ad:<26}aday {int(msk.sum()):>5}  kesinlik_T {A[msk,1].mean():.3f}  "
              f"kesinlik_R {A[msk,2].mean():.3f}  kapsama_R "
              f"%{100*A[msk,2].sum()/max(gt_top,1):.1f}")
    with io.open(MAKBUZ, "w", encoding="utf-8") as f:
        json.dump({"gt": gt_top, "aday": len(A), "tablo": tablo,
                   "hedef_kesinlik": HEDEF_KESINLIK}, f, indent=1)
    print(f"\nmakbuz -> {MAKBUZ}")


if __name__ == "__main__":
    main()
