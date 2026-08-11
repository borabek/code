"""K2.1 TAVAN SONDASI: periyodiklik yayilimi kac FN kurtarabilir?

Kolu KURMADAN once tavanini olc (bugunun dersi: K1.7'de zincir tavani %39 ciktI ve
kol bosunaydi).

Iki soru:
  1. GT CP'leri gercekten periyodik mi? (parca ici komsu-mesafe dagilimi)
  2. KAHIN yayilim: TP'lerden periyodu bilseydik, FN'lerin kacini uretebilirdik?

Kahin yayilim TANIMI (bilerek IYIMSER -- tavan olcuyoruz):
  - parcanin TP'lerinden en iyi dogruyu (yon) ve adimi (periyot) AL
  - o dogru boyunca +-N adim ilerle
  - uretilen noktalarin tespit toleransi icinde bir FN'e dusenlerini say
Gercek kol bundan DAHA IYI olamaz.
"""
import os
import sys
import json

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, ".")

import d6_kayit                      # noqa: E402
import wire_gate                     # noqa: E402
import pickle                        # noqa: E402
from p1c_esik import maske           # noqa: E402
from sina_kume import esle_macar     # noqa: E402

CIKTI = "results/k21_periyodiklik_tavani.json"
ADIM_N = 6          # dogru boyunca +-6 adim


def periyot_ve_yon(X):
    """TP kumesinden ana eksen (PCA) + o eksendeki medyan komsu adimi."""
    if len(X) < 3:
        return None, None
    C = X - X.mean(0)
    _u, _s, Vt = np.linalg.svd(C, full_matrices=False)
    yon = Vt[0]
    t = np.sort(C @ yon)
    fark = np.diff(t)
    fark = fark[fark > 1e-6]
    if not len(fark):
        return None, None
    return yon, float(np.median(fark))


def main():
    sv = d6_kayit.sinav()
    kayit = d6_kayit.yukle(set(sv["pidler"]))
    gate = pickle.load(open("results/wire_gate_v5.pkl", "rb"))

    # 1) GT'nin kendi periyodikligi
    duzenlilik = []          # medyan adim / std adim  (buyuk = duzenli)
    # 2) kahin yayilim
    fn_top, fn_kurtarilan = 0, 0
    parca = 0

    for pid, r in kayit.items():
        G = np.asarray(r.get("G", []), float)
        if len(G) < 3:
            continue
        M = d6_kayit.x58(r)
        if M is None or r.get("P") is None or not len(r["P"]):
            continue
        parca += 1

        # --- GT duzenliligi
        yon_g, adim_g = periyot_ve_yon(G)
        if adim_g:
            t = np.sort((G - G.mean(0)) @ yon_g)
            f = np.diff(t)
            f = f[f > 1e-6]
            if len(f) >= 2 and np.median(f) > 0:
                duzenlilik.append(float(np.std(f) / np.median(f)))

        # --- urunun TP/FN'leri (gate sonrasi, tespit toleransi)
        sk = np.asarray(wire_gate.karar_skoru(gate, M), float)
        k = maske(sk, 0.40, 0.30)
        if not k.any():
            continue
        P = np.asarray(r["P"], float)[k]
        D = np.asarray(r["Pd"], float)[k]
        Gd = np.asarray(r["Gd"], float)
        _t, _f, _n, bi = esle_macar(P, D, G, Gd, r["diag"], 0.0, 180.0, True)
        eslesen_gt = {gi for (_pi, gi, *_x) in bi["eslesme"]}
        tp_nokta = np.asarray([G[gi] for gi in sorted(eslesen_gt)], float)
        fn_idx = [i for i in range(len(G)) if i not in eslesen_gt]
        if not fn_idx:
            continue
        fn_top += len(fn_idx)
        if len(tp_nokta) < 3:
            continue

        # --- KAHIN yayilim
        yon, adim = periyot_ve_yon(tp_nokta)
        if yon is None or adim <= 1e-6:
            continue
        uret = []
        for b in tp_nokta:
            for s in range(-ADIM_N, ADIM_N + 1):
                if s:
                    uret.append(b + s * adim * yon)
        if not uret:
            continue
        U = np.asarray(uret, float)
        tol = max(3.0, 0.06 * r["diag"])
        for i in fn_idx:
            if np.min(np.linalg.norm(U - G[i], axis=1)) <= tol:
                fn_kurtarilan += 1

    dz = np.asarray(duzenlilik)
    print(f"parca {parca}\n")
    print("1) GT PERIYODIKLIGI (adim std / adim medyan; 0 = kusursuz periyodik)")
    print(f"   medyan {np.median(dz):.3f} | p25 {np.percentile(dz,25):.3f} | "
          f"p75 {np.percentile(dz,75):.3f}")
    print(f"   'duzenli' (<0.25) parca orani: %{100*(dz<0.25).mean():.1f}")
    print(f"   'cok duzensiz' (>1.0) orani  : %{100*(dz>1.0).mean():.1f}")
    print(f"\n2) KAHIN YAYILIM TAVANI (bilerek iyimser)")
    print(f"   toplam FN {fn_top} | yayilimla erisilebilen {fn_kurtarilan} "
          f"= **%{100*fn_kurtarilan/max(fn_top,1):.1f}**")
    with open(CIKTI, "w") as f:
        json.dump({"parca": parca, "duzenlilik_medyan": float(np.median(dz)),
                   "duzenli_oran": float((dz < 0.25).mean()),
                   "fn_toplam": fn_top, "fn_kahin_kurtarilan": fn_kurtarilan,
                   "kahin_oran": fn_kurtarilan / max(fn_top, 1),
                   "not": "KAHIN tavani -- gercek kol bundan iyi OLAMAZ. "
                          "Kesinlik bedeli DAHIL DEGIL."}, f, indent=1)
    print(f"\nmakbuz -> {CIKTI}")


if __name__ == "__main__":
    main()
