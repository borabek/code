# -*- coding: utf-8 -*-
"""DAGITIM DOGRULAMASI: baglanan yol URUNUN ZINCIRINDEN gecince ne veriyor?

Kod yazip olcmeden "dagitildi" denmez. Bu betik `kanonik_zincir.urun_cikti`'yi
-- yani urunun TEK kanonik zincirini -- cagirir, uzerine poz kafasini uygular
ve D7'de olcer. Beklenen: robot ~0.3090 (ayri betiklerde olculen deger).

Fark cikarsa entegrasyonda bir sey ayrisiyor demektir ve DAGITIM YAPILMAZ.

Iki kol: `URUN_GENIS=0` (eski yol, beklenen ~0.2029) ve `URUN_GENIS=1`.
"""
import collections
import json
import os
import sys

import numpy as np

import makbuz_hash

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, ".")

OB = "results/_p1_olasilik_d7"
N = int(os.environ.get("DOG_N", "0"))     # 0 = tum D7
# POZ KAFASI: dagitilan yolda ACIK (varsayilan). P6 kolu yonu KENDISI sectigi
# icin poz kafasinin uzerine yazip yazmadigi AYRI olculmelidir -> DOG_POZ=0.
POZ = os.environ.get("DOG_POZ", "1") != "0"


def main():
    import kanonik_d7 as K
    import kanonik_zincir
    import urun_genis
    import urun_zinciri
    from sina_kume import esle_macar

    S = K.step_haritasi()
    pids = sorted(f[:-4] for f in os.listdir(OB) if f.endswith(".npz"))
    kay = K.yukle(pids)
    secili = [p for p in pids if p in kay and len(kay[p].get("G", []))]
    if N:
        secili = secili[:N]
    # PAYLI KOSU: `DOG_SHARD=i/n`. P6 kolu parca basina saniyeler suruyor
    # (B-rep + ~2000 secenek icin isin atisi); 835 parca tek islemde saatler
    # alir. Paylar `birlestir_makbuz.py` ile birlestirilir; mikro F1 parca
    # basina TP/FP/FN toplami oldugu icin birlestirme KAYIPSIZDIR.
    sh = os.environ.get("DOG_SHARD")
    if sh:
        i_, n_ = (int(x) for x in sh.split("/"))
        secili = [p for k, p in enumerate(secili) if k % n_ == i_]
        print(f"PAY {i_}/{n_}", flush=True)
    print(f"D7 {len(secili)} parca | genis kol {'ACIK' if urun_genis.ACIK else 'KAPALI'}",
          flush=True)
    rob = collections.defaultdict(lambda: [0, 0, 0])
    tes = []
    kolsuz = 0
    # PARCA BAZINDA kirilim: bootstrap guven araligi olcumu YENIDEN KOSMADAN
    # cikarilabilsin diye. Manset mikro F1'dir; CI ayni sayidan uretilir.
    parca_kirilim = {}
    for i, pid in enumerate(secili, 1):
        r = kay[pid]
        z = np.load(f"{OB}/{pid}.npz")
        V = np.ascontiguousarray(z["V"], np.float64)
        F = np.ascontiguousarray(z["F"], np.int64)
        pbs = [np.asarray(q, float) for q in z["pbs"]]
        cps = kanonik_zincir.urun_cikti(V, F, pbs, S.get(pid))
        P, D = kanonik_zincir.poz_ver(cps)
        if len(P) and POZ:
            P, D = urun_zinciri.tam_poz(V, F, np.mean(pbs, axis=0), P, D,
                                        step_path=S.get(pid))
        G = np.asarray(r["G"], float)
        Gd = np.asarray(r["Gd"], float)
        dg = float(r["diag"])
        tp, fp, fn = esle_macar(P, D, G, Gd, dg, K.YANAL, K.ACI, False,
                                isaretli=True)[:3]
        a = rob[r["mfg"]]
        a[0] += tp; a[1] += fp; a[2] += fn
        t_ = esle_macar(P, D, G, Gd, dg, max(3.0, 0.06 * dg), 180.0, True)[:3]
        tes.append((len(G),) + t_)
        parca_kirilim[pid] = {"mfg": r["mfg"], "rob": [tp, fp, fn],
                              "tes": [int(x) for x in t_]}
        if i % 100 == 0:
            print(f"  {i}/{len(secili)}", flush=True)
    pm = {m: 2 * v[0] / max(2 * v[0] + v[1] + v[2], 1) for m, v in rob.items()}
    mi = float(2 * sum(v[0] for v in rob.values()) /
               max(sum(2 * v[0] + v[1] + v[2] for v in rob.values()), 1))
    import urun_p6
    out = {"robot": mi, "tespit": K.mikro(tes),
           "makro": float(np.mean(list(pm.values()))),
           "en_kotu": float(min(pm.values())), "marka": pm,
           "n_parca": len(secili), "genis_acik": bool(urun_genis.ACIK),
           "p6_acik": bool(urun_p6.ACIK), "poz_kafasi": bool(POZ),
           "p6_sayac": dict(urun_p6.SAYAC),
           "parca_tp_fp_fn": {p: v for p, v in parca_kirilim.items()}}
    print(f"\nURUN ZINCIRI robot {mi:.4f} | tespit {out['tespit']:.4f} | "
          f"makro {out['makro']:.4f} | en kotu {out['en_kotu']:.4f}")
    print(f"BEKLENEN: genis ACIK ~0.3090 | genis KAPALI ~0.2029")
    json.dump({"damga": makbuz_hash.damga(), "sonuc": out,
               "not": "Urunun TEK kanonik zinciri (`kanonik_zincir.urun_cikti`) "
                      "+ poz kafasi. D7 marka-disi, MIKRO."},
              open(os.environ.get("DOG_CIKTI",
                                  "results/dagitim_dogrula.json"), "w"), indent=1)
    print("makbuz yazildi")


if __name__ == "__main__":
    main()
