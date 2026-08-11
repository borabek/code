# -*- coding: utf-8 -*-
"""D6 URUN YOLU OLCUMU -- `sonda_dagitim_dogrula.py`'nin D6 ikizi.

NEDEN AYRI BIR BETIK: kapi kararlari D6'da veriliyor ama D6 olcumlerim
ONBELLEKTEN (`_tam_oz`) kosuyordu. Onbellek `cp_config.json`'un eski halinde
turetilmis ve segmentasyon adaylari kaymis (`results/p6_parite_d6.json`).
Onbellekten olculen sayi URUNUN sayisi degildir; kapi bu yuzden urunun CANLI
yolundan gecmeli.

Bu betik `kanonik_zincir.urun_cikti`'yi cagirir -- yani URUNUN TEK zincirini --
ve D6'da olcer. Kollar cevre degiskenleriyle secilir, tek degiskenli kiyas icin:
    URUN_P6=0 URUN_GENIS=1   dagitilan taban
    URUN_P6=1                P6 ortak siralayici
    DOG_POZ=0                poz kafasi KAPALI (P6 yonu kendi secer)
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

OB = "results/_p1_olasilik"
N = int(os.environ.get("DOG_N", "0"))
POZ = os.environ.get("DOG_POZ", "1") != "0"


def main():
    import d6_kayit
    import kanonik_d7 as K
    import kanonik_zincir
    import robot_cp
    import urun_genis
    import urun_p6
    import urun_zinciri
    from sina_kume import esle_macar

    S = K.step_haritasi()
    pidler = sorted(f[:-4] for f in os.listdir(OB) if f.endswith(".npz"))
    kay = {str(p): r for p, r in d6_kayit.yukle(pidler).items()}
    secili = [p for p in pidler if p in kay and len(kay[p].get("G", []))
              and S.get(p)]
    if N:
        secili = secili[:N]
    sh = os.environ.get("DOG_SHARD")     # `birlestir_makbuz.py` ile birlesir
    if sh:
        i_, n_ = (int(x) for x in sh.split("/"))
        secili = [p for k, p in enumerate(secili) if k % n_ == i_]
        print(f"PAY {i_}/{n_}", flush=True)
    print(f"D6 {len(secili)} parca | P6 {'ACIK' if urun_p6.ACIK else 'KAPALI'}"
          f" | genis {'ACIK' if urun_genis.ACIK else 'KAPALI'}"
          f" | poz kafasi {'ACIK' if POZ else 'KAPALI'}", flush=True)

    rob = collections.defaultdict(lambda: [0, 0, 0])
    tes = []
    kirilim = {}
    cfg = robot_cp._load_cfg()
    for i, pid in enumerate(secili, 1):
        r = kay[pid]
        z = np.load(f"{OB}/{pid}.npz")
        V = np.ascontiguousarray(z["V"], np.float64)
        F = np.ascontiguousarray(z["F"], np.int64)
        pbs = [np.asarray(q, float) for q in z["pbs"]]
        cps = kanonik_zincir.urun_cikti(V, F, pbs, S.get(pid), cfg=cfg)
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
        kirilim[pid] = {"mfg": r["mfg"], "rob": [tp, fp, fn],
                        "tes": [int(x) for x in t_]}
        if i % 50 == 0:
            print(f"  {i}/{len(secili)}", flush=True)

    pm = {m: 2 * v[0] / max(2 * v[0] + v[1] + v[2], 1) for m, v in rob.items()}
    mi = float(2 * sum(v[0] for v in rob.values()) /
               max(sum(2 * v[0] + v[1] + v[2] for v in rob.values()), 1))
    out = {"robot": mi, "tespit": K.mikro(tes),
           "makro": float(np.mean(list(pm.values()))),
           "en_kotu": float(min(pm.values())), "marka": pm,
           "TP": sum(v[0] for v in rob.values()),
           "FP": sum(v[1] for v in rob.values()),
           "FN": sum(v[2] for v in rob.values()),
           "n_parca": len(secili), "p6_acik": bool(urun_p6.ACIK),
           "genis_acik": bool(urun_genis.ACIK), "poz_kafasi": bool(POZ),
           "parca_kirilim": kirilim}
    print(f"\nD6 URUN ZINCIRI robot {mi:.4f} | tespit {out['tespit']:.4f} | "
          f"makro {out['makro']:.4f} | TP {out['TP']} FP {out['FP']} "
          f"FN {out['FN']}")
    yol = os.environ.get("DOG_CIKTI", "results/d6_urun.json")
    json.dump({"damga": makbuz_hash.damga(), "sonuc": out,
               "not": "URUNUN TEK kanonik zinciri, D6 (gorulmemis marka: SUPU/"
                      "UPUN/MOR/NIT/UTL/S+S/SE/ONV). MIKRO."},
              open(yol, "w"), indent=1)
    print(f"makbuz -> {yol}")


if __name__ == "__main__":
    main()
