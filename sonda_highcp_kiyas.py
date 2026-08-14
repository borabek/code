# -*- coding: utf-8 -*-
"""YOGUN PARCADA IKI URUN YOLU YAN YANA: `extract` vs `extract_highcp`.

SEBEP (2026-08-14, gozle ara test): yogun klemenslerde urun agir kaciriyor
(orn. GT=22 -> 11 CP, GT=20'de robot-ISARETLI yalniz 4). Ama kodda ZATEN
yogun parcaya ozel bir urun yolu var (`robot_cp.extract_highcp`,
makbuz `results/product_f1_receipt.json`, OOF F1 0.807 part-out) ve
**GLB ihracatcisi onu HIC CAGIRMIYOR** -- `export_robot_glb.py:155` her
zaman `robot_cp.extract` diyor.

DURUSTLUK -- BU ADIL BIR KIYAS DEGIL, OYLE DE SUNULMAYACAK:
`extract_highcp` **cp_count** ister (ureticinin CP sayisi) ve bu bir
METADATA YARDIMIdir. Burada cp_count GT'den veriliyor. Yani kiyas
"yardimsiz yol" vs "yardimli yol"dur. Sahada cp_count ancak parca
kunyesinden/kutup sayisindan biliniyorsa gecerlidir.
Bu yuzden ucuncu bir sutun daha var: **cp_count YANLIS verilirse**
(GT +/- 2) ne oluyor -- yardimin kirilganligi olculsun.
"""
import io
import json
import os
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kanonik_d7 as K                                  # noqa: E402
import d6_kayit                                         # noqa: E402
from sina_kume import esle_macar                         # noqa: E402

OLCUTLER = (("tespit", 0.0, 180.0, True, False),
            ("rob", 2.0, 10.0, False, False),
            ("rbi", 2.0, 10.0, False, True))


def _birim(v):
    v = np.asarray(v, float).reshape(-1, 3)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def puanla(cps, G, Gd, dg):
    P = (np.asarray([c["point"] for c in cps], float).reshape(-1, 3)
         if cps else np.zeros((0, 3)))
    D = (_birim([c["direction"] for c in cps]) if cps
         else np.zeros((0, 3)))
    out = {}
    for ad, tol, am, pct, isr in OLCUTLER:
        tp, fp, fn, _ = esle_macar(P, D, G, Gd, dg, tol, am, pct,
                                   isaretli=isr)
        out[ad] = (tp, fp, fn)
    return out, len(cps)


def main():
    import torch
    import robot_cp
    from infer_step_cp import load_any

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.load(io.open("cp_config.json", encoding="utf-8"))
    ca = float(cfg.get("robot_conf_auto", 0.5))
    mav = int(cfg.get("robot_min_auto_votes", 3))
    hc = cfg["robot_highcp"]

    m4 = [load_any(c, dev=dev)[:2] for c in cfg["robot_vote2_checkpoints"]]
    m7 = m4 + [load_any(c, dev=dev)[:2] for c in hc["extra_checkpoints"]]
    print(f"cihaz {dev} | taban yol {len(m4)} model | "
          f"yogun yol {len(m7)} model x 2 cozunurluk\n", flush=True)

    kay = K.yukle()
    kay.update({str(p): r for p, r in d6_kayit.yukle().items()
                if str(p) not in kay})
    STEP = K.step_haritasi()
    pids = [x.strip() for x in sys.argv[1:] if x.strip()]

    top = {k: {a: [0, 0, 0] for a, *_ in OLCUTLER}
           for k in ("taban", "yogun", "yogun_yanlis")}
    for pid in pids:
        if pid not in STEP or pid not in kay:
            print(f"  {pid}: STEP/GT yok -- ATLANDI")
            continue
        r = kay[pid]
        G = np.asarray(r["G"], float).reshape(-1, 3)
        Gd = _birim(r["Gd"])
        dg = float(r["diag"])
        n_gt = len(G)

        a, na = puanla(robot_cp.extract(m4, STEP[pid], dev, ca, mav),
                       G, Gd, dg)
        b, nb = puanla(robot_cp.extract_highcp(m7, STEP[pid], dev, ca,
                                               mav, n_gt), G, Gd, dg)
        # cp_count YANLIS verilirse (2 eksik) -- yardimin kirilganligi
        yanlis = max(1, n_gt - 2)
        c, nc = puanla(robot_cp.extract_highcp(m7, STEP[pid], dev, ca,
                                               mav, yanlis), G, Gd, dg)
        for ad, v in (("taban", a), ("yogun", b), ("yogun_yanlis", c)):
            for k in v:
                for i in range(3):
                    top[ad][k][i] += v[k][i]
        print(f"  {pid:12s} GT={n_gt:3d} | TABAN cp={na:3d} "
              f"ISARETLI={a['rbi'][0]:3d} | YOGUN cp={nb:3d} "
              f"ISARETLI={b['rbi'][0]:3d} | YOGUN(cp_count-2) cp={nc:3d} "
              f"ISARETLI={c['rbi'][0]:3d}", flush=True)

    def f1(t):
        return 2 * t[0] / max(2 * t[0] + t[1] + t[2], 1)

    print(f"\n{'yol':22s} {'tespit':>8s} {'robot':>8s} {'robot-ISR':>10s}")
    for ad, isim in (("taban", "TABAN (extract)"),
                     ("yogun", "YOGUN (cp_count=GT)"),
                     ("yogun_yanlis", "YOGUN (cp_count-2)")):
        v = top[ad]
        print(f"{isim:22s} {f1(v['tespit']):8.4f} {f1(v['rob']):8.4f} "
              f"{f1(v['rbi']):10.4f}")
    print("\nUYARI: YOGUN yol cp_count ISTER -> METADATA YARDIMLI."
          "\nTABAN yol yardimsizdir. Kiyas ADIL DEGILDIR ve oyle sunulmaz.")
    json.dump({k: {a: f1(v) for a, v in d.items()} for k, d in top.items()},
              io.open("results/highcp_kiyas.json", "w", encoding="utf-8"),
              indent=1)
    print("-> results/highcp_kiyas.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
