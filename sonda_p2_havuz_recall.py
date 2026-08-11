# -*- coding: utf-8 -*-
"""P2 KARAR KAPISI: yeni ag D7'de HAVUZ RECALL'u artiriyor mu?

KURAL (bu oturumda konuldu): yeni bir segmentasyon agi once HAVUZ RECALL ile
sinanir, F1 ile DEGIL. Gerekce: gate'siz F1 fazla adaylari FP sayar ve iyi bir
havuzu kotu gosterir; ayrica gate eski dagilimda egitildigi icin uctan uca dusus
agin degil gate'in uyumsuzlugunun isareti olur.

TABAN (`results/havuz_recall_d7.json`): kanonik G7 havuz recall **0.6654**.
Yeni ag bunu GECMEZSE kol KAPANIR ve uctan uca olcume SOKULMAZ.

Aday uretimi urunun kendi fonksiyonuyla (`robot_cp.adaylari_uret`), tespit
toleransi, aci serbest, bire-bir Macar.
"""
import collections
import json
import os
import sys

import numpy as np

import makbuz_hash

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, ".")
import kanonik_d7 as K          # noqa: E402
import robot_cp                 # noqa: E402
from sina_kume import esle_macar  # noqa: E402


def olc(ob, kay, S):
    TP = FN = nA = 0
    per = collections.defaultdict(lambda: [0, 0])
    yok = hata = 0
    for pid, r in sorted(kay.items()):
        G = np.asarray(r.get("G", []), float)
        f = f"{ob}/{pid}.npz"
        if not len(G):
            continue
        if not os.path.exists(f):
            yok += 1
            continue
        z = np.load(f)
        V = np.ascontiguousarray(z["V"], np.float64)
        F = np.ascontiguousarray(z["F"], np.int64)
        pbs = [np.asarray(q, float) for q in z["pbs"]]
        # CIPLAK except YOK: olcemeyen betik KARAR URETMEZ, patlar.
        cps, _op, _cok, _per = robot_cp.adaylari_uret(V, F, pbs, S.get(pid))
        P = np.asarray([c["point"] for c in cps], float) if cps else np.zeros((0, 3))
        D = np.asarray([c["direction"] for c in cps], float) if cps \
            else np.zeros((0, 3))
        tol = max(3.0, 0.06 * float(r["diag"]))
        tp, _fp, fn = esle_macar(P, D, G, np.asarray(r["Gd"], float),
                                 float(r["diag"]), tol, 180.0, True)[:3]
        TP += tp
        FN += fn
        nA += len(P)
        a = per[r["mfg"]]
        a[0] += tp
        a[1] += fn
    return {"recall": TP / max(TP + FN, 1), "aday": nA,
            "aday_per_parca": nA / max(len(kay), 1),
            "marka": {m: a[0] / max(a[0] + a[1], 1) for m, a in per.items()},
            "onbellegi_olmayan": yok}


def main():
    S = K.step_haritasi()
    kay = K.yukle(json.load(open("results/d7_sinav_kumesi.json"))["pidler"])
    kollar = {
        "G7 (kanonik TABAN)": "results/_p1_olasilik_d7",
        "P2-brep (YENI)": "results/_p1_olasilik_p2brep",
    }
    out = {}
    for ad, ob in kollar.items():
        if not os.path.isdir(ob):
            raise SystemExit(f"{ob} YOK -- olcum yapilamaz")
        out[ad] = olc(ob, kay, S)
        c = out[ad]
        print(f"{ad:<22} havuz recall {c['recall']:.4f} | aday/parca "
              f"{c['aday_per_parca']:.1f} | onbellegi yok {c['onbellegi_olmayan']}",
              flush=True)
    a = out["G7 (kanonik TABAN)"]
    b = out["P2-brep (YENI)"]
    d = b["recall"] - a["recall"]
    print(f"\nFARK {d:+.4f}")
    art = sum(1 for m in b["marka"] if b["marka"][m] > a["marka"].get(m, 0) + 1e-9)
    print(f"artan marka {art}/{len(b['marka'])}")
    print(f"\n{'marka':<8} {'G7':>8} {'P2':>8} {'fark':>8}")
    for m in sorted(a["marka"], key=lambda k: a["marka"][k]):
        print(f"  {m:<7} {a['marka'][m]:>7.4f} {b['marka'].get(m, 0):>8.4f} "
              f"{b['marka'].get(m, 0)-a['marka'][m]:>+8.4f}")
    print("\nKARAR: " + ("GECTI -- uctan uca olcume gecilir"
                         if d > 0 else "KALDI -- kol KAPANIR"))
    json.dump({"damga": makbuz_hash.damga(), "sonuc": out, "fark": d,
               "artan_marka": art, "gecti": bool(d > 0),
               "not": "HAVUZ RECALL karar kapisi. F1 DEGIL. D7 marka-disi."},
              open("results/p2_havuz_recall.json", "w"), indent=1)
    print("makbuz -> results/p2_havuz_recall.json")


if __name__ == "__main__":
    main()
