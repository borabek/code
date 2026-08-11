# -*- coding: utf-8 -*-
"""D6'da EKSIK HUCRE: gate + MEVCUT(`v_o`), MIKRO, marka basina.

p5-v2 kaydinda taban "secim YOK, gate YOK, hep `v_o`" idi (0.0823 mikro) ve
p5v2+gate 0.1751 olarak +0.0928 gosteriyordu. Ama D7'de ayni kiyas gate'li
tabana karsi +0.0021 verdi. Fark tabanin gate'siz olmasindan mi geliyor?
Bu betik o hucreyi doldurur: EGITIM GEREKMEZ, yalniz gate maskesi + `v_o`.
"""
import collections, json, os, pickle, sys
import numpy as np
import makbuz_hash
os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"]="1"; os.environ["WG_TOPO"]="1"; os.environ["WG_ZENGIN"]="1"
sys.path.insert(0, ".")
import d6_kayit, wire_gate, kanonik_d7 as K
from p1c_esik import maske
from sina_kume import esle_macar

gate = K.gate_yukle()
d6 = d6_kayit.sinav(); k6 = d6_kayit.yukle(set(d6["pidler"]))
per = collections.defaultdict(list)
for pid, r in k6.items():
    X = d6_kayit.x58(r); G = np.asarray(r.get("G", []), float)
    if X is None or not len(G):
        continue
    P = np.asarray(r["P"], float); D = np.asarray(r["Pd"], float)
    gs = np.asarray(wire_gate.karar_skoru(gate, X), float)
    k = maske(gs, 0.40, 0.30)
    Ps, Ds = (P[k], D[k]) if k.any() else (P[:0], D[:0])
    tp, fp, fn = esle_macar(Ps, Ds, G, np.asarray(r["Gd"], float), r["diag"],
                            K.YANAL, K.ACI, False, isaretli=True)[:3]
    per[r["mfg"]].append((len(G), tp, fp, fn))

P5 = {"SUPU": 0.2903, "UPUN": 0.3215, "MOR": 0.1201, "NIT": 0.0652, "UTL": 0.0784}
ham = {"SUPU": 0.1285, "UPUN": 0.1246, "MOR": 0.0607, "NIT": 0.0610, "UTL": 0.0369}
print(f"{'marka':<6} {'n':>4} {'ham v_o':>9} {'gate+v_o':>9} {'p5v2+gate':>10} {'fark':>8}")
sat = {}
for m in P5:
    rows = per.get(m, [])
    g = K.mikro(rows) if rows else float("nan")
    sat[m] = g
    print(f"{m:<6} {len(rows):>4} {ham[m]:>9.4f} {g:>9.4f} {P5[m]:>10.4f} "
          f"{P5[m]-g:>+8.4f}")
gv = float(np.mean(list(sat.values()))); pv = float(np.mean(list(P5.values())))
print(f"\nORT   ham {np.mean(list(ham.values())):.4f} | gate+v_o {gv:.4f} | "
      f"p5v2+gate {pv:.4f} | FARK {pv-gv:+.4f}")
json.dump({"damga": makbuz_hash.damga(), "gate_vo": sat, "p5v2_gate": P5,
           "ham_vo": ham, "ort": {"gate_vo": gv, "p5v2_gate": pv, "fark": pv-gv},
           "not": "MIKRO. Egitim yok; gate maskesi + v_o. p5v2 sayilari LOMO."},
          open("results/d6_eksik_hucre.json", "w"), indent=1)
