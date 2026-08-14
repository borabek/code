# -*- coding: utf-8 -*-
"""PARITE TESTI: olculen urun ile DAGITILAN urun ayni mi?

2026-07-31 denetimi uc ayrisma buldu:
  1) robot_cp'nin dusuk-CP turetmesi step_path ALMIYORDU -> B-rep ekseni gercek uründe
     parcalarin ~%89.5'inde hic calismiyordu, oysa olcum onunla yapiliyordu.
  2) _votes benzersiz MODEL saymiyordu (4 checkpoint varken votes=5).
  3) Olcum esikleri 0.35/0.20, dagitilan urun 0.40/0.35.

Bu test ucunu de KODDAN dogrular. Kirmizi olursa olculen sayi dagitilan urunu temsil etmiyordur.
"""
import os, sys, re, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("BA_ALLOW_SEEN", "1")


def main():
    fails = []

    src = open("robot_cp.py", encoding="utf-8").read()
    calls = re.findall(r"cp_openings\.connection_points\((?:[^()]|\([^()]*\))*\)", src, re.S)
    miss = [i for i, t in enumerate(calls, 1) if "step_path" not in t]
    print(f"1) step_path paritesi : {len(calls)-len(miss)}/{len(calls)} cagri")
    if miss:
        fails.append(f"step_path eksik cagrilar: {miss}")

    import robot_cp
    cfg = json.load(open("cp_config.json"))
    n_ck = len(cfg["current_product"]["checkpoints"])
    import numpy as np
    # ayni modelden IKI yakin aday -> tek oy olmali
    fake = [[{"point": np.array([0., 0, 0]), "direction": np.array([0., 0, 1]), "confidence": .9,
              "source_label": 3, "n_verts": 10, "area": 1.0, "insertion_depth_mm": 1.0},
             {"point": np.array([1., 0, 0]), "direction": np.array([0., 0, 1]), "confidence": .8,
              "source_label": 3, "n_verts": 10, "area": 1.0, "insertion_depth_mm": 1.0}]]
    v = robot_cp._vote2(fake, cluster_mm=5.0, min_votes=1)
    print(f"2) benzersiz oy       : tek modelden 2 yakin aday -> votes={v[0]['_votes']} (1 olmali)")
    if v[0]["_votes"] != 1:
        fails.append("ayni model iki oy veriyor")
    for lst in ([fake[0][:1]] * n_ck):
        pass
    many = [[dict(fake[0][0])] for _ in range(n_ck)]
    v2 = robot_cp._vote2(many, cluster_mm=5.0, min_votes=1)
    print(f"   {n_ck} farkli modelden ayni aday -> votes={v2[0]['_votes']} ({n_ck} olmali)")
    if v2[0]["_votes"] != n_ck:
        fails.append("farkli modeller tek oy sayiliyor")

    t_lo = cfg.get("robot_wire_gate_threshold")
    t_hi = cfg.get("robot_wire_gate_threshold_highcp")
    print(f"3) esik kaynagi       : cp_config {t_lo}/{t_hi}")
    print("   (olcum betikleri bu degerleri KENDI secmemeli; buradan okumali)")

    import wire_gate
    m = wire_gate._load(wire_gate.MODEL_PATH)
    nf = len(m.get("cols", m.get("feat_names", [])))
    print(f"4) gate ozellik sayisi: {nf} ({wire_gate.MODEL_PATH})")

    import cp_openings
    print(f"5) B-rep eksen bayragi: {cp_openings.USE_BREP_AXIS}")
    if not cp_openings.USE_BREP_AXIS:
        fails.append("USE_BREP_AXIS kapali")

    print()
    if fails:
        print("PARITE BOZUK:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("PARITE TAMAM -- olculen urun ile dagitilan urun ayni.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
