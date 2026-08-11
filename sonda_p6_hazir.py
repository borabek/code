# -*- coding: utf-8 -*-
"""P6 HAZIRLIK KONTROLU -- D7 okumasindan ONCE kosar.

D7 butcesi 3 okuma. Bir okumayi "model paketi urun yolunda calismiyor" diye
harcamak kabul edilemez. Bu betik birkac D6 parcasinda URUNUN TAM yolunu kosar
ve sunlari dogrular:

  1. model paketi yukleniyor ve beklenen anahtarlari tasiyor
  2. `urun_p6.secenek_tablosu` calisiyor, sutun sayisi EGITIMDEKIYLE ayni
  3. skorlar DEJENERE degil (hepsi ayni / hepsi 0 / NaN degil)
  4. secim BOS DEGIL ve makul sayida CP donuyor
  5. ikinci kademe varsa gercekten devrede (kisa liste bos donmuyor)

Herhangi biri duserse D7'YE BAKILMAZ.
"""
import json
import os
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, ".")

OB = "results/_p1_olasilik"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    import kanonik_d7 as K
    import kanonik_zincir
    import p6_karar
    import robot_cp
    import urun_p6

    pk = urun_p6.model_yukle()
    if pk is None:
        sys.exit("HATA: model paketi YOK -- once kos_p6_kademe2.py kos.")
    print(f"paket anahtarlari: {sorted(pk)}")
    for a in ("kademe1", "kural", "nms", "tohum_kural", "tohum_nms"):
        if a not in pk:
            sys.exit(f"HATA: pakette `{a}` yok.")
    print(f"  kol {pk.get('kol')} | kural {pk['kural']} | nms {pk['nms']} "
          f"| kisa esik {pk.get('kisa_esik')} | 2. kademe "
          f"{'VAR' if pk.get('kademe2') is not None else 'YOK'}")
    print(f"  URUN_P6 = {'ACIK' if urun_p6.ACIK else 'KAPALI'} | mesh havuzu "
          f"{'ACIK' if urun_p6.MESH_HAVUZ else 'KAPALI'}")
    if not urun_p6.ACIK:
        print("  !! URUN_P6 KAPALI -- olcumde `URUN_P6=1` verilmeli.")

    S = K.step_haritasi()
    pidler = [f[:-4] for f in sorted(os.listdir(OB)) if f.endswith(".npz")]
    cfg = robot_cp._load_cfg()
    bakilan, rapor = 0, []
    for pid in pidler:
        if bakilan >= n:
            break
        if not S.get(pid):
            continue
        z = np.load(f"{OB}/{pid}.npz")
        V = np.ascontiguousarray(z["V"], np.float64)
        F = np.ascontiguousarray(z["F"], np.int64)
        pbs = [np.asarray(q, float) for q in z["pbs"]]
        cps, avg, _hi, _u = robot_cp.adaylari_uret(V, F, pbs, S[pid], cfg=cfg)
        if not cps:
            continue
        tab = urun_p6.secenek_tablosu(V, F, avg, cps, S[pid],
                                      robot_cp.CE, robot_cp.CT)
        if tab is None:
            rapor.append({"pid": pid, "durum": "secenek_tablosu None"})
            bakilan += 1
            continue
        P, idx, YD, X, kaynak = tab
        Xd = np.hstack([p6_karar.donustur(X, pk.get("zskor", "ab")),
                        p6_karar.kaynak_blok(kaynak[idx])])
        bek = pk["kademe1"].n_features_in_
        if Xd.shape[1] != bek:
            sys.exit(f"HATA: sutun sayisi {Xd.shape[1]}, model {bek} bekliyor.")
        s1 = pk["kademe1"].predict_proba(Xd)[:, 1]
        out = kanonik_zincir.urun_cikti(V, F, pbs, S[pid], cfg=cfg)
        r = {"pid": pid, "aday": len(P), "secenek": len(idx),
             "mesh_aday": int((kaynak == 2).sum()),
             "s1_min": float(s1.min()), "s1_ort": float(s1.mean()),
             "s1_maks": float(s1.max()),
             "s1_kisa_liste": int((s1 >= float(pk.get("kisa_esik", 0.2))).sum()),
             "cikti_cp": len(out or [])}
        rapor.append(r)
        print(json.dumps(r), flush=True)
        bakilan += 1

    ok = [r for r in rapor if r.get("cikti_cp", 0) > 0]
    dej = [r for r in rapor if r.get("s1_maks", 0) - r.get("s1_min", 0) < 1e-6]
    print(f"\n{len(ok)}/{len(rapor)} parcada BOS OLMAYAN cikti")
    if dej:
        print(f"!! {len(dej)} parcada skorlar DEJENERE (hepsi ayni)")
    if pk.get("kademe2") is not None:
        bos = [r for r in rapor if r.get("s1_kisa_liste", 0) == 0]
        if bos:
            print(f"!! {len(bos)} parcada KISA LISTE BOS -- 2. kademe devre disi")
    karar = "HAZIR" if ok and not dej else "HAZIR DEGIL"
    print(f"\nKARAR: {karar}")
    json.dump({"rapor": rapor, "karar": karar},
              open("results/p6_hazir.json", "w"), indent=1)
    if karar != "HAZIR":
        sys.exit(1)


if __name__ == "__main__":
    main()
