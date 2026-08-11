# -*- coding: utf-8 -*-
"""P6 PARITE TESTI: egitimdeki havuz = urundeki havuz mu?

BULGU (2026-08-11, ilk kosu): AYRISIYOR ve sebebi benim eklemem DEGIL.
`_tam_oz` onbellegi `cp_config.json`'un ESKI halinde turetilmis; segmentasyon
adaylari o gunden beri kaymis (ornek 016029: onbellek 7, canli 12). B-rep tarafi
BIREBIR ayni (silindir 50/50, aciklik 0/0) -- yani fark tamamen segmentasyon
kolunda. Dagitilan `kazanan_hgb_derin` da AYNI onbellekten egitildi, yani bu
onceden var olan bir durum ve kiyas adil kalir.

SONUC: onbellekten olculen sayilar URUNUN sayisi DEGILDIR. Kapi olcumleri
(`sonda_d6_urun.py`, `sonda_dagitim_dogrula.py`) URUNUN CANLI YOLUNDAN gecer.
Bu betik artik bir PARITE KAPISI degil, bir AYRISMA OLCERIDIR.


Egitim `kos_p6_oznitelik.py` ile ONBELLEKTEN (silindir pkl, `_tam_oz` havuzu)
uretiliyor; urun (`urun_p6.secenek_tablosu`) ayni seyi STEP'ten YENIDEN
hesapliyor. Ikisi ayrisirsa model egitildigi uzaydan farkli bir uzayda skor
uretir ve bu SESSIZ olur -- bu projede iki kez oldu.

Bu betik birkac parcada iki yolu yan yana kosar ve
  * havuz boyutu / konumlari
  * secenek sayisi
  * 92 sutunun MAKSIMUM MUTLAK FARKI
raporlar. Fark buyukse D7 OKUMASI YAPILMAZ.

Kullanim:  python sonda_p6_parite.py [kume] [n]
"""
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

KUME = {"d6": ("results/_p1_olasilik", "all_wscad_stp"),
        "d7": ("results/_p1_olasilik_d7", "all_wscad_stp")}


def main():
    on = sys.argv[1] if len(sys.argv) > 1 else "d6"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    ob, _ = KUME[on]

    import kanonik_d7 as K
    import robot_cp
    import urun_p6
    S = K.step_haritasi()

    fs = sorted(f for f in os.listdir("results/_p6_oz")
                if f.startswith(on + "_") and f.endswith(".npz"))
    if not fs:
        sys.exit(f"onbellek yok: results/_p6_oz/{on}_*.npz")
    rapor = []
    bak = 0
    for f in fs:
        if bak >= n:
            break
        pid = f[len(on) + 1:-4]
        step = S.get(pid)
        mf = f"{ob}/{pid}.npz"
        if not step or not os.path.exists(mf):
            continue
        z = np.load(mf)
        V = np.ascontiguousarray(z["V"], np.float64)
        Fc = np.ascontiguousarray(z["F"], np.int64)
        pbs = [np.asarray(q, float) for q in z["pbs"]]
        cfg = robot_cp._load_cfg()
        cps, avg, _hi, _u = robot_cp.adaylari_uret(V, Fc, pbs, step, cfg=cfg)
        if not cps:
            continue
        tab = urun_p6.secenek_tablosu(V, Fc, avg, cps, step,
                                      robot_cp.CE, robot_cp.CT)
        if tab is None:
            rapor.append({"pid": pid, "durum": "URUN YOLU None"})
            bak += 1
            continue
        Pu, idxu, YDu, Xu = tab
        c = np.load(f"results/_p6_oz/{f}")
        Xc = np.asarray(c["X"], float)
        Pc = np.asarray(c["P"], float)
        d = {"pid": pid, "urun_aday": len(Pu), "onbellek_aday": len(Pc),
             "urun_secenek": len(Xu), "onbellek_secenek": len(Xc)}
        if len(Pu) == len(Pc):
            d["konum_maks_fark_mm"] = float(np.abs(Pu - Pc).max())
        if Xu.shape == Xc.shape:
            fark = np.abs(Xu - Xc)
            d["oz_maks_fark"] = float(fark.max())
            d["oz_maks_fark_sutun"] = int(fark.max(0).argmax())
            d["oz_ort_fark"] = float(fark.mean())
        rapor.append(d)
        print(json.dumps(d, ensure_ascii=False), flush=True)
        bak += 1

    ayni = [r for r in rapor if r.get("oz_maks_fark") is not None
            and r["oz_maks_fark"] < 1e-6]
    print(f"\n{len(ayni)}/{len(rapor)} parcada oznitelikler BIREBIR ayni")
    if len(ayni) != len(rapor):
        print("!! AYRISMA VAR -- D7 OKUMASI YAPILMAZ, once sebebi bulunur.")
    json.dump({"damga": makbuz_hash.damga(), "kume": on, "rapor": rapor,
               "birebir": len(ayni), "bakilan": len(rapor),
               "not": "Egitim onbellegi ile urun yolunun oznitelik paritesi."},
              open(f"results/p6_parite_{on}.json", "w"), indent=1)
    print(f"makbuz -> results/p6_parite_{on}.json")


if __name__ == "__main__":
    main()
