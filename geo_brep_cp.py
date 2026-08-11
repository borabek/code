# -*- coding: utf-8 -*-
"""GEO-BREP: saf geometrik CP dedektoru -- EGITIM YOK, SEGMENTASYON YOK, ML YOK.

NEDEN DEGERLI (2026-07-29 gecesinin en aci bulgusu):
  ML kolu URETICI-DISI bolunmede 0.35'e dusuyor -- yani gorulmemis bir ureticide cokuyor.
  Saf geometrik bir dedektor egitim verisi gerektirmedigi icin bu sorundan MUAF: yeni bir uretici
  icin sifir veriyle calisir. Mevcut geo_cp (mesh konkavligi) F1 0.400 ile zayif.

NEDEN SIMDI MUMKUN: bugun iki yeni alet kazanildi --
  1. calisan isin motoru (cp_geometry.ray_hits; onceden rtree yoklugunda sessizce patliyordu)
  2. B-rep erisimi (gmsh yuzey tipi/alan/yaricap; tessellation sirasinda ATILAN bilgi)

FIKIR: tel girisi bir MESH DUZENSIZLIGI degil, TAM TANIMLI bir imalat ozelligidir --
belirli yaricapta, yuzeye acilan, belirli derinlikte silindirik (ya da prizmatik) bir delik.
B-rep bunu ZATEN tam olarak tarif ediyor; mesh konkavligiyla tahmin etmeye gerek yok.

SUZGECLER (hepsi fiziksel, ogrenilmis parametre YOK):
  * yaricap araligi: tel kesitleri 0.5-4.0 mm (standart iletken capları)
  * yuzeye acilmali: agiz noktasindan disari isin malzeme kesmemeli
  * derinlik: kanal en az MIN_DEPTH mm devam etmeli (sig pah/vida yuvasi degil)
  * eksen: silindir ekseni yuzey normaline yakin olmali (yanal delik degil)
"""
import numpy as np

R_MIN, R_MAX = 0.5, 4.0        # tel giris yaricapi araligi (mm)
MIN_DEPTH = 2.0                # kanal en az bu kadar derin olmali (mm)
MERGE_MM = 3.0                 # ayni delige ait yuzeyleri birlestir


def _axis_of(lo, hi):
    """Silindir ekseni ~ bbox'in EN UZUN yonu (silindirde diger iki boyut = cap)."""
    ext = hi - lo
    a = np.zeros(3); a[int(np.argmax(ext))] = 1.0
    return a, float(ext.max())


def detect(step_path, V, F, r_min=R_MIN, r_max=R_MAX, min_depth=MIN_DEPTH):
    """STEP + remeshlenmis (V,F) -> CP adaylari [{'point','direction','radius','depth'}].

    (V, F) yalnizca isin testleri icin gerekir; aday URETIMI tamamen B-rep'ten gelir.
    """
    from cp_geometry import ray_hits
    import tel_g_brep as B

    surf = B.read_brep(step_path)
    mesh = (np.asarray(V, float), np.asarray(F, np.int64))
    cands = []
    for t, area, com, lo, hi, rad in surf:
        if t != "Cylinder" or not (r_min <= rad <= r_max):
            continue
        axis, length = _axis_of(lo, hi)
        if length < min_depth:
            continue
        # deligin IKI agzi: eksen boyunca silindirin uclari
        for sgn in (+1.0, -1.0):
            mouth = com + sgn * axis * (length / 2.0)
            out = sgn * axis
            # 1) AGIZ DISARI ACILMALI: agizdan disari isin malzeme kesmemeli
            if len(ray_hits(mesh, mouth + out * 0.3, out, 8.0)):
                continue
            # 2) ICERI dogru kanal DEVAM etmeli (sig degil): karsi yone ilk kesisim uzak olmali
            hin = ray_hits(mesh, mouth + out * 0.3, -out, 40.0)
            depth = float(hin[0]) if len(hin) else 40.0
            if depth < min_depth:
                continue
            cands.append({"point": mouth, "direction": -out,
                          "radius": float(rad), "depth": depth, "area": float(area)})

    # ayni fiziksel delige ait birden cok yuzey -> birlestir
    out_c = []
    for c in sorted(cands, key=lambda x: -x["depth"]):
        p = np.asarray(c["point"], float)
        if any(np.linalg.norm(p - np.asarray(k["point"], float)) < MERGE_MM for k in out_c):
            continue
        out_c.append(c)
    return out_c
