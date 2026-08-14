# -*- coding: utf-8 -*-
"""G5-b: OTO-ETIKETLERI `load_extra` BICIMINE CEVIR + SESSIZ DUSMEYI IMKANSIZ KIL.

`g5_agiz_etiket.py` her parcayi `_label_auto/<pid>.npz` olarak yaziyor (V/F/labels).
Ama `train_seg_extra.load_extra` **parca-basina-KLASOR** bekliyor:
    <kok>/<pid>/<pid>.obj  +  <kok>/<pid>.labels.txt

BU AYRIM SESSIZ BIR TUZAK: `load_extra` okuyamadigi klasoru `except: continue` ile
ATLIYOR. Yani bicim uymazsa egitim HATA VERMEZ -- yalnizca 71 parcayla kosar ve
"basarili" der. Kaynak dosyanin kendi yorumunda bu tam olarak yasanmis:
"this bug made a whole '+132 EEC' experiment train on 71 parts while reporting success".

BU YUZDEN IKI KORUMA:
  1. Donusturucu, yazdigi her parcayi GERI OKUR (`load_extra` ile) ve sayar.
  2. `dogrula()` egitimden ONCE cagrilir; beklenen sayiyi tutturamazsa HATA FIRLATIR.

TEZ DEGISMEZ: etiket iceriginе dokunulmaz, yalniz dosya bicimi cevrilir.
"""
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

KAYNAK_VARSAYILAN = "_label_auto"
HEDEF_VARSAYILAN = "_label_auto_obj"
# FEW-SHOT (K6.5-b): her kosum KENDI dizin ciftini kullanir. Sabit kalsaydi
# onceki kosumlarin parcalari birikir ve fine-tune k parca yerine yuzlercesiyle
# egitilirdi -- hata vermeden. (Ayni tuzak g5_agiz_etiket.py'de de vardi.)
# NOT: modul sabitini fonksiyon icinde ATAMAYIN; Python onu YEREL yapar ve
# argparse default'u UnboundLocalError verir (bu gece bir kez yasandi).
KAYNAK = KAYNAK_VARSAYILAN
HEDEF = HEDEF_VARSAYILAN


def yaz_obj(yol, V, F):
    with io.open(yol, "w", encoding="utf-8") as f:
        f.write("".join(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in V))
        f.write("".join(f"f {a+1} {b+1} {c+1}\n" for a, b, c in F))


def donustur(KAYNAK=KAYNAK_VARSAYILAN, HEDEF=HEDEF_VARSAYILAN):
    os.makedirs(HEDEF, exist_ok=True)
    n = atl = 0
    for fn in sorted(os.listdir(KAYNAK)):
        if not fn.endswith(".npz"):
            continue
        pid = os.path.splitext(fn)[0]
        d = os.path.join(HEDEF, pid)
        if os.path.exists(os.path.join(d, pid + ".obj")):
            atl += 1; continue
        try:
            z = np.load(os.path.join(KAYNAK, fn))
            V = np.asarray(z["V"], float); F = np.asarray(z["F"], np.int64)
            L = np.asarray(z["labels"], np.int64)
        except Exception:
            continue
        if len(V) != len(L) or not len(F) or F.max() >= len(V):
            continue
        if not set(np.unique(L)).issubset(set(range(5))):
            continue                       # 5 SINIF DISINA CIKAN ETIKET KABUL EDILMEZ
        os.makedirs(d, exist_ok=True)
        yaz_obj(os.path.join(d, pid + ".obj"), V, F)
        with io.open(os.path.join(d, pid + ".labels.txt"), "w", encoding="utf-8") as f:
            f.write(" ".join(map(str, L.tolist())))
        n += 1
    return n, atl


def dogrula(beklenen=None, HEDEF=HEDEF_VARSAYILAN):
    """load_extra GERCEKTEN kac parca okuyor? Egitimden ONCE cagrilir."""
    from train_seg_extra import load_extra
    R = load_extra(HEDEF)
    print(f"  load_extra('{HEDEF}') -> {len(R)} parca okudu")
    if R:
        L = np.concatenate([r["labels"] for r in R])
        u, c = np.unique(L, return_counts=True)
        print(f"  sinif dagilimi: {dict(zip(u.tolist(), (100*c/len(L)).round(2).tolist()))} (%)")
    if beklenen is not None and len(R) < beklenen:
        raise AssertionError(
            f"SESSIZ DUSME: {beklenen} parca yazildi ama load_extra {len(R)} okudu. "
            f"Egitim BASLATILMAZ -- yoksa 71 parcayla kosup 'basarili' der.")
    return len(R)


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--kaynak", default=KAYNAK_VARSAYILAN)
    _ap.add_argument("--hedef", default=HEDEF_VARSAYILAN)
    _a = _ap.parse_args()
    n, atl = donustur(_a.kaynak, _a.hedef)
    var = len([x for x in os.listdir(_a.kaynak) if x.endswith(".npz")])
    print(f"cevrildi {n} | zaten vardi {atl} | kaynak npz {var}")
    okunan = dogrula(beklenen=n + atl, HEDEF=_a.hedef)
    print(f"DOGRULANDI: {okunan} parca egitime hazir -> {_a.hedef}")
