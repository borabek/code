# -*- coding: utf-8 -*-
"""D20 UCUNCU SIZINTI KATMANI: GEOMETRI IKIZI dislama.

Marka onegi ve parca kimligi yetmez -- ayni parcanin BASKA bir marka/kimlik
altindaki IKIZI de sinav bilgisini tasir ([[geometry-twin-leakage]]: parcalarin
%80'inin ikizi var). `results/_geo_yasak.json` sinav/LOCKED parcalarinin
geometri anahtarlarini tutuyor (609 adet).

Bu betik URETILMIS korpusu tarar, ikiz cikanlari KARANTINAYA tasir.
"""
import io, json, os, shutil, sys
import numpy as np
sys.path.insert(0, ".")
import geometri_anahtar

KORPUS = "_label_ds1_obj"
KARANTINA = "_KARANTINA_d20_geo_ikiz"


def obj_oku(yol):
    V, F = [], []
    for ln in io.open(yol, encoding="utf-8"):
        if ln.startswith("v "):
            V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("f "):
            F.append([int(t.split("/")[0]) - 1 for t in ln.split()[1:4]])
    return np.asarray(V, float), np.asarray(F, int)


def main():
    yasak = set(json.load(io.open("results/_geo_yasak.json", encoding="utf-8")).values())
    print(f"yasak geometri anahtari: {len(yasak)}")
    adlar = [d for d in os.listdir(KORPUS) if os.path.isdir(os.path.join(KORPUS, d))]
    print(f"korpus: {len(adlar)} parca")
    os.makedirs(KARANTINA, exist_ok=True)
    ikiz, hata = [], 0
    for i, ad in enumerate(adlar, 1):
        o = os.path.join(KORPUS, ad, ad + ".obj")
        if not os.path.exists(o):
            hata += 1; continue
        try:
            V, F = obj_oku(o)
            k = geometri_anahtar.anahtar(V, F)
        except Exception:
            hata += 1; continue
        if k in yasak:
            ikiz.append(ad)
            shutil.move(os.path.join(KORPUS, ad), os.path.join(KARANTINA, ad))
        if i % 500 == 0:
            print(f"  {i}/{len(adlar)} | ikiz {len(ikiz)}", flush=True)
    print(f"\nIKIZ BULUNAN: {len(ikiz)} parca -> {KARANTINA}")
    print(f"KALAN TEMIZ KORPUS: {len(adlar)-len(ikiz)-hata} | okunamayan {hata}")
    json.dump({"ikiz": ikiz, "n_yasak_anahtar": len(yasak),
               "kalan": len(adlar) - len(ikiz) - hata},
              io.open("results/d20_geo_ikiz.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
