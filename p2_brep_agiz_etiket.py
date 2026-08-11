# -*- coding: utf-8 -*-
"""P2 OTOMATIK: agiz etiketini B-rep'in GERCEK SINIRINDAN uret (insan gerekmez).

NEDEN ONCEKI OTO-ETIKET COKTU: `g5_agiz_etiket` GT noktasi etrafina SABIT
yaricapli disk boyuyordu. Olculdu ki o yolla egitilen g10, gorulmemis markada
DAHA AZ aday uretiyor (havuz recall 0.6654 -> 0.5847,
[[g10-zinciri-d7de-gerileme]]). Once h3 de r=2mm diskle atesleme oranini
%78'den %24'e cokertmisti ([[h3-highcp-finetune-dead]]).

FARK: artik agzin GERCEK SINIRI elimizde. B-rep silindir agzi kendi YARICAPINI
tasiyor, duzlemsel aciklik kendi halkasini. Etiket sabit bir diskle degil,
parcanin kendi geometrisiyle ciziliyor -- "metrik-hizali agiz segmentasyonu"nun
tam tanimi budur.

YONTEM (parca basina):
 1. Uretici GT'si (konum + yon) ile B-rep agizlarini ESLESTIR: GT'ye `ESLES_MM`
    icinde ve ekseni GT yonuyle `ESLES_ACI` icinde olan en yakin agiz.
 2. Eslesen agzin GERCEK disk bolgesini boya: eksene dik uzaklik <= R*`R_PAY`,
    agiz duzleminden eksenel derinlik [-`GERI`*R, +`ILERI`*R].
 3. Hicbir GT eslesmiyorsa parcayi HIC YAZMA (yanlis etiket bosluktan kotudur).

CIKTI SOZLESMESI (`train_seg_extra --partial-dir`): CableEntry=3 isaretlenir,
gerisi 0; maskeli kayip yalnizca CE-vs-degil'i denetler, isaretsiz sinifları
Housing diye OGRETMEZ.

TEZE SADIK: 5 sinif, ~6000 uniform izotropik remesh, `v_o` turetmesi DEGISMEZ.
Degisen tek sey ETIKETIN AGZA NE KADAR OTURDUGU.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("BA_ALLOW_SEEN", "1")

CE = 3
# ESLESME OLCUSU METRIGIN KENDISIYLE AYNI OLMALI.
# ILK SURUM YANLISTI: GT ile agiz merkezi arasi DUZ OKLID mesafesi kullaniyordum
# ve eslesme %19.5'te kaldi. Sebep: uretici CP'lerinin yaklasik yarisi GOVDE
# ICINDE duruyor ([[fiziksel-cerrahi-s-serisi]]), yani agizdan EKSENEL olarak
# uzakta. Urunun metrigi zaten eksenel kaymayi 40mm'ye kadar serbest birakiyor
# ve yalnizca YANAL hataya bakiyor. Etiket eslesmesi de oyle olmali.
ESLES_YANAL = 2.5     # GT'nin agiz EKSENINE dik uzakligi (mm)
ESLES_EKSENEL = 40.0  # eksen boyunca serbest kayma (mm) -- metrikle AYNI
ESLES_ACI = 25.0      # GT yonu ile agiz ekseni arasi en buyuk aci
R_PAY = 1.15          # agiz yaricapina pay (remesh tepe araligi icin)
GERI, ILERI = 0.6, 1.2   # agiz duzleminden geri/ileri eksenel band (R kati)
EN_AZ_TEPE = 4        # bu kadar tepe boyanmayan agiz SAYILMAZ


def _birim(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def agizlar(cyl, acik):
    """(merkez, eksen, yaricap) listesi -- silindir agizlari + duzlemsel acikliklar."""
    out = []
    for c in cyl or []:
        a = _birim(np.asarray(c["axis"], float))
        if a is None:
            continue
        r = float(c.get("radius", 0.0))
        if r <= 1e-6:
            continue
        for u, s in ((c.get("mouth_a"), 1.0), (c.get("mouth_b"), -1.0)):
            if u is not None:
                out.append((np.asarray(u, float), s * a, r))
    for o in acik or []:
        if not isinstance(o, dict) or o.get("center") is None:
            continue
        n = _birim(np.asarray(o.get("normal", [0, 0, 1]), float))
        r = float(o.get("esd_r", 0.0))
        if n is None or r <= 1e-6:
            continue
        out.append((np.asarray(o["center"], float), n, r))
    return out


def boya(V, G, Gd, cyl, acik):
    """Doner: (etiket dizisi, eslesen GT sayisi, toplam GT)."""
    L = np.zeros(len(V), np.int64)
    A = agizlar(cyl, acik)
    if not A:
        return L, 0, len(G)
    M = np.asarray([a[0] for a in A], float)
    AX = np.asarray([a[1] for a in A], float)
    eslesen = 0
    for j in range(len(G)):
        u = _birim(np.asarray(Gd[j], float))
        if u is None:
            continue
        # YANAL/EKSENEL AYRIMI: GT'yi her agzin KENDI ekseni boyunca ayristir.
        w = G[j][None] - M
        eks = np.einsum("ij,ij->i", w, AX)
        yan = np.linalg.norm(w - eks[:, None] * AX, axis=1)
        aci = np.degrees(np.arccos(
            np.clip(np.abs(AX @ u), -1.0, 1.0)))
        aday = np.where((yan <= ESLES_YANAL) &
                        (np.abs(eks) <= ESLES_EKSENEL) &
                        (aci <= ESLES_ACI))[0]
        if not len(aday):
            continue
        # en iyi = yanal hatasi en kucuk olan (metrigin baktigi buyukluk)
        i = int(aday[np.argmin(yan[aday])])
        m, a, r = A[i]
        w = V - m
        eks = w @ a
        dik = np.linalg.norm(w - eks[:, None] * a[None], axis=1)
        k = (dik <= r * R_PAY) & (eks >= -GERI * r) & (eks <= ILERI * r)
        if int(k.sum()) < EN_AZ_TEPE:
            continue
        L[k] = CE
        eslesen += 1
    return L, eslesen, len(G)


def obj_yaz(yol, V, F):
    with open(yol, "w") as f:
        for v in V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in F:
            f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cikti", default="_p2_brep_etiket")
    ap.add_argument("--en-az-oran", type=float, default=0.5,
                    help="parcanin GT'lerinin en az bu orani eslesmeli")
    a = ap.parse_args()
    import d6_kayit
    import kanonik_d7 as K

    os.makedirs(a.cikti, exist_ok=True)
    d7 = {str(p) for p in json.load(
        open("results/d7_sinav_kumesi.json"))["pidler"]}

    isler = [
        ("korpus", "results/_p1_olasilik_brepegit",
         "results/_brepegit_silindirler.pkl", "results/_brepegit_acikliklar.pkl",
         K.yukle([str(p) for p in json.load(
             open("results/brep_egitim_kumesi.json"))["pidler"]])),
        ("d6", "results/_p1_olasilik_g7", "results/_d6_silindirler.pkl",
         "results/_d6_acikliklar.pkl",
         d6_kayit.yukle(set(d6_kayit.sinav()["pidler"]))),
    ]
    top = {"yazilan": 0, "atlanan": 0, "gt": 0, "eslesen": 0}
    t0 = time.time()
    for ad, ob, cylf, acf, kay in isler:
        cy = pickle.load(open(cylf, "rb"))
        ac = pickle.load(open(acf, "rb"))
        n = 0
        for pid, r in sorted(kay.items()):
            pid = str(pid)
            if pid in d7:
                raise SystemExit(f"SIZINTI: {pid} D7'de -- egitim etiketi URETILMEZ")
            G = np.asarray(r.get("G", []), float)
            f = f"{ob}/{pid}.npz"
            if not len(G) or not os.path.exists(f):
                top["atlanan"] += 1
                continue
            z = np.load(f)
            V = np.asarray(z["V"], float)
            F = np.asarray(z["F"], np.int64)
            L, es, ng = boya(V, G, np.asarray(r["Gd"], float),
                             cy.get(pid), ac.get(pid))
            top["gt"] += ng
            top["eslesen"] += es
            if ng == 0 or es / ng < a.en_az_oran:
                top["atlanan"] += 1
                continue
            d = os.path.join(a.cikti, pid)
            os.makedirs(d, exist_ok=True)
            obj_yaz(os.path.join(d, f"{pid}.obj"), V, F)
            np.savetxt(os.path.join(d, f"{pid}.labels.txt"), L, fmt="%d")
            top["yazilan"] += 1
            n += 1
            if n % 200 == 0:
                print(f"  {ad} {n} yazildi ({time.time()-t0:.0f}s)", flush=True)
        print(f"{ad} bitti: {n} parca", flush=True)

    print(f"\nYAZILAN {top['yazilan']} | ATLANAN {top['atlanan']}")
    print(f"GT eslesme: {top['eslesen']}/{top['gt']} "
          f"(%{100*top['eslesen']/max(top['gt'],1):.1f})")
    json.dump({"yazilan": top["yazilan"], "atlanan": top["atlanan"],
               "gt": top["gt"], "eslesen": top["eslesen"],
               "esles_yanal": ESLES_YANAL, "esles_eksenel": ESLES_EKSENEL,
               "esles_aci": ESLES_ACI, "r_pay": R_PAY,
               "band": [GERI, ILERI], "en_az_oran": a.en_az_oran,
               "not": "B-rep GERCEK agiz sinirindan uretilen kismi CableEntry "
                      "etiketi. D7 kesisimi SIFIR (kod icinde assert)."},
              open("results/p2_brep_etiket.json", "w"), indent=1)
    print(f"-> {a.cikti}")


if __name__ == "__main__":
    main()
