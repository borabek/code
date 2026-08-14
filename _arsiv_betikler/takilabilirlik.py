# -*- coding: utf-8 -*-
"""K1.1: TAKILABILIRLIK TESTI -- fizik ONERIR, ogrenme ELER.

BUGUNKU MIMARI: ogrenme onerir -> gate eler. Gorulmemis ureticide ONERI asamasi
cokuyor (ADAY_YOK, GT kaybinin %32'si) ve yon secimi cokuyor (aci hatasi IKI TEPELI:
%43.9 dogru, %51.1 tamamen yanlis, ortada rafine edilebilir bant yalniz %5).

BU MODUL TERSINE CEVIRIR: fizik onerir -> ogrenme eler.

PRIMITIF -- SILINDIR YURUTME: bir (nokta, yon) cifti icin, yaricapi r olan bir silindiri
katiya DEGMEDEN d mm sokabiliyor muyuz? Sokabiliyorsak orasi FIZIKSEL olarak bir kablo
girisidir. Bu, robot metriginin BIREBIR tanimidir (ferrul giriyor mu?).

NEDEN TRANSFER GARANTISI YAPISAL: hicbir sey EGITILMIYOR. "Gorulmemis uretici" diye bir
kavram yok -- ayni geometri her markada ayni sonucu verir.

OLU KAYITLI "SAF CAD DEDEKTORU"NDEN FARKI: orada geometri KARAR veriyordu ve tavan
0.332'de tikanmisti. Burada geometri yalnizca ONERIYOR; "bu delik kablo girisi mi, alet
yuvasi mi, montaj deligi mi" ayrimini yine OGRENILMIS model yapiyor. Bu ayrim SEMANTIK
ve semantik geometriden daha iyi transfer eder.

TEZ: `v_o` turetmesi, 5 sinif ve ~6000 remesh AYNEN kalir. Bu bir ADAY URETECIDIR ve
dagitilirsa tez-yolu-tek-basina sayisi YAN YANA raporlanir.
"""
import numpy as np

R_ADAY = (0.9, 1.4, 2.0, 2.8, 4.0)   # ferrul yaricaplari (mm) -- 1.5-8mm^2 iletken bandi
DERINLIK = 6.0                        # bu kadar mm engelsiz girebilmeli
N_ISIN = 8                            # silindir kesitini orneklemek icin cevre isini


def _fibonacci_yon(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def _dik_taban(d):
    a = np.array([0.0, 0, 1.0])
    if abs(float(d @ a)) > 0.9:
        a = np.array([1.0, 0, 0])
    e1 = np.cross(d, a); e1 /= (np.linalg.norm(e1) + 1e-12)
    return e1, np.cross(d, e1)


def silindir_gecer_mi(mesh, p, d, r, derinlik=DERINLIK, n_isin=N_ISIN):
    """Yaricapi r olan silindir, p'den d yonunde `derinlik` mm engelsiz gidebiliyor mu?

    Silindirin CEVRESINDEN n_isin adet paralel isin atilir; hepsi `derinlik` mm
    engelsiz giderse silindir gecer. Merkez isini de dahil edilir.
    """
    d = np.asarray(d, float); d = d / (np.linalg.norm(d) + 1e-12)
    e1, e2 = _dik_taban(d)
    ac = np.linspace(0, 2 * np.pi, n_isin, endpoint=False)
    O = [np.asarray(p, float)]
    for a in ac:
        O.append(np.asarray(p, float) + r * (np.cos(a) * e1 + np.sin(a) * e2))
    O = np.asarray(O, float)
    D = np.repeat(d[None, :], len(O), 0)
    try:
        loc, idx, _t = mesh.ray.intersects_location(O, D, multiple_hits=False)
    except Exception:
        return False, 0.0
    mes = np.full(len(O), np.inf)
    for L, i in zip(loc, idx):
        m = float(np.linalg.norm(L - O[i]))
        if m < mes[i]:
            mes[i] = m
    en_kisa = float(np.min(mes))
    return bool(en_kisa >= derinlik), en_kisa


def en_buyuk_gecen_yaricap(mesh, p, d, derinlik=DERINLIK):
    """Bu (nokta, yon) icin gecebilen EN BUYUK ferrul yaricapi. Gecmiyorsa 0."""
    en = 0.0
    for r in R_ADAY:
        ok, _ = silindir_gecer_mi(mesh, p, d, r, derinlik)
        if ok:
            en = r
        else:
            break
    return en


def yon_ara(mesh, p, yonler=None, derinlik=DERINLIK, n_yon=26):
    """Bir NOKTA icin en iyi giris yonu: en buyuk ferrulu en derine sokan yon.

    Doner: (yon, yaricap, derinlik) -- hicbiri gecmezse (None, 0, 0).
    """
    Y = _fibonacci_yon(n_yon) if yonler is None else np.asarray(yonler, float)
    en = (None, 0.0, 0.0)
    for d in Y:
        r = en_buyuk_gecen_yaricap(mesh, p, d, derinlik)
        if r > en[1]:
            _ok, mes = silindir_gecer_mi(mesh, p, d, r, derinlik)
            en = (d / (np.linalg.norm(d) + 1e-12), r, mes)
    return en


def kanal_var_mi(mesh, p, d, r, once=1.0, sonra=5.0, n_ac=8):
    """FIZIKSEL SORU: bu noktada, bu yonde yaricapi r olan bir KANAL var mi?

    Iki yanlis kurgudan sonra dogru formulasyon (2026-08-07):
      * v1: GT noktasindan ISIN at -> GT'lerin ~%47'si GOVDE ICINDE oldugu icin hicbir
        yon gecmiyordu (ONV 2/47, SE 0/36).
      * v2: parcanin ceyrek kosegeni kadar DISARIDAN basla -> telin once uzun bir serbest
        koridoru gecmesini sart kosuyor; yogun dizide komsu kutuplar kapatiyor (22/184).
      * v3 (bu): kanal LOKAL bir ozelliktir. Eksen boyunca [-once, +sonra] araliginda
        ornekle; her ornekte eksene DIK r yaricapli diskin BOS olmasini iste.
        Uzaktaki engeller onemsiz -- sorulan sey "ferrul buraya oturur mu".
    """
    d = np.asarray(d, float); d = d / (np.linalg.norm(d) + 1e-12)
    e1, e2 = _dik_taban(d)
    ac = np.linspace(0, 2 * np.pi, n_ac, endpoint=False)
    cev = np.stack([np.cos(a) * e1 + np.sin(a) * e2 for a in ac], 0)
    ts = np.linspace(-once, sonra, max(3, int((once + sonra) / 0.8)))
    O, D = [], []
    for t in ts:
        m = np.asarray(p, float) + t * d
        for c in cev:
            O.append(m); D.append(c)          # eksene DIK isin: r icinde engel var mi
    O = np.asarray(O, float); D = np.asarray(D, float)
    try:
        loc, idx, _t = mesh.ray.intersects_location(O, D, multiple_hits=False)
    except Exception:
        return False
    mes = np.full(len(O), np.inf)
    for L, i in zip(loc, idx):
        v = float(np.linalg.norm(L - O[i]))
        if v < mes[i]:
            mes[i] = v
    return bool(np.min(mes) >= r)


def kanal_yon_ara(mesh, p, n_yon=42, r_min=0.9):
    """Bu noktada EN BUYUK kanali veren yon. Doner: (yon, yaricap)."""
    Y = _fibonacci_yon(n_yon)
    en = (None, 0.0)
    for d in Y:
        for r in R_ADAY:
            if r < r_min:
                continue
            if kanal_var_mi(mesh, p, d, r):
                if r > en[1]:
                    en = (d, r)
            else:
                break
    return en
