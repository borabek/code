# -*- coding: utf-8 -*-
"""A1 -- ISIN ATMA ile DELIK EKSENI: yon bir SINIFLANDIRMA degil OLCUM isi

FIZIK. Bir kablo girisi bir DELIKTIR. Ekseni boyunca uzun ve engelsiz bir
kanal vardir; dik yonde birkac mm'de duvara carpilir. O halde eksen,
"govde icinde en uzun TUP"un yonudur.

TUP SKORU. Bir u yonu icin:
    d0   = u boyunca ilk carpma mesafesi (CAP ile sinirli, carpma yoksa CAP)
    halka= u cevresinde THETA acisinda k isinin ilk carpma mesafeleri
    tup  = d0 / (ortanca(halka) + eps)
Kanalda d0 buyuk, halka kucuktur (delik duvari) -> tup yuksek.
Duz yuzeyde ikisi de benzer -> tup ~ 1.

NEDEN BU KOL DIGER YON KOLLARINDAN FARKLI:
  * B-rep agzi CP'de bulunma 0.011 -> B-rep'e BAGIMLI DEGIL
  * mesh normali secili adaylarda  0.334 -> yerel normale BAGIMLI DEGIL
  * yalnizca ucgenler ve isin-ucgen kesisimi gerekir

ISARET. Tup skoru u ile -u'yu ayirmaz (tup iki yone de uzanir). Isaret
AYRICA "govdeden disari" kuralindan turetilir (GT sozlesmesi 1.000 disari).
"""
import numpy as np

CAP = 40.0
THETA = 25.0
N_HALKA = 8
EPS_ILERI = 0.15      # baslangici yuzeyden ayir (kendi ucgenine carpmasin)


def _birim(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _dik_ikili(u):
    """u'ya dik iki birim vektor."""
    a = np.array([1.0, 0.0, 0.0])
    if abs(u @ a) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    e1 = _birim(np.cross(u, a).reshape(1, 3))[0]
    e2 = np.cross(u, e1)
    return e1, e2


def halka_yonleri(u, theta=THETA, k=N_HALKA):
    """u cevresinde theta acisinda k yon."""
    u = _birim(np.asarray(u, float).reshape(1, 3))[0]
    e1, e2 = _dik_ikili(u)
    t = np.radians(theta)
    fi = np.linspace(0, 2 * np.pi, k, endpoint=False)
    return _birim(np.cos(t) * u[None, :] +
                  np.sin(t) * (np.cos(fi)[:, None] * e1[None, :] +
                               np.sin(fi)[:, None] * e2[None, :]))


def ilk_carpma(isinci, kokler, yonler, cap=CAP):
    """her (kok, yon) icin ilk carpma mesafesi; carpma yoksa cap."""
    kokler = np.asarray(kokler, float).reshape(-1, 3)
    yonler = _birim(np.asarray(yonler, float).reshape(-1, 3))
    k0 = kokler + EPS_ILERI * yonler
    yer, ind_r = isinci.intersects_location(k0, yonler,
                                            multiple_hits=False)[:2]
    d = np.full(len(yonler), cap, float)
    if len(ind_r):
        uz = np.linalg.norm(yer - k0[ind_r], axis=1) + EPS_ILERI
        # ayni isin birden fazla donerse en kucugu al
        for i, r_ in enumerate(ind_r):
            if uz[i] < d[r_]:
                d[r_] = uz[i]
    return np.minimum(d, cap)


def tup_skoru(isinci, kok, yonler, theta=THETA, k=N_HALKA, cap=CAP):
    """her yon icin tup skoru. yonler: (n,3). doner: (n,)"""
    yonler = _birim(np.asarray(yonler, float).reshape(-1, 3))
    n = len(yonler)
    # eksen isinlari
    d0 = ilk_carpma(isinci, np.tile(kok, (n, 1)), yonler, cap)
    # halka isinlari
    hy = np.vstack([halka_yonleri(u, theta, k) for u in yonler])
    hd = ilk_carpma(isinci, np.tile(kok, (n * k, 1)), hy, cap)
    hd = hd.reshape(n, k)
    return d0 / (np.median(hd, axis=1) + 1e-6)


def kendini_dogrula():
    """SENTETIK YETENEK TESTI: ekseni BILINEN bir delikte bulabiliyor mu.

    Bu adim S4'te yanlis negatiften kurtarmisti: mekanizma bilinen bir
    delikte ekseni bulamiyorsa gercek veride aramanin anlami yok.
    """
    import trimesh
    ok = True
    # EGIK EKSENLER: agiz +z YUZUNDEN ciksin diye z bileseni baskin secilir.
    # Ilk denememde eksen [1,1,0.3] idi ve `eksen*15` kutunun YUZUNE degil
    # KENARINA dusuyordu -- yani agzin olmadigi bir noktadan isin atiyordum
    # (17.9 derece sapma mekanizmanin degil TESTIN kusuruydu).
    for eksen in (np.array([0.0, 0.0, 1.0]),
                  _birim(np.array([[0.30, 0.20, 1.0]]))[0],
                  _birim(np.array([[0.55, -0.35, 1.0]]))[0]):
        kutu = trimesh.creation.box(extents=(30, 30, 30))
        T = trimesh.geometry.align_vectors([0, 0, 1], eksen)
        sil = trimesh.creation.cylinder(radius=2.0, height=80.0, transform=T)
        parca = kutu.difference(sil)
        assert parca.is_watertight, "sentetik parca su gecirmez degil"
        isinci = trimesh.ray.ray_triangle.RayMeshIntersector(parca)
        # delik agzi: eksenin +z yuzunu (z=15) deldigi nokta
        kok = eksen * (15.0 / eksen[2])
        assert np.all(np.abs(kok[:2]) < 14.0), "agiz yuzde degil, kenarda"
        # 64 aday yon (Fibonacci)
        i = np.arange(64) + 0.5
        fi = np.arccos(1 - 2 * i / 64)
        te = np.pi * (1 + 5 ** 0.5) * i
        Y = np.c_[np.cos(te) * np.sin(fi), np.sin(te) * np.sin(fi),
                  np.cos(fi)]
        # dogru yonu listeye KOY (gercek veride de aday listesinden secilir)
        Y = np.vstack([Y, eksen, -eksen])
        t = tup_skoru(isinci, kok, Y)
        sec = Y[int(np.argmax(t))]
        aci = np.degrees(np.arccos(np.clip(abs(float(sec @ eksen)), -1, 1)))
        print(f"  eksen {np.round(eksen, 2)} -> secilen sapma {aci:5.1f} "
              f"derece | tup {t.max():.2f} (ortanca {np.median(t):.2f})")
        ok &= aci <= 10.0
    print("SENTETIK YETENEK:", "GECTI" if ok else "KALDI")
    return ok


if __name__ == "__main__":
    kendini_dogrula()
