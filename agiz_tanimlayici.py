# -*- coding: utf-8 -*-
"""AGIZ TANIMLAYICILARI: "buraya gercekten tel girer mi?"

NEDEN: B-rep havuzu robot tavanini 0.3412 -> 0.5748 acti ama hicbir secici
kullanamadi (brep-havuz-tavan-acildi-secici-acamadi). Sebep, denenen 58
ozniteligin bir agzin TEL GIRISI mi VIDA DELIGI mi ALET YUVASI mi oldugunu
soylememesi. Bu modul o bilgiyi URETIR -- hazir oznitelik aramak yerine.

Hepsi FIZIKSEL ve etiketsiz hesaplanabilir:
  yaricap        silindir yaricapi
  derinlik       delik uzunlugu (iki agiz arasi)
  narinlik       derinlik / yaricap (vida deligi sig, tel kanali derin)
  girme          agizdan ICERI serbest yol (eksende ilk carpisma)
  girme_kenar    ayni ama yaricapin %70'i kadar YANA kaydirilmis 4 isin -> telin
                 govdesi gercekten giriyor mu (eksende bos, kenarda dolu = sahte)
  erisim         agizdan DISARI serbest yol -> robot bu agza ulasabiliyor mu
  es_eksen       parcada ayni yone bakan ve yaricapi yakin kardes sayisi
                 (gercek CP'ler SIRA olusturur; tekil delik cogunlukla vida)
  aralik_duzeni  kardeslerin eksen boyu araliklarinin duzenliligi (1 = kusursuz)
  yaricap_yuzde  yaricapin PARCA ICINDEKI yuzdeligi (marka olceginden bagimsiz)

TEZE SADIK: segmentasyon, remesh, `v_o` turetmesi DEGISMEZ. Bunlar yalnizca
ADAY PUANLAMA icin ek olculerdir.
"""
import numpy as np

AD = ["yaricap", "derinlik", "narinlik", "girme", "girme_kenar", "erisim",
      "es_eksen", "aralik_duzeni", "yaricap_yuzde"]


TOPAK = 1500        # tek cagrida atilacak en fazla isin


def _ilk_mesafe(mesh, O, Dv, uzak):
    """Isin basina ilk carpisma mesafesi. trimesh imzasindan BAGIMSIZ.

    `intersects_location` (konum, isin_idx, ucgen_idx) dondurur; isin indeksini
    ACIKCA kullaniriz. `intersects_id`'nin donus sirasi surumler arasi degisiyor
    ve sessizce yanlis sutunu okumak bu projede daha once yasandi.

    TOPAKLI (2026-08-11): `multiple_hits=True` her isin icin TUM kesisimleri
    dondurur. Mesh tepesi havuzu acilinca parca basina secenek ~200'den ~5000'e
    cikti ve tek cagri 14 MILYON kesisim uretip belleği patlatti (8 payin 4'u
    MemoryError ile oldu). Topaklama sonucu DEGISTIRMEZ -- her isin icin en
    kucuk mesafe alindigi icin bolerek hesaplamak ayni sayiyi verir.
    """
    out = np.full(len(O), float(uzak))
    if not len(O):
        return out
    for b in range(0, len(O), TOPAK):
        s = slice(b, b + TOPAK)
        loc, ir, _tri = mesh.ray.intersects_location(
            O[s], Dv[s], multiple_hits=True)
        if not len(loc):
            continue
        Ob = O[s]
        d = np.linalg.norm(loc - Ob[ir], axis=1)
        alt = out[s].copy()
        np.minimum.at(alt, ir, d)      # dongu yerine vektorel
        out[s] = alt
    return out


def _dik_eksenler(ax):
    a = np.array([1.0, 0.0, 0.0])
    if abs(float(ax @ a)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(ax, a)
    n = np.linalg.norm(u)
    if n < 1e-12:
        return np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    u = u / n
    return u, np.cross(ax, u)


def tanimla(P, D, met, mesh, diag):
    """(n,9) tanimlayici matrisi. `met` = brep_adaylari(meta=True) ucuncu ogesi."""
    P = np.asarray(P, float).reshape(-1, 3)
    D = np.asarray(D, float).reshape(-1, 3)
    n = len(P)
    X = np.zeros((n, len(AD)))
    if n == 0:
        return X
    uzak = float(diag)
    rad = np.array([float(m.get("radius", m.get("esd_r", 0.0)) or 0.0)
                    for m in met])
    der = np.zeros(n)
    for i, m in enumerate(met):
        a, b = m.get("mouth_a"), m.get("mouth_b")
        if a is not None and b is not None:
            der[i] = float(np.linalg.norm(np.asarray(a, float) -
                                          np.asarray(b, float)))
    X[:, 0] = rad
    X[:, 1] = der
    X[:, 2] = der / np.maximum(rad, 1e-6)

    eps = max(1e-3, 1e-4 * diag)
    X[:, 3] = _ilk_mesafe(mesh, P - eps * D, -D, uzak)     # ICERI
    X[:, 5] = _ilk_mesafe(mesh, P + eps * D, D, uzak)      # DISARI

    ken = np.full(n, uzak)
    O, Dv, sahip = [], [], []
    for i in range(n):
        if rad[i] <= 1e-6:
            continue
        u, v = _dik_eksenler(D[i])
        for off in (u, -u, v, -v):
            O.append(P[i] + 0.7 * rad[i] * off - eps * D[i])
            Dv.append(-D[i])
            sahip.append(i)
    if O:
        d = _ilk_mesafe(mesh, np.asarray(O), np.asarray(Dv), uzak)
        for j, i in enumerate(sahip):
            if d[j] < ken[i]:
                ken[i] = d[j]
    X[:, 4] = ken

    K = np.zeros(n)
    DUZ = np.zeros(n)
    if n > 1:
        C = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
        cos = np.abs(C @ C.T)
        for i in range(n):
            k = (cos[i] > 0.99) & (np.abs(rad - rad[i]) < 0.3)
            k[i] = False
            K[i] = float(k.sum())
            if K[i] >= 2:
                # SIRA YONU VERIDEN CIKARILIR, secilmez. Ilk surumde eksene dik
                # RASTGELE bir `u`ya yansitiyordum; kardesler o eksende cakisinca
                # araliklar sifir cikiyor ve DUZENSIZ bir dizi "kusursuz" (1.0)
                # gorunuyordu. Testi bu yakaladi.
                off = np.vstack([P[k] - P[i], np.zeros(3)])
                off = off - off.mean(0)
                yay = np.linalg.norm(off, axis=1).max()
                if yay < 1e-6:
                    continue                       # hepsi cakisik -> duzen YOK
                u = np.linalg.svd(off, full_matrices=False)[2][0]
                t = np.sort(off @ u)
                fark = np.diff(t)
                ort = abs(float(fark.mean()))
                if len(fark) and ort > 1e-6:
                    DUZ[i] = 1.0 / (1.0 + float(fark.std()) / ort)
        X[:, 8] = (rad[:, None] > rad[None, :]).mean(1)
    else:
        X[:, 8] = 0.5
    X[:, 6] = K
    X[:, 7] = DUZ
    return X
