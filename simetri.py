# -*- coding: utf-8 -*-
"""SIMETRI KISITI: klemensler cogunlukla AYNA SIMETRIK.

FIKIR. Bir klemensin govdesi genelde bir duzleme gore simetriktir ve kontaklar
o simetrinin ESLERI halinde gelir. Bu, markadan tamamen BAGIMSIZ yapisal bir
bilgidir ve aday-basina oznitelikte YOKTUR: bir aday tek basina bakildiginda
sira disi gorunmezken, simetrigi olmadigi icin supheli olabilir.

Simetri duzlemi MESHTEN cikarilir (GT'den DEGIL):
  * parcanin ana eksenleri (SVD) aday normaller
  * her aday normal icin: tepe bulutunu yansit, en yakin komsu mesafelerinin
    medyanini olc; kucukse o duzlem simetri duzlemidir

Oznitelik (aday basina 4 sutun):
  sim_var        parcada simetri duzlemi bulundu mu
  sim_es_mesafe  bu adayin YANSIMASINA en yakin adayin mesafesi
  sim_es_skor    o adayin (varsa) skoru -- "esim de guclu mu"
  sim_kendi      aday duzlemin uzerinde mi (kendi kendinin esi)

TEZ: turetme/remesh degismez; bu bir puanlama olcusudur.
"""
import numpy as np

TOL_ORAN = 0.02        # medyan yansima mesafesi / kosegen -> simetri esigi
DUZLEM_TOL = 1.0       # mm; bu kadar yakinsa aday duzlemin uzerindedir


def _birim(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


def yansit(P, n, d):
    """P noktalarini `n . x = d` duzlemine gore yansit."""
    P = np.asarray(P, float).reshape(-1, 3)
    t = P @ n - d
    return P - 2.0 * t[:, None] * n[None, :]


def duzlem_bul(V, aday_n=None, orneklem=3000):
    """Mesh tepelerinden EN IYI ayna simetri duzlemi. Doner: (n, d, skor) ya da None.

    `skor` = medyan yansima mesafesi / kosegen; KUCUK iyidir.
    """
    V = np.asarray(V, float).reshape(-1, 3)
    if len(V) < 32:
        return None
    if len(V) > orneklem:
        V = V[np.random.default_rng(0).choice(len(V), orneklem, replace=False)]
    c = V.mean(0)
    kos = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0
    if aday_n is None:
        aday_n = np.linalg.svd(V - c, full_matrices=False)[2]
    en = None
    for n in np.asarray(aday_n, float).reshape(-1, 3):
        n = _birim(n)
        d = float(c @ n)                     # duzlem merkezden gecsin
        Y = yansit(V, n, d)
        # her yansimanin en yakin gercek tepeye mesafesi
        m = np.linalg.norm(Y[:, None, :] - V[None, :, :], axis=-1).min(1) \
            if len(V) <= 1500 else _en_yakin_parcali(Y, V)
        s = float(np.median(m)) / kos
        if en is None or s < en[2]:
            en = (n, d, s)
    if en is None or en[2] > TOL_ORAN:
        return None
    return en


def _en_yakin_parcali(Y, V, topak=500):
    out = np.empty(len(Y))
    for b in range(0, len(Y), topak):
        s = slice(b, b + topak)
        out[s] = np.linalg.norm(Y[s][:, None, :] - V[None, :, :],
                                axis=-1).min(1)
    return out


OZ_AD = ["sim_var", "sim_es_mesafe", "sim_es_skor", "sim_kendi"]


def oznitelik(P, V, skor=None, duzlem=None):
    """Aday basina simetri olculeri. Doner: (n, 4).

    `duzlem` verilmezse meshten bulunur (parca basina BIR KEZ bulup gecirmek
    cok daha ucuzdur).
    """
    P = np.asarray(P, float).reshape(-1, 3)
    X = np.zeros((len(P), len(OZ_AD)))
    if not len(P):
        return X
    X[:, 1] = 1e3
    dz = duzlem if duzlem is not None else duzlem_bul(V)
    if dz is None:
        return X
    n, d, _s = dz
    Y = yansit(P, n, d)
    D = np.linalg.norm(Y[:, None, :] - P[None, :, :], axis=-1)
    j = np.argmin(D, axis=1)
    m = D[np.arange(len(P)), j]
    X[:, 0] = 1.0
    X[:, 1] = m
    if skor is not None:
        X[:, 2] = np.asarray(skor, float)[j]
    X[:, 3] = (np.abs(P @ n - d) <= DUZLEM_TOL).astype(float)
    return X
