# -*- coding: utf-8 -*-
"""KANONIK CERCEVE: adayi parcanin KENDI eksen sisteminde ifade et.

SORUN. Model bugun adayin konumunu ve yonunu DUNYA koordinatlarinda goruyor.
Ayni klemens CAD'de 90 derece dondurulmus modellenmisse ogrenilen her konumsal
kalip bozulur. Gorulmemis MARKA sinavinda bu dogrudan zarar: yeni ureticinin
modelleme ekseni bizimkiyle ayni olmak zorunda degil.

COZUM. Parcanin kendi ana eksenlerinden (PCA) bir cerceve kurulur ve aday
konumu/yonu bu cerceveye yansitilir. Boylece "govdenin ust yuzunde, uzun eksen
boyunca sirali" gibi bir kalip MARKADAN BAGIMSIZ ogrenilir.

ISARET BELIRSIZLIGI. PCA ekseninin yonu keyfidir (e ve -e ayni eksen). Uc
momente (carpiklik) gore isaret sabitlenir; simetrik govdelerde carpiklik ~0
oldugu icin bu KARARSIZ kalir. Bu yuzden hem ISARETLI hem MUTLAK sutunlar
verilir: mutlak sutunlar isaret secimi ne olursa olsun ayni kalir.

Sutunlar (10):
  k1,k2,k3    aday konumunun eksen boyu yeri, o eksendeki uzanima bolunmus [-1,1]
  d1,d2,d3    secenek yonunun eksenlerle ic carpimi (ISARETLI)
  a1,a2,a3    ayni ic carpimlarin MUTLAK degeri (isaretten bagimsiz)
  uzanim_or    en kisa / en uzun eksen orani (govde biciminin kaba olcusu)
"""
import numpy as np

OZ_AD = ["k1", "k2", "k3", "d1", "d2", "d3", "a1", "a2", "a3", "uzanim_or"]


def _birim(V):
    V = np.asarray(V, float).reshape(-1, 3)
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)


def cerceve(V):
    """(merkez, E, uzanim). `E` satirlari birim eksenler, uzunlugu azalan sirada."""
    V = np.asarray(V, float).reshape(-1, 3)
    merkez = V.mean(0)
    Q = V - merkez
    if len(V) < 3:
        return merkez, np.eye(3), np.ones(3)
    # kovaryansin ozvektorleri = ana eksenler
    _, s, Vt = np.linalg.svd(Q, full_matrices=False)
    E = Vt[:3]
    if E.shape[0] < 3:                       # dejenere (duzlemsel) govde
        E = np.vstack([E, np.eye(3)[:3 - E.shape[0]]])
    # ISARETI carpiklikla sabitle -- deterministik olmali
    for i in range(3):
        pr = Q @ E[i]
        c = float((pr ** 3).sum())
        if c < 0:
            E[i] = -E[i]
        elif c == 0.0 and float(E[i].sum()) < 0:
            E[i] = -E[i]                     # tam simetride sabit bir kural
    # sag el sistemi
    if float(np.dot(np.cross(E[0], E[1]), E[2])) < 0:
        E[2] = -E[2]
    uzanim = np.array([max(float(np.ptp(Q @ E[i])), 1e-9) for i in range(3)])
    return merkez, E, uzanim


def oznitelik(P, D, V):
    """(n, 10). `V` parcanin tepe noktalari (cerceve ondan kurulur)."""
    P = np.asarray(P, float).reshape(-1, 3)
    D = _birim(D)
    n = len(P)
    X = np.zeros((n, len(OZ_AD)))
    if not n:
        return X
    V = np.asarray(V, float).reshape(-1, 3)
    if len(V) < 3:
        V = P if len(P) >= 3 else np.vstack([P, P + 1.0])
    merkez, E, uzanim = cerceve(V)
    Q = P - merkez
    for i in range(3):
        X[:, i] = np.clip((Q @ E[i]) / (0.5 * uzanim[i]), -3.0, 3.0)
        X[:, 3 + i] = D @ E[i]
    X[:, 6:9] = np.abs(X[:, 3:6])
    X[:, 9] = float(uzanim.min() / uzanim.max())
    return X
