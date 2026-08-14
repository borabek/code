# -*- coding: utf-8 -*-
"""K2.2 + K2.1: SATIR AYRIMI + PERIYODIK YAYILIM (son-islem, tez NOTR).

Klemensler tekrarlanan kutuplardan olusur. Urunun bulduğu adaylardan SATIRLARI
cikarip her satirin ADIMINI olcer, +-N adim yayarak KACIRILAN kutuplari onerir.

TAVAN OLCULDU (sonda_k21_periyodiklik.py, D6):
  +-1 adim -> isabet %39.4, FN'lerin %21.4'u | +-2 -> %29.2 | +-6 -> %16.0
Urunun kendi tespit kesinligi 0.4673 -- yani +-1 AYNI LIGDE. **DAR tut.**

Onceki sondanin KUSURU giderildi: tek PCA ekseni cok sirali parcada sıralari
ust uste bindiriyordu. Burada ONCE satirlar ayrilir (yon benzerligi + dikey
mesafe), SONRA her satirda ayri periyot cikarilir.
"""
import numpy as np


def _satirlar(P, D, tol_mm=4.0, aci_cos=0.90):
    """Adaylari SATIRLARA ayir: yonu benzer + ortak bir dogruya yakin olanlar."""
    n = len(P)
    kalan = list(range(n))
    out = []
    while len(kalan) >= 3:
        i0 = kalan[0]
        # ayni yone bakanlar
        ayni = [i for i in kalan
                if abs(float(D[i] @ D[i0])) >= aci_cos]
        if len(ayni) < 3:
            kalan.remove(i0); continue
        Q = P[ayni]
        C = Q - Q.mean(0)
        _u, _s, Vt = np.linalg.svd(C, full_matrices=False)
        eks = Vt[0]
        # dogruya dik mesafe <= tol olanlar SATIRI olusturur
        t = C @ eks
        dik = np.linalg.norm(C - np.outer(t, eks), axis=1)
        grup = [ayni[k] for k in range(len(ayni)) if dik[k] <= tol_mm]
        if len(grup) >= 3:
            out.append((grup, eks))
            for g in grup:
                if g in kalan:
                    kalan.remove(g)
        else:
            kalan.remove(i0)
    return out


def yay(P, D, adim_n=1, tol_mm=4.0, min_adim=2.0, maks_adim=40.0):
    """Satir basina periyot cikar, +-adim_n yay. Doner: (yeni_P, yeni_D)."""
    P = np.asarray(P, float); D = np.asarray(D, float)
    if len(P) < 3:
        return np.zeros((0, 3)), np.zeros((0, 3))
    yeniP, yeniD = [], []
    for grup, eks in _satirlar(P, D, tol_mm):
        Q = P[grup]
        t = np.sort((Q - Q.mean(0)) @ eks)
        f = np.diff(t); f = f[f > 1e-6]
        if not len(f):
            continue
        adim = float(np.median(f))
        if not (min_adim <= adim <= maks_adim):
            continue
        for gi in grup:
            for s in range(-adim_n, adim_n + 1):
                if s == 0:
                    continue
                yeniP.append(P[gi] + s * adim * eks)
                yeniD.append(D[gi])
    if not yeniP:
        return np.zeros((0, 3)), np.zeros((0, 3))
    U = np.asarray(yeniP, float); V = np.asarray(yeniD, float)
    # MEVCUT adaylara ve BIRBIRINE cok yakin olanlari ELE
    tut = []
    for i in range(len(U)):
        if np.min(np.linalg.norm(P - U[i], axis=1)) <= tol_mm:
            continue
        if tut and np.min(np.linalg.norm(U[tut] - U[i], axis=1)) <= tol_mm:
            continue
        tut.append(i)
    return U[tut], V[tut]
