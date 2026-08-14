# -*- coding: utf-8 -*-
"""P4-b: MESH tabanli ek poz dallari -- B-rep'in ulasamadigi morfolojiler icin.

Silindir + duzlemsel aciklik dallari kahini 0.4002'de tikadi (P4 kapisi 0.82).
Iki dal daha, ikisi de MESH'ten (B-rep gerektirmez, dolayisiyla SE/NIT gibi
"silindiri olmayan" parcalarda da calisir):

  AGIZ-HALKASI DUZLEM NORMALI: adayin cevresindeki yuzey tepelerine en iyi oturan
    duzlemin normali. Tezin `v_o - v_s`'sinden BAGIMSIZ bir kestirim -- o, oturma
    noktasi ile agiz merkezini birlestirir; bu ise agiz KONTURUNUN yonelimini kullanir.
  BOSLUK (free-space) YONU: adaydan cikan isinlarin EN UZUN engelsiz gittigi yon.
    Tel giris kanali tanimi geregi en derin bosluktur.

TEZ DEGISMEZ: `v_o` turetmesi, 5 sinif, ~6000 remesh aynen kalir; bunlar SECENEK
uretir, secimi P3C/P5 yapar.
"""
import numpy as np

R_YEREL = 4.0        # aday cevresi yaricapi (mm) -- agiz konturu icin
N_ISIN = 42          # bosluk taramasi icin isin sayisi (Fibonacci kuresi)


def agiz_duzlem_normali(V, p, r=R_YEREL, min_tepe=8):
    """Adayin `r` yaricapindaki tepelere en iyi oturan duzlemin normali (PCA)."""
    d = np.linalg.norm(V - p, axis=1)
    Q = V[d <= r]
    if len(Q) < min_tepe:
        return None
    Q = Q - Q.mean(0)
    try:
        _u, _s, vt = np.linalg.svd(Q, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    return vt[2] / (np.linalg.norm(vt[2]) + 1e-12)      # en kucuk varyans yonu


def _fibonacci(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi),
                     np.cos(phi)], 1)


def bosluk_yonu(mesh, p, n=N_ISIN, max_mm=40.0):
    """Adaydan en UZUN engelsiz giden yon. Tel kanali en derin bosluktur."""
    D = _fibonacci(n)
    try:
        loc, idx, _tri = mesh.ray.intersects_location(
            np.repeat(p[None, :], n, 0), D, multiple_hits=False)
    except Exception:
        return None
    mesafe = np.full(n, max_mm, float)
    for L, i in zip(loc, idx):
        m = float(np.linalg.norm(L - p))
        if m < mesafe[i]:
            mesafe[i] = m
    j = int(np.argmax(mesafe))
    if mesafe[j] < 2.0:
        return None
    return D[j] / (np.linalg.norm(D[j]) + 1e-12)
