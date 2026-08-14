# -*- coding: utf-8 -*-
"""Turetme kayitlarini TEK yerden yukle -- feature genisligi TUTARLI olsun.

TRAP (2026-08-05, two times vurdu): `d5_birlestir.py` birlesik `_der_yeni.pkl`'i yazarken
X'i **58 -> 22 sutuna KIRPAR** (old measurement korpusu 22 bekliyor). Shard dosyalari
(`_der_yeni_0.pkl` ...) whereas 58 sutunla kalir.

`glob("results/_der_yeni*.pkl")` IKISINI DE getirir and sirali gezildiginde
`_der_yeni.pkl` ('.' < '_') SHARD'LARDAN ONCE gelir. `setdefault` first geleni tuttugu
for 22 sutunlu version kazanir. Sonra `M.shape[1]*2 == gate["n_feat"]` kontrolu
22*2=44 != 116 diye **HER parcayi sessizce eler** and F1 TAM 0.0000 cikar.

Belirti aldatici: "gate very kotu" like gorunur, oysa gate never calismamistir. Ilk
kurbanlar: `p1_gate_v5.py` (three arm da 0.0000, TABAN dahil).

COZUM: X22 with XR'nin BIRLESIMI already 58 sutundur and `X[:, 22:] == XR` oldugu
`d5_birlestir.py` inside 984/984 kayitta dogrulanmistir. Yani 22'lik kayit KAYIPSIZ
sekilde 58'e geri kurulabilir.
"""
import glob 
import io 
import json 
import os 
import pickle 

import numpy as np 

TAM_GENISLIK =58 


def x58 (r ):
    """Kaydin X'ini HER ZAMAN 58 column as dondur (otherwise None)."""
    X =r .get ("X")
    if X is None :
        return None 
    X =np .asarray (X ,float )
    if X .shape [1 ]==TAM_GENISLIK :
        return X 
    XR =r .get ("XR")
    if XR is None :
        return None 
    XR =np .asarray (XR ,float )
    if X .shape [1 ]+XR .shape [1 ]!=TAM_GENISLIK :
        return None 
    return np .hstack ([X ,XR ])


def yukle (pidler =None ,desen ="results/_der_yeni*.pkl"):
    """pid -> kayit. SHARD'LAR ONCE okunur; birlesik file only eksigi tamamlar.

    Sira onemli: shard'lar 58 sutunlu ORIJINALI carries. Yine de `x58()` each two
    genisligi de kaldirir, i.e. order a more silent hataya donusemez.
    """
    P =None if pidler is None else set (pidler )
    dosyalar =sorted (glob .glob (desen ))
    shard =[f for f in dosyalar if os .path .basename (f )!="_der_yeni.pkl"]
    birlesik =[f for f in dosyalar if os .path .basename (f )=="_der_yeni.pkl"]
    rec_ ={}
    for f in shard +birlesik :
        try :
            with open (f ,"rb")as h :
                for r in pickle .load (h ):
                    if P is not None and r ["pid"]not in P :
                        continue 
                    rec_ .setdefault (r ["pid"],r )
        except (OSError ,ValueError ,EOFError ,pickle .UnpicklingError ):
            continue 
    return rec_ 


def exam (yol ="results/d6_sinav_kumesi.json"):
    with io .open (yol ,encoding ="utf-8")as f :
        return json .load (f )
