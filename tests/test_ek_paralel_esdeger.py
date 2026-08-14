# -*- coding: utf-8 -*-
"""Paralel ek-blok yolu SERI yolla BIT-AYNI mi?

Windows `spawn` cocuk surecte __main__'i YENIDEN ICE AKTARIR; that is why test
stdin'den not GERCEK DOSYADAN kosmali (stdin heredoc'ta asiliyor).
"""
import os 
import sys 
import time 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ .setdefault ("P6_DIZIN","results/_p6_oz_u25")
sys .path .insert (0 ,".")

import numpy as np # noqa: E402
from run_p6_ortak import yukle # noqa: E402


def kos (blok ,n ,isci ):
    os .environ ["EK_BLOK"]=blok 
    import importlib 

    import run_extra_feature as EK 
    importlib .reload (EK )
    EK .ISCI =isci 
    data_ =yukle ("d6",n )
    for d in data_ :
        d ["_kume"]="d6"
    oof =[np .full (len (d ["idx"]),0.5 )for d in data_ ]
    t =time .time ()
    out =EK .ek_hepsi (data_ ,oof )
    return np .vstack (out ),time .time ()-t 


def main ():
    n =int (os .environ .get ("T_N","4"))
    isci =int (os .environ .get ("T_ISCI","3"))
    for blok in os .environ .get ("T_BLOK","simetri,depth").split (","):
        A ,ta =kos (blok ,n ,1 )
        B ,tb =kos (blok ,n ,isci )
        d =float (np .abs (np .nan_to_num (A )-np .nan_to_num (B )).max ())
        print (f"{blok :<10} sekil {A .shape }  seri {ta :6.1f}s  "
        f"paralel({isci }) {tb :6.1f}s  hizlanma {ta /max (tb ,1e-9 ):.2f}x  "
        f"MAKS FARK {d :.3g}  {'AYNI'if d ==0 else 'FARKLI!'}",
        flush =True )


if __name__ =="__main__":
    main ()
