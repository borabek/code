# -*- coding: utf-8 -*-
"""VII.2 -- ADET GEOMETRIDEN OKUNABILIR MI? (ogrenmesiz)

WHY IMPORTANT. Ust-k kahin deneyi adedin degerini olctu: UPUN'da only
correct CP sayisini bilmek F1'i 0.5359 -> 0.6892 does (**+0.1533**), SUPU'da
+0.0314. Ama that a KAHINDI. Soru: adedi OGRENMEDEN, parcanin own
geometrisinden okuyabilir miyiz?

HIPOTEZ (birlestirici ilkeden): a klemenste butun CP'ler AYNI bore capinda
becomes. O halde CP count ~ B-rep'teki KIPSEL YARICAPLI silindir count.

Bu probe birkac ogrenmesiz tahminciyi real adete karsi olcer:
  n_kipsel   : kipsel radius kovasindaki silindir count
  n_kipsel_y : same + axis yonu de kipsel yonle uyumlu olanlar
  n_hepsi    : tum silindirler (baseline cizgisi)

OLCU: median mutlak error, and |prediction - real| <= 1 orani (full isabete yakin).

MODEL YOK. D7'ye BAKILMAZ.
"""
import collections 
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,".")
import canonical_d7 as K # noqa: E402
import d6_record # noqa: E402

SIL =os .environ .get ("AG_SIL","results/_d6_silindirler.pkl")
MARKALAR =set (os .environ .get ("AG_MARKA","NIT,MOR,SUPU,UPUN").split (","))
R_TOL =float (os .environ .get ("AG_RTOL","0.15"))# kipsel radius bandi (mm)
A_TOL =float (os .environ .get ("AG_ATOL","12.0"))# kipsel direction bandi (derece)


def _kipsel_yaricap (R ):
    """En kalabalik radius kovasinin merkezi (0.1mm cozunurluk)."""
    if not len (R ):
        return None 
    kova =np .round (np .asarray (R ,float )/0.1 )*0.1 
    d =collections .Counter (kova )
    return max (d ,key =lambda k :(d [k ],-k ))


def tahminler (sil ):
    """Bir parcanin silindirlerinden ogrenmesiz count tahminleri."""
    if not sil :
        return {"n_hepsi":0 ,"n_kipsel":0 ,"n_kipsel_y":0 }
    R =np .array ([float (c ["radius"])for c in sil ])
    A =np .array ([np .asarray (c ["axis"],float )for c in sil ])
    A =A /np .maximum (np .linalg .norm (A ,axis =1 ,keepdims =True ),1e-12 )
    rk =_kipsel_yaricap (R )
    m_r =np .abs (R -rk )<=R_TOL 
    n_kipsel =int (m_r .sum ())
    # kipsel YON: most kalabalik axis yonu (unsigned)
    n_ky =n_kipsel 
    if n_kipsel >=2 :
        Ak =A [m_r ]
        cos =np .abs (Ak @Ak .T )
        aci =np .degrees (np .arccos (np .clip (cos ,-1 ,1 )))
        komsu =(aci <=A_TOL ).sum (1 )
        n_ky =int (komsu .max ())
    return {"n_hepsi":len (sil ),"n_kipsel":n_kipsel ,"n_kipsel_y":n_ky }


def main ():
    cy =pickle .load (open (SIL ,"rb"))
    kay =K .yukle ()
    kay .update ({str (p ):r for p ,r in d6_record .yukle ().items ()
    if str (p )not in kay })
    ist =collections .defaultdict (lambda :collections .defaultdict (list ))
    n =0 
    for pid ,r in kay .items ():
        if r .get ("mfg")not in MARKALAR :
            continue 
        G =r .get ("G",[])
        if not len (G ):
            continue 
        sil =cy .get (str (pid ))
        if not sil :
            continue 
        t =tahminler (sil )
        a =ist [r ["mfg"]]
        a ["gercek"].append (len (G ))
        for k ,v in t .items ():
            a [k ].append (v )
        n +=1 
    print (f"{n } part | kipsel yaricap bandi +-{R_TOL }mm, direction bandi {A_TOL }deg\n")

    yontemler =("n_hepsi","n_kipsel","n_kipsel_y")
    print (f"{'brand':<7}{'part':>6}{'gercek ort':>12}"+
    "".join (f"{y :>22}"for y in yontemler ))
    print (f"{'':<7}{'':<6}{'':<12}"+
    "".join (f"{'ort':>8}{'|error|':>7}{'<=1':>7}"for _ in yontemler ))
    out ={}
    for m_ in sorted (ist ,key =lambda x :-sum (ist [x ]["gercek"])):
        a =ist [m_ ]
        g =np .array (a ["gercek"],float )
        sat =f"{m_ :<7}{len (g ):>6}{g .mean ():>12.1f}"
        r ={"part":len (g ),"gercek_ort":float (g .mean ())}
        for y in yontemler :
            p =np .array (a [y ],float )
            error =np .abs (p -g )
            r [y ]={"ort":float (p .mean ()),
            "ortanca_hata":float (np .median (error )),
            "isabet_1":float ((error <=1 ).mean ())}
            sat +=(f"{p .mean ():>8.1f}{np .median (error ):>7.1f}"
            f"{(error <=1 ).mean ():>7.2f}")
        out [m_ ]=r 
        print (sat )
    json .dump ({"r_tol":R_TOL ,"a_tol":A_TOL ,"brand":out ,
    "not":"Ogrenmesiz adet tahmini: kipsel yaricapli silindir "
    "sayisi. |error| = ortanca mutlak error, <=1 = tam "
    "isabete yakin ratio. D7'ye BAKILMADI."},
    open ("results/adet_geometri.json","w"),indent =1 )
    print ("\nmakbuz -> results/adet_geometri.json")
    print ("OKUMA: <=1 orani yuksekse adet OGRENMEDEN okunabiliyor demektir;")
    print ("       ust-k deneyi bunun UPUN'da +0.1533 degerinde oldugunu gosterdi.")


if __name__ =="__main__":
    main ()
