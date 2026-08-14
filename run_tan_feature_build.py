# -*- coding: utf-8 -*-
"""Agiz tanimlayicilarini `_tam_oz` HAVUZUYLA HIZALI uret.

WHY YENIDEN: `results/_agiz_tan` only `brep_adaylari` ciktisi for
uretilmisti; `merged_pool` segmentasyon adayina yakin B-rep onerilerini
ELEDIGI for siralamalar ORTUSMUYOR. Hizalanmamis feature sessizce wrong
satira baglanir -- this projede most pahali error turu.

Burada tanimlayicilar dogrudan `_tam_oz`'daki (P, D) uzerinden is computed:
each candidate for EN YAKIN silindir (eksene dik mesafeyle) ebeveyn sayilir, otherwise
radius/depth 0 kalir and only isin olculeri dolar.

Ayrica DAHA ONCE OLCULEN AUC'lar BOZUK ETIKETLE alinmisti; this arm duzeltilmis
etiketle (lateral + axial 40 + acgozlu bire-a) yeniden sinanacak.
"""
import json 
import os 
import pickle 
import sys 
import time 

import numpy as np 
import trimesh 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,".")
import mouth_descriptor as AT # noqa: E402
import d6_record # noqa: E402
import canonical_d7 as K # noqa: E402

OZ ="results/_tam_oz"
CIK ="results/_tan_hizali"
OB ={"d7":"results/_p1_olasilik_d7","d6":"results/_p1_olasilik_g7",
"tam":"results/_p1_olasilik_brepegit"}
CYL ={"d7":"results/_d7_silindirler.pkl","d6":"results/_d6_silindirler.pkl",
"tam":"results/_brepegit_silindirler.pkl"}


def ebeveyn (P ,cyl ):
    """Her candidate for most yakin silindirin (centre, axis, radius) kaydi."""
    if not cyl :
        return [{}for _ in range (len (P ))]
    C ,A ,R ,MA ,MB =[],[],[],[],[]
    for c in cyl :
        a =np .asarray (c ["axis"],float )
        n =np .linalg .norm (a )
        if n <1e-9 :
            continue 
        C .append (np .asarray (c ["center"],float ))
        A .append (a /n )
        R .append (float (c .get ("radius",0.0 )))
        MA .append (c .get ("mouth_a"))
        MB .append (c .get ("mouth_b"))
    if not C :
        return [{}for _ in range (len (P ))]
    C =np .asarray (C );A =np .asarray (A );R =np .asarray (R )
    out =[]
    for p in P :
        w =p [None ]-C 
        e =np .einsum ("ij,ij->i",w ,A )
        d =np .linalg .norm (w -e [:,None ]*A ,axis =1 )
        i =int (np .argmin (d ))
        m ={"radius":float (R [i ])}
        if MA [i ]is not None and MB [i ]is not None :
            m ["mouth_a"]=MA [i ]
            m ["mouth_b"]=MB [i ]
        out .append (m )
    return out 


def main ():
    os .makedirs (CIK ,exist_ok =True )
    for ad in ("d7","d6","tam"):
        cy =pickle .load (open (CYL [ad ],"rb"))
        if ad =="d6":
            kay =d6_record .yukle (set (d6_record .exam ()["pidler"]))
        elif ad =="d7":
            kay =K .yukle (json .load (
            open ("results/d7_sinav_kumesi.json"))["pidler"])
        else :
            kay =K .yukle ([str (p )for p in json .load (
            open ("results/brep_egitim_kumesi.json"))["pidler"]])
        t0 =time .time ()
        n =atlanan =0 
        for pid ,r in sorted (kay .items ()):
            pid =str (pid )
            src_ =f"{OZ }/{ad }_{pid }.npz"
            hedef =f"{CIK }/{ad }_{pid }.npz"
            if os .path .exists (hedef )or not os .path .exists (src_ ):
                atlanan +=1 
                continue 
            f =f"{OB [ad ]}/{pid }.npz"
            if not os .path .exists (f ):
                atlanan +=1 
                continue 
            z =np .load (src_ )
            P =np .asarray (z ["P"],float )
            D =np .asarray (z ["D"],float )
            zz =np .load (f )
            mesh =trimesh .Trimesh (np .asarray (zz ["V"],float ),
            np .asarray (zz ["F"],np .int64 ),process =False )
            T =AT .tanimla (P ,D ,ebeveyn (P ,cy .get (pid )),mesh ,
            float (r ["diag"]))
            if len (T )!=len (P ):
                raise SystemExit (f"{ad }_{pid }: tanimlayici {len (T )} vs candidate "
                f"{len (P )} -- HIZALAMA BOZUK, duruyorum")
            np .savez_compressed (hedef ,T =T .astype (np .float32 ))
            n +=1 
            if n %200 ==0 :
                print (f"  {ad } {n } yazildi {(time .time ()-t0 )/n :.2f}s/part",
                flush =True )
        print (f"{ad } BITTI: {n } yazildi | atlanan {atlanan }",flush =True )
    print ("->",CIK )


if __name__ =="__main__":
    main ()
