# -*- coding: utf-8 -*-
"""KALDIRAC SONDASI: kuresel seviye olu, YEREL karsitlik yasiyor mu?

OTOPSI BULGUSU (results/autopsy_segmentation.json):
  NIT'te GT'de probability 0.5244, RASTGELE yuzeyde 0.4394 -> ayrim 1.19x
  SUPU'da 0.4815 / 0.0058 -> 83x
Yani dense parcada segmentasyon TUM govdeyi boyuyor. Zincirin most basindaki
most guclu feature orada bilgi tasimiyor; asagidaki five mekanizmanin all of them
that skoru tuketiyordu.

HIPOTEZ. 0.44 zemin on 0.52 vertex KURESEL siralamada kaybolur but
YEREL komsulukta still tepedir. Oyleyse olasiligin KENDISI not, yerel
KARSITLIGI ayirt edici may be.

OLCULEN (GT'de vs RASTGELE yuzeyde, R = 2/5/10 mm toplarda):
  ham        : p
  difference       : p - median(p, R topu)
  yuzdelik   : p'nin R topundaki yuzdelik order
  vertex       : p, R topundaki most high mi (0/1)
Her biri for AYRIM = median(GT) - median(rastgele), and AUC.
AUC, olcegi ne olursa olsun kiyaslanabilir single sayidir.

KAPI: NIT'te herhangi a yerel olcunun AUC'si ham AUC'yi >= 0.05 asarsa
kaldirac CANLI -> EK blogu yazilir. Asmazsa yazilmaz.

D7'ye BAKILMAZ.
"""
import collections 
import json 
import os 
import sys 

import numpy as np 
from scipy .spatial import cKDTree 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,".")
import connector3d # noqa: E402
import canonical_d7 as K # noqa: E402
import d6_record # noqa: E402

MESH =os .environ .get ("YK_MESH","results/_p1_olasilik")
MARKALAR =set (os .environ .get ("YK_MARKA","NIT,MOR,SUPU,UPUN").split (","))
YARICAP =[float (x )for x in os .environ .get ("YK_R","2,5,10").split (",")]
CE =int (connector3d .CABLE_ENTRY )
CT =int (connector3d .CONTACT )
N_RAST =400 


def yerel (p ,V ,agac ,R ):
    """each vertex for (difference, yuzdelik, tepe_mi) -- R yaricapli topta."""
    kom =agac .query_ball_point (V ,R )
    f =np .zeros (len (V ))
    y =np .zeros (len (V ))
    t =np .zeros (len (V ))
    for i ,kk in enumerate (kom ):
        if len (kk )<3 :
            y [i ]=0.5 
            continue 
        q =p [kk ]
        f [i ]=p [i ]-np .median (q )
        y [i ]=float ((q <p [i ]).mean ())
        t [i ]=float (p [i ]>=q .max ()-1e-12 )
    return f ,y ,t 


def auc (poz ,neg ):
    """Mann-Whitney AUC -- olcekten bagimsiz ayirt edicilik."""
    if not len (poz )or not len (neg ):
        return 0.5 
    h =np .concatenate ([poz ,neg ])
    r =np .argsort (np .argsort (h ))+1.0 
    return float ((r [:len (poz )].sum ()-len (poz )*(len (poz )+1 )/2.0 )
    /(len (poz )*len (neg )))


def main ():
    kay =K .yukle ()
    kay .update ({str (p ):r for p ,r in d6_record .yukle ().items ()
    if str (p )not in kay })
    ad =["ham"]+[f"{k }_{int (R )}"for R in YARICAP 
    for k in ("diff","yuzdelik","tepe")]
    ist =collections .defaultdict (lambda :collections .defaultdict (list ))
    n =0 
    for pid ,r in kay .items ():
        if r .get ("mfg")not in MARKALAR :
            continue 
        G =np .asarray (r .get ("G",[]),float )
        mf =f"{MESH }/{pid }.npz"
        if not len (G )or not os .path .exists (mf ):
            continue 
        z =np .load (mf )
        V =np .ascontiguousarray (z ["V"],np .float64 )
        p =np .mean ([np .asarray (q ,float )for q in z ["pbs"]],axis =0 )
        p =p [:,CE ]+p [:,CT ]
        agac =cKDTree (V )
        gi =agac .query (G )[1 ]# GT'ye most yakin vertices
        rng =np .random .default_rng (0 )
        # RASTGELE baseline: GT'den >= 4mm uzak vertices (komsulari sayma)
        uzak =np .where (cKDTree (G ).query (V )[0 ]>=4.0 )[0 ]
        if len (uzak )<50 :
            continue 
        ri =rng .choice (uzak ,min (N_RAST ,len (uzak )),replace =False )
        sut ={"ham":p }
        for R in YARICAP :
            f ,y ,t =yerel (p ,V ,agac ,R )
            sut [f"fark_{int (R )}"]=f 
            sut [f"yuzdelik_{int (R )}"]=y 
            sut [f"tepe_{int (R )}"]=t 
        a =ist [r ["mfg"]]
        a ["gt"].append (len (G ))
        for k_ in ad :
            a [k_ ].append (auc (sut [k_ ][gi ],sut [k_ ][ri ]))
        n +=1 
        if n %50 ==0 :
            print (f"  {n } part",flush =True )

    print (f"\n{n } part | AUC: GT tepeleri vs GT'den >=4mm uzak tepeler")
    print (f"{'brand':<7}{'GT':>7}"+"".join (f"{k :>14}"for k in ad ))
    out ={}
    for m_ in sorted (ist ,key =lambda x :-sum (ist [x ]["gt"])):
        a =ist [m_ ]
        r ={k :float (np .mean (a [k ]))for k in ad }
        r ["gt"]=sum (a ["gt"])
        out [m_ ]=r 
        print (f"{m_ :<7}{r ['gt']:>7}"+"".join (f"{r [k ]:>14.4f}"for k in ad ))
    print ("\n=== HAM'A GORE (dense brand NIT) ===")
    if "NIT"in out :
        h =out ["NIT"]["ham"]
        en ,ei =h ,"ham"
        for k_ in ad [1 :]:
            f =out ["NIT"][k_ ]-h 
            print (f"  {k_ :<14}{out ['NIT'][k_ ]:.4f}   {f :+.4f}"
            +("  <- KAPI GECTI"if f >=0.05 else ""))
            if out ["NIT"][k_ ]>en :
                en ,ei =out ["NIT"][k_ ],k_ 
        print (f"\n  EN IYI: {ei } = {en :.4f} (ham {h :.4f})")
    json .dump ({"brand":out ,"yaricap":YARICAP ,
    "not":"Segmentasyon olasiliginin YEREL karsitligi. AUC = GT "
    "tepeleri vs >=4mm uzak tepeler. D7'ye BAKILMADI."},
    open ("results/local_contrast.json","w"),indent =1 )
    print ("receipt -> results/local_contrast.json")


if __name__ =="__main__":
    main ()
