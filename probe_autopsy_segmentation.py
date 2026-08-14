# -*- coding: utf-8 -*-
"""OTOPSI: dense part duvarinin KOKENI segmentasyonda mi?

BUGUN BES MEKANIZMA DENENDI, BESI DE ZINCIRIN SONUNDAYDI:
  baglam modeli (S4) · lattice yayilimi · analitik direction · mesh normali · count
Hicbiri EN BASA, SEGMENTASYONA bakmadi.

HIPOTEZ. NIT'te 24 CP'nin only BIRKACINDA segmentasyon ateşliyor;
gerisi mesh seyreltmesinden geliyor and ZAYIF feature tasiyor. Bu, olculen
deseni birebir aciklar:
  * ILK correct secenek 18. sirada, SONUNCUSU 1065. (model ilkini buluyor)
  * poz-neg skor ayrimi 0.050 (skor bilgi tasimiyor)
  * havuzda konum VAR (0.843) but secilemiyor

OLCULEN: each GT'nin konumunda segmentasyon olasiligi (CE+CT) nedir?
  p_gt_ortanca : GT noktalarindaki olasiligin ortancasi
  p_gt_en_dusuk: parcadaki EN DUSUK GT olasiligi (most hard CP)
  p_gt>0.5     : olasiligi 0.5 ustunde which is GT orani
  p_gt>0.1     : 0.1 ustunde which is GT orani
  p_rastgele   : rastgele surface noktalarinda median (baseline cizgisi)

Eger NIT'te p_gt>0.5 low but p_gt>0.1 yuksekse: segmentasyon ZAYIF but
VAR -> threshold/duyarlilik kolu acilir.
Eger p_gt>0.1 de dusukse: segmentasyon that CP'leri HIC GORMUYOR -> darbogaz
most basta, and asagidaki no numara kurtaramaz.

D7'ye BAKILMAZ.
"""
import collections 
import json 
import os 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,".")
import connector3d # noqa: E402
import canonical_d7 as K # noqa: E402
import d6_record # noqa: E402

MESH =os .environ .get ("OS_MESH","results/_p1_olasilik")
MARKALAR =set (os .environ .get ("OS_MARKA","NIT,MOR,SUPU,UPUN").split (","))
CE =int (connector3d .CABLE_ENTRY )
CT =int (connector3d .CONTACT )


def main ():
    kay =K .yukle ()
    kay .update ({str (p ):r for p ,r in d6_record .yukle ().items ()
    if str (p )not in kay })
    ist =collections .defaultdict (lambda :collections .defaultdict (list ))
    n =0 
    for pid ,r in kay .items ():
        if r .get ("mfg")not in MARKALAR :
            continue 
        G =np .asarray (r .get ("G",[]),float )
        if not len (G ):
            continue 
        mf =f"{MESH }/{pid }.npz"
        if not os .path .exists (mf ):
            continue 
        z =np .load (mf )
        V =np .ascontiguousarray (z ["V"],np .float64 )
        pb =np .mean ([np .asarray (q ,float )for q in z ["pbs"]],axis =0 )
        p_ce =pb [:,CE ]+pb [:,CT ]
        # each GT'ye EN YAKIN tepenin olasiligi
        yak =np .argmin (np .linalg .norm (G [:,None ,:]-V [None ,:,:],
        axis =-1 ),axis =1 )
        pg =p_ce [yak ]
        a =ist [r ["mfg"]]
        a ["gt"].append (len (G ))
        a ["med"].append (float (np .median (pg )))
        a ["min"].append (float (pg .min ()))
        a ["ust50"].append (float ((pg >=0.5 ).mean ()))
        a ["ust10"].append (float ((pg >=0.1 ).mean ()))
        a ["ust01"].append (float ((pg >=0.01 ).mean ()))
        rng =np .random .default_rng (0 )
        a ["rast"].append (float (np .median (
        p_ce [rng .choice (len (V ),min (len (V ),500 ),replace =False )])))
        n +=1 
    print (f"{n } part | olasilik = CE + CT (segmentasyon)\n")
    print (f"{'brand':<7}{'GT':>7}{'p_gt ortanca':>14}{'p_gt en dusuk':>15}"
    f"{'>0.5':>8}{'>0.1':>8}{'>0.01':>8}{'rastgele':>10}")
    out ={}
    for m_ in sorted (ist ,key =lambda x :-sum (ist [x ]["gt"])):
        a =ist [m_ ]
        r ={"gt":sum (a ["gt"]),
        "p_ortanca":float (np .median (a ["med"])),
        "p_en_dusuk":float (np .median (a ["min"])),
        "ust50":float (np .mean (a ["ust50"])),
        "ust10":float (np .mean (a ["ust10"])),
        "ust01":float (np .mean (a ["ust01"])),
        "rastgele":float (np .median (a ["rast"]))}
        out [m_ ]=r 
        print (f"{m_ :<7}{r ['gt']:>7}{r ['p_ortanca']:>14.4f}"
        f"{r ['p_en_dusuk']:>15.4f}{r ['ust50']:>8.3f}{r ['ust10']:>8.3f}"
        f"{r ['ust01']:>8.3f}{r ['rastgele']:>10.4f}")
    json .dump ({"brand":out ,
    "not":"GT konumundaki segmentasyon olasiligi (CE+CT). "
    "Zincirin EN BASI. D7'ye BAKILMADI."},
    open ("results/otopsi_segmentasyon.json","w"),indent =1 )
    print ("\nmakbuz -> results/otopsi_segmentasyon.json")
    print ("OKUMA:")
    print ("  >0.5 dusuk ama >0.1 yuksek -> segmentasyon ZAYIF ama VAR")
    print ("     (threshold/duyarlilik kolu acilir)")
    print ("  >0.1 de dusuk -> segmentasyon o CP'leri HIC GORMUYOR;")
    print ("     darbogaz EN BASTA ve asagidaki hicbir numara kurtaramaz")


if __name__ =="__main__":
    main ()
