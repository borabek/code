# -*- coding: utf-8 -*-
"""VII.4 -- URETILEN KONUMDA YON KURTARILABILIYOR MU?

VII.0m kafesin GT olmadan bulunabildigini showed (NIT kapsama 0.676). AMA
that measurement only KONUMU kontrol ediyordu (lateral 2mm / axial 40mm). Robot
metrigi also **ISARETLI ACI <= 10 derece** istiyor.

BU SONDA full kabul kutusunu uygular. Uretilen each izgara noktasinin
CEVRESINDEKI adaylarin direction-bankasi secenekleri toplanir and GT, however
(konum kutusu) VE (direction kutusu) birlikte saglanirsa kapsanmis sayilir.

Yon secimi KAHINDIR (mevcut options icinden EN IYISI). Yani this a TAVAN:
"uretilen konumda correct direction MEVCUT MU?" sorusunu yanitlar. Mevcut degilse
no selector onu bulamaz -- arm orada becomes. Mevcutsa is seciciye kalir.

WHY IMPORTANT: onceki lattice denemesi (K2.1) full here coktu -- yonu
KOPYALAMISTI and robot metrigi -0.0100 dusmustu (detection +0.0126 iken).

D7'ye BAKILMAZ.
"""
import collections 
import json 
import os 
import sys 
import time 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ .setdefault ("P6_DIZIN","results/_p6_oz_tam4")
sys .path .insert (0 ,".")
import canonical_d7 as K # noqa: E402
from run_p6_ortak import yukle # noqa: E402
from probe_lattice_v2 import kafes_ara # noqa: E402  (GT'SIZ arama)

KUME =os .environ .get ("KY_KUME","d6")
MARKALAR =set (os .environ .get ("KY_MARKA","NIT,MOR,SUPU,UPUN").split (","))
YANAL ,EKSENEL =2.0 ,40.0 
YAKIN_R =float (os .environ .get ("KY_YAKIN","2.0"))# izgara -> candidate yaricapi


def main ():
    t0 =time .time ()
    data_ =yukle (KUME ,int (os .environ .get ("P6_TR","0")))
    ist =collections .defaultdict (lambda :collections .defaultdict (list ))
    n =0 
    for d in data_ :
        if d ["mfg"]not in MARKALAR :
            continue 
        G =np .asarray (d ["G"],float )
        if len (G )<3 :
            continue 
        Gd =np .asarray (d ["Gd"],float )
        Gn =Gd /np .maximum (np .linalg .norm (Gd ,axis =1 ,keepdims =True ),1e-12 )
        P =np .asarray (d ["P"],float )# ADAY konumlari
        idx =np .asarray (d ["idx"],int )# option -> candidate
        YD =np .asarray (d ["YD"],float )# option yonleri
        Pu =np .unique (np .round (P ,3 ),axis =0 )
        found =kafes_ara (Pu )
        if not found :
            continue 
        uret =np .vstack ([b [2 ]for b in found ])

        # 1) YALNIZ KONUM (VII.0m with same olcu)
        v =uret [:,None ,:]-G [None ,:,:]
        al =(v *Gn [None ,:,:]).sum (-1 )
        yan =np .linalg .norm (v -al [...,None ]*Gn [None ,:,:],axis =-1 )
        konum_ok =(yan <=YANAL )&(np .abs (al )<=EKSENEL )# (uret, gt)

        # 2) KONUM + YON: izgara noktasinin cevresindeki adaylarin secenekleri
        d_ua =np .linalg .norm (uret [:,None ,:]-P [None ,:,:],axis =-1 )
        yakin_aday =d_ua <=YAKIN_R # (uret, candidate)
        tam_ok =np .zeros (len (G ),bool )
        for j in range (len (G )):
            u_ler =np .where (konum_ok [:,j ])[0 ]
            if not len (u_ler ):
                continue 
            candidates =np .where (yakin_aday [u_ler ].any (0 ))[0 ]
            if not len (candidates ):
                continue 
            sec =np .isin (idx ,candidates )
            if not sec .any ():
                continue 
            cos =YD [sec ]@Gn [j ]
            aci =np .degrees (np .arccos (np .clip (cos ,-1 ,1 )))# ISARETLI
            if (aci <=K .ACI ).any ():
                tam_ok [j ]=True 

        a =ist [d ["mfg"]]
        a ["gt"].append (len (G ))
        a ["konum"].append (int (konum_ok .any (0 ).sum ()))
        a ["tam"].append (int (tam_ok .sum ()))
        a ["uret"].append (len (uret ))
        n +=1 
        if n %30 ==0 :
            print (f"  {n } part ({time .time ()-t0 :.0f} s)",flush =True )

    print (f"\n{'brand':<7}{'GT':>7}{'uret/p':>8}{'KONUM':>9}{'KONUM+YON':>11}"
    f"{'direction kaybi':>11}")
    out ={}
    for m_ in sorted (ist ,key =lambda x :-sum (ist [x ]["gt"])):
        a =ist [m_ ]
        g =sum (a ["gt"])
        k =sum (a ["konum"])/max (g ,1 )
        t =sum (a ["tam"])/max (g ,1 )
        out [m_ ]={"gt":g ,"konum":k ,"konum_yon":t ,"yon_kaybi":k -t ,
        "uret_parca":float (np .mean (a ["uret"]))}
        print (f"{m_ :<7}{g :>7}{np .mean (a ['uret']):>8.0f}{k :>9.3f}{t :>11.3f}"
        f"{k -t :>11.3f}")
    json .dump ({"lateral":YANAL ,"axial":EKSENEL ,"aci":K .ACI ,
    "yakin_r":YAKIN_R ,"brand":out ,
    "not":"Uretilen izgara noktalarinda YON, cevredeki adaylarin "
    "direction-bankasi seceneklerinden KAHIN gibi secilir. TAVAN "
    "olcumudur: correct direction MEVCUT MU? D7'ye BAKILMADI."},
    open (f"results/kafes_yon_{KUME }.json","w"),indent =1 )
    print (f"\nmakbuz -> results/kafes_yon_{KUME }.json")
    print ("OKUMA: direction kaybi KUCUKSE arm canli (direction uretilen konumda mevcut);")
    print ("       BUYUKSE K2.1'in coktugu yere geri donduk demektir.")


if __name__ =="__main__":
    main ()
