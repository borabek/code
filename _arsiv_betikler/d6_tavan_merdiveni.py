# -*- coding: utf-8 -*-
"""D6-MERDIVEN: robot 0.70 planini TAHMINE not OLCUME dayandir.

Her kademe "this arm MUKEMMEL calissaydi" sorusunun yanitidir. Kademeler arasi FARK,
that kolun UST SINIR odulüdür. Bir kola yatirim, however farki buyukse anlamlidir.

  0 GERCEK                 : urunun bugunku hali
  1 +ISARET                : direction isareti hep correct
  2 +ACI                   : angle kisiti kalkti (lateral<=2 kalir)
  3 +YANAL                 : lateral kisiti gevsek (angle<=10 kalir)
  4 +IKISI (= TESPIT)      : direction and lateral mukemmel -> robot == tespit
  5 +GATE (temsil tavani)  : mevcut adaylarla mukemmel gate + mukemmel poz
  6 +ADAY (mutlak ceiling)   : each GT for candidate VAR + mukemmel gate + mukemmel poz = 1.0

REJIM AYRIMI SART ([[two-regimes-low-vs-high-cp]]): duz mean ASLA verilmez.
"""
import collections 
import glob 
import io 
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

KUME ="results/d6_sinav_kumesi.json"
MAKBUZ ="results/d6_tavan_merdiveni.json"


def main ():
    import wire_gate 
    from sina_cluster import match_greedy ,f1w 

    sv =json .load (io .open (KUME ,encoding ="utf-8"))
    PID =set (sv ["pidler"])
    import d6_record 
    kayit =d6_record .yukle (PID )
    with open ("results/wire_gate.pkl","rb")as f :
        gate =pickle .load (f )

    AD =["0 GERCEK","1 +ISARET","2 +ACI","3 +YANAL","4 +IKISI(=TESPIT)",
    "5 +GATE(temsil)","6 +ADAY(mutlak)"]
    S ={a :[]for a in AD }
    rejim_s ={a :collections .defaultdict (list )for a in AD }

    for pid ,r in kayit .items ():
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        rj ="cok"if r ["n"]>=8 else "dusuk"
        diag =r ["diag"];tt =max (3.0 ,0.06 *diag )
        P0 =np .asarray (r ["P"],float )if r .get ("P")is not None else np .zeros ((0 ,3 ))
        D0 =np .asarray (r ["Pd"],float )if r .get ("Pd")is not None else np .zeros ((0 ,3 ))
        P =np .zeros ((0 ,3 ));D =np .zeros ((0 ,3 ))
        if r .get ("X")is not None and len (P0 ):
            M =np .asarray (r ["X"],float )
            if M .shape [1 ]*2 ==gate ["n_feat"]:
                k =wire_gate .decision_mask (wire_gate .decision_score (gate ,M ))
                if k .any ():
                    P =P0 [k ];D =D0 [k ]

        def ekle (ad ,satir ):
            S [ad ].append (satir );rejim_s [ad ][rj ].append (satir )

            # 0 GERCEK
        ekle (AD [0 ],(rj ,)+match_greedy (P ,D ,G ,Gd ,diag ,2.0 ,10.0 ,False ,
        signed =True )[:3 ])
        # 1 +ISARET: eslesen ciftte sign GT'ye according to duzeltilmis
        Dk =D .copy ()
        if len (P )and len (G ):
            _t ,_f ,_n ,bi =match_greedy (P ,D ,G ,Gd ,diag ,0.0 ,180.0 ,True )
            for (pi ,gi ,*_x )in bi ["eslesme"]:
                if float (D [pi ]@Gd [gi ])<0 :
                    Dk [pi ]=-D [pi ]
        ekle (AD [1 ],(rj ,)+match_greedy (P ,Dk ,G ,Gd ,diag ,2.0 ,10.0 ,False ,
        signed =True )[:3 ])
        # 2 +ACI: angle serbest, lateral 2mm
        ekle (AD [2 ],(rj ,)+match_greedy (P ,D ,G ,Gd ,diag ,2.0 ,180.0 ,False )[:3 ])
        # 3 +YANAL: lateral tespit toleransi, angle<=10 ISARETLI
        ekle (AD [3 ],(rj ,)+match_greedy (P ,D ,G ,Gd ,diag ,0.0 ,10.0 ,True ,
        signed =True )[:3 ])
        # 4 +IKISI = tespit
        ekle (AD [4 ],(rj ,)+match_greedy (P ,D ,G ,Gd ,diag ,0.0 ,180.0 ,True )[:3 ])
        # 5 +GATE: gate ONCESI adaylarla, mukemmel gate (FP=0)
        if len (P0 )and len (G ):
            d =P0 [:,None ,:]-G [None ,:,:]
            al =(d *Gd [None ,:,:]).sum (-1 )
            pe =np .linalg .norm (d -al [...,None ]*Gd [None ,:,:],axis =-1 )
            pe =np .where (np .abs (al )<=40.0 ,pe ,np .inf )
            kg =int ((pe .min (0 )<=tt ).sum ())
        else :
            kg =0 
        ekle (AD [5 ],(rj ,kg ,0 ,len (G )-kg ))
        # 6 mutlak
        ekle (AD [6 ],(rj ,len (G ),0 ,0 ))

    print (f"TEMIZ SINAV: {len (kayit )} part, {sum (len (np .asarray (r ['G']))for r in kayit .values ())} GT CP\n")
    print (f"{'kademe':<20}{'AGIRLIKLI':>11}{'dusuk-CP':>11}{'cok-CP':>10}{'kazanc':>9}")
    onc =None 
    sonuc ={}
    for a in AD :
        v =f1w (S [a ])
        dl =f1w (rejim_s [a ]["dusuk"])if rejim_s [a ]["dusuk"]else float ("nan")
        ck =f1w (rejim_s [a ]["cok"])if rejim_s [a ]["cok"]else float ("nan")
        kz =""if onc is None else f"{v -onc :+.4f}"
        print (f"{a :<20}{v :>11.4f}{dl :>11.4f}{ck :>10.4f}{kz :>9}")
        sonuc [a ]={"agirlikli":v ,"dusuk":dl ,"cok":ck }
        onc =v 
    json .dump (sonuc ,io .open (MAKBUZ ,"w",encoding ="utf-8"),indent =1 )
    print (f"\nmakbuz -> {MAKBUZ }")


if __name__ =="__main__":
    main ()
