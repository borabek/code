# -*- coding: utf-8 -*-
"""P0-4: TUM TAVANLARI TEK and OPTIMAL eslestiriciyle yeniden hesapla.

Onceki merdiven acgozlu eslestiriciyle cikarilmisti. Acgozlu, kalabalik parcada a
tahmini wrong GT'ye baglayip digerini empty birakabilir; i.e. "KALABALIK" kovasinin
(GT'nin %12.8'i) a kismi GERCEK bilgi eksigi not ATAMA artefaktidir. Macar
yontemi same kabul kutusu inside TP'yi enbuyukler, therefore two sayinin FARKI
dogrudan "saf atama kaybi"ni gives.

KADEMELER (all of them AYNI kayitlar, AYNI kabul kutusu):
  0 GERCEK            urunun bugunku hali (gate v5 + threshold + selector)
  1 +ISARET           direction isareti hep correct
  2 +ACI              angle kisiti kalkti (lateral 2mm kalir)
  3 +YANAL            lateral kisiti gevsek (angle<=10 kalir)
  4 +IKISI (=TESPIT)  poz mukemmel -> robot == tespit
  5 +ATAMA            tespit, mukemmel atama with (Macar) -- KALABALIK kovasinin tavani
  6 +GATE             mevcut adaylarla mukemmel gate
  7 +ADAY             each GT for candidate present = 1.0
"""
import collections 
import io 
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

import d6_record 

MAKBUZ ="results/p0_tavan_yeniden.json"


def kod_muhru ():
    """P0-6: config + model + kod hash'i. Sonuc hangi durumdan output, geri izlenebilsin."""
    import hashlib 
    h ={}
    for f in ("cp_config.json","results/wire_gate_v5.pkl","results/p3c_axis_selector.pkl",
    "sina_cluster.py","wire_gate.py","cp_openings.py","robot_cp.py"):
        if os .path .exists (f ):
            with open (f ,"rb")as fh :
                h [os .path .basename (f )]=hashlib .sha256 (fh .read ()).hexdigest ()[:12 ]
    return h 


def main ():
    import protocol 
    protocol .tez_dogrula ()
    import p3c_axis_selector as P3C 
    from sina_cluster import match_greedy ,match_hungarian ,f1w 

    sv =d6_record .exam ()
    rec_ =d6_record .yukle (set (sv ["pidler"]))
    with open ("results/wire_gate_v5.pkl","rb")as f :
        gate =pickle .load (f )
    with open ("results/_d6_silindirler.pkl","rb")as f :
        ob =pickle .load (f )
    with open ("results/p3c_axis_selector.pkl","rb")as f :
        sec =pickle .load (f )["clf"]

    AD =["0 GERCEK","1 +ISARET","2 +ACI","3 +YANAL","4 +IKISI(=TESPIT)",
    "5 +ATAMA(macar)","6 +GATE","7 +ADAY(mutlak)"]
    S ={a :[]for a in AD }
    rj_s ={a :collections .defaultdict (list )for a in AD }
    kova =collections .Counter ()

    for pid ,r in rec_ .items ():
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        rj ="cok"if r ["n"]>=8 else "dusuk"
        diag =r ["diag"];tt =max (3.0 ,0.06 *diag )
        P0 =np .asarray (r ["P"],float )if r .get ("P")is not None else np .zeros ((0 ,3 ))
        # URUNUN BUGUNKU HALI: gate v5 + threshold 0.40/0.30 + ogrenilmis axis selector
        P =np .zeros ((0 ,3 ));D =np .zeros ((0 ,3 ))
        pak =P3C .parca_adaylari (r ,gate ,ob )
        if pak is not None :
            P ,D ,Sk ,komsu =pak 
            cy =ob .get (pid )or []
            P2 =P .copy ();D2 =D .copy ()
            for i in range (len (P )):
                opt =P3C .secenekler (cy ,P [i ],D [i ],diag ,float (Sk [i ]),komsu ,len (P ))
                if len (opt )==1 :
                    continue 
                sk =sec .predict_proba (np .asarray ([o [2 ]for o in opt ],float ))[:,1 ]
                j =int (np .argmax (sk ))
                if j !=0 :
                    P2 [i ]=opt [j ][0 ];D2 [i ]=opt [j ][1 ]
            P ,D =P2 ,D2 

        def ek (ad ,line_ ):
            S [ad ].append (line_ );rj_s [ad ][rj ].append (line_ )

        ek (AD [0 ],(rj ,)+match_hungarian (P ,D ,G ,Gd ,diag ,2.0 ,10.0 ,False ,
        signed =True )[:3 ])
        Dk =D .copy ()
        if len (P )and len (G ):
            _t ,_f ,_n ,bi =match_hungarian (P ,D ,G ,Gd ,diag ,0.0 ,180.0 ,True )
            for (pi ,gi ,*_x )in bi ["eslesme"]:
                if float (D [pi ]@Gd [gi ])<0 :
                    Dk [pi ]=-D [pi ]
        ek (AD [1 ],(rj ,)+match_hungarian (P ,Dk ,G ,Gd ,diag ,2.0 ,10.0 ,False ,
        signed =True )[:3 ])
        ek (AD [2 ],(rj ,)+match_hungarian (P ,D ,G ,Gd ,diag ,2.0 ,180.0 ,False )[:3 ])
        ek (AD [3 ],(rj ,)+match_hungarian (P ,D ,G ,Gd ,diag ,0.0 ,10.0 ,True ,
        signed =True )[:3 ])
        tsp =match_hungarian (P ,D ,G ,Gd ,diag ,0.0 ,180.0 ,True )
        ek (AD [4 ],(rj ,)+tsp [:3 ])
        ek (AD [5 ],(rj ,)+tsp [:3 ])# same: Macar already optimal atama
        # KOVA: GT duzeyi loss nedeni (Macar with)
        es_gt ={e [1 ]for e in tsp [3 ]["eslesme"]}

        def kapsanan (Pq ):
            if not len (Pq )or not len (G ):
                return np .zeros (len (G ),bool )
            dd =Pq [:,None ,:]-G [None ,:,:]
            al =(dd *Gd [None ,:,:]).sum (-1 )
            pe =np .linalg .norm (dd -al [...,None ]*Gd [None ,:,:],axis =-1 )
            return np .where (np .abs (al )<=40.0 ,pe ,np .inf ).min (0 )<=tt 
        on =kapsanan (P0 );last_ =kapsanan (P )
        for gi in range (len (G )):
            if gi in es_gt :
                kova ["ESLESTI"]+=1 
            elif not on [gi ]:
                kova ["ADAY_YOK"]+=1 
            elif not last_ [gi ]:
                kova ["GATE_REDDI"]+=1 
            else :
                kova ["KALABALIK"]+=1 
        kg =int (on .sum ())
        ek (AD [6 ],(rj ,kg ,0 ,len (G )-kg ))
        ek (AD [7 ],(rj ,len (G ),0 ,0 ))

        # --- acgozlu with difference (only tespit and robot for)
    Ta ,Ra =[],[]
    for pid ,r in rec_ .items ():
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        rj ="cok"if r ["n"]>=8 else "dusuk"
        P =np .zeros ((0 ,3 ));D =np .zeros ((0 ,3 ))
        pak =P3C .parca_adaylari (r ,gate ,ob )
        if pak is not None :
            P ,D =pak [0 ],pak [1 ]
        Ta .append ((rj ,)+match_greedy (P ,D ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True )[:3 ])
        Ra .append ((rj ,)+match_greedy (P ,D ,G ,Gd ,r ["diag"],2.0 ,10.0 ,False ,
        signed =True )[:3 ])

    n_gt =sum (kova .values ())
    print (f"TEMIZ SINAV {len (rec_ )} part, {n_gt } GT CP  (eslestirici: MACAR)\n")
    print ("GT KAYIP DAGILIMI:")
    for k ,v in kova .most_common ():
        print (f"  {k :<12}{v :>6}  %{100 *v /n_gt :.1f}")
    print (f"\n{'kademe':<20}{'AGIRLIKLI':>11}{'dusuk-CP':>11}{'cok-CP':>10}{'kazanc':>9}")
    onc ,res_ =None ,{}
    for a in AD :
        v =f1w (S [a ])
        dl =f1w (rj_s [a ]["dusuk"])if rj_s [a ]["dusuk"]else float ("nan")
        ck =f1w (rj_s [a ]["cok"])if rj_s [a ]["cok"]else float ("nan")
        kz =""if onc is None else f"{v -onc :+.4f}"
        print (f"{a :<20}{v :>11.4f}{dl :>11.4f}{ck :>10.4f}{kz :>9}")
        res_ [a ]={"agirlikli":v ,"dusuk":dl ,"cok":ck }
        onc =v 
    print (f"\nESLESTIRICI FARKI (secicisiz baseline): "
    f"tespit acgozlu {f1w (Ta ):.4f} | robot acgozlu {f1w (Ra ):.4f}")
    with io .open (MAKBUZ ,"w",encoding ="utf-8")as f :
        json .dump ({"cluster":"d6","muhur":sv ["sha16"],"kova":dict (kova ),
        "merdiven":res_ ,"hash":kod_muhru ()},f ,indent =1 ,
        ensure_ascii =False )
    print (f"receipt -> {MAKBUZ }")


if __name__ =="__main__":
    main ()
