# -*- coding: utf-8 -*-
"""D6 URUN YOLU OLCUMU -- `probe_dagitim_dogrula.py`'nin D6 ikizi.

WHY SEPARATE BIR BETIK: gate kararlari D6'da veriliyor but D6 olcumlerim
ONBELLEKTEN (`_tam_oz`) kosuyordu. Onbellek `cp_config.json`'un old halinde
turetilmis and segmentasyon adaylari kaymis (`results/p6_parite_d6.json`).
Onbellekten olculen number URUNUN count degildir; gate that is why urunun CANLI
yolundan gecmeli.

Bu betik `canonical_chain.product_output`'yi cagirir -- i.e. URUNUN TEK zincirini --
and D6'da olcer. Kollar cevre degiskenleriyle secilir, single degiskenli kiyas for:
    URUN_P6=0 URUN_GENIS=1   dagitilan baseline
    URUN_P6=1                P6 ortak siralayici
    DOG_POZ=0                poz kafasi KAPALI (P6 yonu own selects)
"""
import collections 
import json 
import os 
import sys 

import numpy as np 

import makbuz_hash 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,".")

OB ="results/_p1_olasilik"
N =int (os .environ .get ("DOG_N","0"))
POZ =os .environ .get ("DOG_POZ","1")!="0"


def main ():
    import d6_record 
    import canonical_d7 as K 
    import canonical_chain 
    import robot_cp 
    import product_genis 
    import product_p6 
    import product_zinciri 
    from sina_cluster import match_hungarian 

    S =K .step_map ()
    pidler =sorted (f [:-4 ]for f in os .listdir (OB )if f .endswith (".npz"))
    kay ={str (p ):r for p ,r in d6_record .yukle (pidler ).items ()}
    secili =[p for p in pidler if p in kay and len (kay [p ].get ("G",[]))
    and S .get (p )]
    if N :
        secili =secili [:N ]
    sh =os .environ .get ("DOG_SHARD")# `birlestir_makbuz.py` with birlesir
    if sh :
        i_ ,n_ =(int (x )for x in sh .split ("/"))
        secili =[p for k ,p in enumerate (secili )if k %n_ ==i_ ]
        print (f"PAY {i_ }/{n_ }",flush =True )
    print (f"D6 {len (secili )} part | P6 {'ACIK'if product_p6 .ACIK else 'KAPALI'}"
    f" | genis {'ACIK'if product_genis .ACIK else 'KAPALI'}"
    f" | poz kafasi {'ACIK'if POZ else 'KAPALI'}",flush =True )

    rob =collections .defaultdict (lambda :[0 ,0 ,0 ])
    tes =[]
    kirilim ={}
    cfg =robot_cp ._load_cfg ()
    for i ,pid in enumerate (secili ,1 ):
        r =kay [pid ]
        z =np .load (f"{OB }/{pid }.npz")
        V =np .ascontiguousarray (z ["V"],np .float64 )
        F =np .ascontiguousarray (z ["F"],np .int64 )
        pbs =[np .asarray (q ,float )for q in z ["pbs"]]
        cps =canonical_chain .product_output (V ,F ,pbs ,S .get (pid ),cfg =cfg )
        P ,D =canonical_chain .poz_ver (cps )
        if len (P )and POZ :
            P ,D =product_zinciri .tam_poz (V ,F ,np .mean (pbs ,axis =0 ),P ,D ,
            step_path =S .get (pid ))
        G =np .asarray (r ["G"],float )
        Gd =np .asarray (r ["Gd"],float )
        dg =float (r ["diag"])
        tp ,fp ,fn =match_hungarian (P ,D ,G ,Gd ,dg ,K .YANAL ,K .ACI ,False ,
        signed =True )[:3 ]
        a =rob [r ["mfg"]]
        a [0 ]+=tp ;a [1 ]+=fp ;a [2 ]+=fn 
        t_ =match_hungarian (P ,D ,G ,Gd ,dg ,max (3.0 ,0.06 *dg ),180.0 ,True )[:3 ]
        tes .append ((len (G ),)+t_ )
        kirilim [pid ]={"mfg":r ["mfg"],"rob":[tp ,fp ,fn ],
        "tes":[int (x )for x in t_ ]}
        if i %50 ==0 :
            print (f"  {i }/{len (secili )}",flush =True )

    pm ={m :2 *v [0 ]/max (2 *v [0 ]+v [1 ]+v [2 ],1 )for m ,v in rob .items ()}
    mi =float (2 *sum (v [0 ]for v in rob .values ())/
    max (sum (2 *v [0 ]+v [1 ]+v [2 ]for v in rob .values ()),1 ))
    out ={"robot":mi ,"tespit":K .mikro (tes ),
    "makro":float (np .mean (list (pm .values ()))),
    "en_kotu":float (min (pm .values ())),"brand":pm ,
    "TP":sum (v [0 ]for v in rob .values ()),
    "FP":sum (v [1 ]for v in rob .values ()),
    "FN":sum (v [2 ]for v in rob .values ()),
    "n_parca":len (secili ),"p6_acik":bool (product_p6 .ACIK ),
    "p6_sayac":dict (product_p6 .SAYAC ),
    "genis_acik":bool (product_genis .ACIK ),"poz_kafasi":bool (POZ ),
    "parca_kirilim":kirilim }
    print (f"\nD6 URUN ZINCIRI robot {mi :.4f} | tespit {out ['tespit']:.4f} | "
    f"makro {out ['makro']:.4f} | TP {out ['TP']} FP {out ['FP']} "
    f"FN {out ['FN']}")
    yol =os .environ .get ("DOG_CIKTI","results/d6_urun.json")
    json .dump ({"damga":makbuz_hash .damga (),"sonuc":out ,
    "not":"URUNUN TEK kanonik zinciri, D6 (gorulmemis brand: SUPU/"
    "UPUN/MOR/NIT/UTL/S+S/SE/ONV). MIKRO."},
    open (yol ,"w"),indent =1 )
    print (f"receipt -> {yol }")


if __name__ =="__main__":
    main ()
