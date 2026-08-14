# -*- coding: utf-8 -*-
"""HAVUZ GERI CAGIRMA: temsilin real olcusu (F1 DEGIL).

Onceki kosuda "candidate kahini" diye yazdigim column kahin DEGILDI: gate'siz havuzun
F1'iydi and genis pool extra candidates yuzunden FP cezasi yiyordu -- full as
`gate-before-poz-after-tavani-kirpiyor` kaydindaki 2 numarali error. Bir havuzun
temsil gucu GERI CAGIRMA with olculur: GT'nin yuzde kaci havuzda ULASILABILIR.
Bire-a Macar, TESPIT toleransi, angle serbest.
"""
import collections ,json ,os ,pickle ,sys 
import numpy as np 
import makbuz_hash 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,".")
from sina_cluster import match_hungarian 

d7 =set (map (str ,json .load (open ("results/d7_sinav_kumesi.json"))["pidler"]))
out ={}
for ad ,f in (("G7 (kanonik)","results/_der_yeni_G7BIRLESIK.pkl"),
("g10 (A3-A4)","results/_der_yeni_g10.pkl")):
    R =[r for r in pickle .load (open (f ,"rb"))if str (r ["pid"])in d7 ]
    TP =FN =0 ;nA =0 ;per =collections .defaultdict (lambda :[0 ,0 ])
    for r in R :
        G =np .asarray (r .get ("G",[]),float )
        if not len (G ):
            continue 
        P =np .asarray (r ["P"],float );D =np .asarray (r ["Pd"],float )
        tp ,fp ,fn =match_hungarian (P ,D ,G ,np .asarray (r ["Gd"],float ),r ["diag"],
        max (3.0 ,0.06 *r ["diag"]),180.0 ,True )[:3 ]
        TP +=tp ;FN +=fn ;nA +=len (P )
        a =per [r ["mfg"]];a [0 ]+=tp ;a [1 ]+=fn 
    rc =TP /max (TP +FN ,1 )
    pm ={m :a [0 ]/max (a [0 ]+a [1 ],1 )for m ,a in per .items ()}
    out [ad ]={"recall":rc ,"aday_sayisi":nA ,"n_parca":len (R ),
    "aday_per_parca":nA /max (len (R ),1 ),"brand":pm }
    print (f"{ad :<14} pool recall {rc :.4f} | toplam candidate {nA :>6} "
    f"({nA /max (len (R ),1 ):.1f}/part)")
a ,b =out ["G7 (kanonik)"],out ["g10 (A3-A4)"]
print (f"\nRECALL FARKI {b ['recall']-a ['recall']:+.4f} | "
f"ADAY SAYISI {b ['aday_per_parca']/max (a ['aday_per_parca'],1e-9 ):.2f}x")
print (f"\n{'brand':<8} {'G7':>8} {'g10':>8} {'fark':>8}")
for m in sorted (a ["brand"],key =lambda k :-a ["brand"][k ]):
    print (f"  {m :<7} {a ['brand'][m ]:>7.4f} {b ['brand'].get (m ,0 ):>8.4f} "
    f"{b ['brand'].get (m ,0 )-a ['brand'][m ]:>+8.4f}")
json .dump ({"damga":makbuz_hash .damga (),"sonuc":out ,
"not":"HAVUZ GERI CAGIRMA (F1 DEGIL). D7 brand-disi, tespit tolerans, "
"aci serbest, bire-a Macar."},
open ("results/havuz_recall_d7.json","w"),indent =1 )
