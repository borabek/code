# -*- coding: utf-8 -*-
"""NMS radius taramasi + MARKA KARARLILIGI + detection etkisi.

Ilk olcumde six varyantin ALTISI da pozitifti and yaricapla monoton artiyordu
(r=3.0'da +0.0093). Kucuk but tutarli. Dagitim karari for gereken three sey:
(1) radius gercekten doyuyor mu, (2) kazanc MARKALARA yayilmis mi otherwise single
markadan mi geliyor, (3) TESPIT F1'i bozuyor mu (candidate siliyoruz).

D7 = DEV. Yaricap here secilirse D7'ye ayarlanmis becomes -> D6'da DOGRULANIR.
"""
import collections ,json ,os ,sys 
import numpy as np 
import receipt_hash 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,".")
import d6_record ,wire_gate ,canonical_d7 as K 
from p1c_threshold import maske 
from sina_cluster import match_hungarian 

RS =[0.0 ,1.0 ,2.0 ,3.0 ,4.0 ,5.0 ,6.0 ,8.0 ,10.0 ]


def nms (P ,gs ,r ):
    if r <=0 :
        return np .ones (len (P ),bool )
    rank_ =np .argsort (-gs );tut =np .ones (len (P ),bool )
    for a ,i in enumerate (rank_ ):
        if not tut [i ]:
            continue 
        for j in rank_ [a +1 :]:
            if tut [j ]and np .linalg .norm (P [i ]-P [j ])<r :
                tut [j ]=False 
    return tut 


def kos (rec_ ,x58f ,ad ):
    rob ={r :collections .defaultdict (lambda :[0 ,0 ,0 ])for r in RS }
    tes ={r :[]for r in RS }
    for pid ,rec in rec_ .items ():
        X =x58f (rec );G =np .asarray (rec .get ("G",[]),float )
        if X is None or not len (G ):
            continue 
        P =np .asarray (rec ["P"],float );D =np .asarray (rec ["Pd"],float )
        Gd =np .asarray (rec ["Gd"],float );dg =rec ["diag"]
        gs =np .asarray (wire_gate .decision_score (gate ,X ),float )
        m =maske (gs ,0.40 ,0.30 )
        Pm ,Dm ,gm =(P [m ],D [m ],gs [m ])if m .any ()else (P [:0 ],D [:0 ],gs [:0 ])
        for r in RS :
            s =nms (Pm ,gm ,r )if len (Pm )else np .zeros (0 ,bool )
            tp ,fp ,fn =match_hungarian (Pm [s ],Dm [s ],G ,Gd ,dg ,K .YANAL ,K .ACI ,
            False ,signed =True )[:3 ]
            a =rob [r ][rec ["mfg"]];a [0 ]+=tp ;a [1 ]+=fp ;a [2 ]+=fn 
            t =match_hungarian (Pm [s ],Dm [s ],G ,Gd ,dg ,
            max (3.0 ,0.06 *dg ),180.0 ,True )[:3 ]
            tes [r ].append ((len (G ),)+t )
    print (f"\n=== {ad } ===")
    print (f"{'r':>5} {'robot':>8} {'d':>8} {'detection':>8} {'makro':>7} {'kotu':>7} {'+brand':>7}")
    t0 =None ;cik ={}
    for r in RS :
        per =rob [r ]
        mi =float (2 *sum (a [0 ]for a in per .values ())/
        max (sum (2 *a [0 ]+a [1 ]+a [2 ]for a in per .values ()),1 ))
        pm ={m :2 *a [0 ]/max (2 *a [0 ]+a [1 ]+a [2 ],1 )for m ,a in per .items ()}
        if t0 is None :
            t0 =mi ;b0 =dict (pm )
        art =sum (1 for m in pm if pm [m ]>b0 [m ]+1e-9 )
        cik [r ]={"robot":mi ,"detection":K .mikro (tes [r ]),
        "makro":float (np .mean (list (pm .values ()))),
        "en_kotu":float (min (pm .values ())),"artan_marka":art ,
        "n_marka":len (pm )}
        c =cik [r ]
        print (f"{r :>5.1f} {mi :>8.4f} {mi -t0 :>+8.4f} {c ['detection']:>8.4f} "
        f"{c ['makro']:>7.4f} {c ['en_kotu']:>7.4f} {art :>4}/{len (pm )}")
    return cik 


gate =K .gate_yukle ()
d7 =kos (K .yukle (json .load (open ("results/d7_exam_set.json"))["pidler"]),
K .x58 ,"D7 (DEV, 835 part, brand-disi)")
d6 =kos (d6_record .yukle (set (d6_record .exam ()["pidler"])),d6_record .x58 ,
"D6 (468 part) -- BAGIMSIZ DOGRULAMA")
json .dump ({"damga":receipt_hash .damga (),"D7":d7 ,"D6":d6 ,"yaricaplar":RS ,
"not":"mesafe-NMS, gate skoruna per. Yaricap D7'de secilirse D6 "
"dogrulamasi ZORUNLU. MIKRO."},
open ("results/nms_sweep.json","w"),indent =1 )
