# -*- coding: utf-8 -*-
"""P2 DECISION KAPISI: new network D7'de HAVUZ RECALL'u artiriyor mu?

RULE (this oturumda konuldu): new a segmentasyon agi before HAVUZ RECALL with
sinanir, F1 with DEGIL. Gerekce: gate'siz F1 extra adaylari FP sayar and iyi a
havuzu kotu gosterir; also gate old dagilimda egitildigi for uctan uca dusus
agin not gate'in uyumsuzlugunun isareti becomes.

TABAN (`results/pool_recall_d7.json`): kanonik G7 pool recall **0.6654**.
Yeni network bunu GECMEZSE arm KAPANIR and uctan uca olcume SOKULMAZ.

Aday uretimi urunun own fonksiyonuyla (`robot_cp.derive_candidates`), detection
toleransi, angle serbest, bire-a Macar.
"""
import collections 
import json 
import os 
import sys 

import numpy as np 

import receipt_hash 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,".")
import canonical_d7 as K # noqa: E402
import robot_cp # noqa: E402
from sina_cluster import match_hungarian # noqa: E402


def olc (ob ,kay ,S ):
    TP =FN =nA =0 
    per =collections .defaultdict (lambda :[0 ,0 ])
    none =error =0 
    for pid ,r in sorted (kay .items ()):
        G =np .asarray (r .get ("G",[]),float )
        f =f"{ob }/{pid }.npz"
        if not len (G ):
            continue 
        if not os .path .exists (f ):
            none +=1 
            continue 
        z =np .load (f )
        V =np .ascontiguousarray (z ["V"],np .float64 )
        F =np .ascontiguousarray (z ["F"],np .int64 )
        pbs =[np .asarray (q ,float )for q in z ["pbs"]]
        # CIPLAK except YOK: olcemeyen betik DECISION URETMEZ, patlar.
        cps ,_op ,_cok ,_per =robot_cp .derive_candidates (V ,F ,pbs ,S .get (pid ))
        P =np .asarray ([c ["point"]for c in cps ],float )if cps else np .zeros ((0 ,3 ))
        D =np .asarray ([c ["direction"]for c in cps ],float )if cps else np .zeros ((0 ,3 ))
        tol =max (3.0 ,0.06 *float (r ["diag"]))
        tp ,_fp ,fn =match_hungarian (P ,D ,G ,np .asarray (r ["Gd"],float ),
        float (r ["diag"]),tol ,180.0 ,True )[:3 ]
        TP +=tp 
        FN +=fn 
        nA +=len (P )
        a =per [r ["mfg"]]
        a [0 ]+=tp 
        a [1 ]+=fn 
    return {"recall":TP /max (TP +FN ,1 ),"candidate":nA ,
    "aday_per_parca":nA /max (len (kay ),1 ),
    "brand":{m :a [0 ]/max (a [0 ]+a [1 ],1 )for m ,a in per .items ()},
    "onbellegi_olmayan":none }


def main ():
    S =K .step_map ()
    kay =K .yukle (json .load (open ("results/d7_exam_set.json"))["pidler"])
    kollar ={
    "G7 (kanonik TABAN)":"results/_p1_olasilik_d7",
    "P2-brep (YENI)":"results/_p1_olasilik_p2brep",
    }
    out ={}
    for ad ,ob in kollar .items ():
        if not os .path .isdir (ob ):
            raise SystemExit (f"{ob } YOK -- measurement yapilamaz")
        out [ad ]=olc (ob ,kay ,S )
        c =out [ad ]
        print (f"{ad :<22} pool recall {c ['recall']:.4f} | candidate/part "
        f"{c ['aday_per_parca']:.1f} | onbellegi none {c ['onbellegi_olmayan']}",
        flush =True )
    a =out ["G7 (kanonik TABAN)"]
    b =out ["P2-brep (YENI)"]
    d =b ["recall"]-a ["recall"]
    print (f"\nFARK {d :+.4f}")
    art =sum (1 for m in b ["brand"]if b ["brand"][m ]>a ["brand"].get (m ,0 )+1e-9 )
    print (f"artan brand {art }/{len (b ['brand'])}")
    print (f"\n{'brand':<8} {'G7':>8} {'P2':>8} {'diff':>8}")
    for m in sorted (a ["brand"],key =lambda k :a ["brand"][k ]):
        print (f"  {m :<7} {a ['brand'][m ]:>7.4f} {b ['brand'].get (m ,0 ):>8.4f} "
        f"{b ['brand'].get (m ,0 )-a ['brand'][m ]:>+8.4f}")
    print ("\nKARAR: "+("GECTI -- uctan uca olcume gecilir"
    if d >0 else "KALDI -- arm KAPANIR"))
    json .dump ({"damga":receipt_hash .damga (),"sonuc":out ,"diff":d ,
    "artan_marka":art ,"gecti":bool (d >0 ),
    "not":"HAVUZ RECALL karar kapisi. F1 DEGIL. D7 brand-disi."},
    open ("results/p2_pool_recall.json","w"),indent =1 )
    print ("receipt -> results/p2_pool_recall.json")


if __name__ =="__main__":
    main ()
