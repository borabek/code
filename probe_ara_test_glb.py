# -*- coding: utf-8 -*-
"""ARA TEST: robot GLB yolu, TANIDIK markadan GORULMEMIS 5 part.

Amac SAYI SISIRMEK DEGIL, urunun sahada ne yaptigini part part gormek:
each part for GT kac CP diyor, urun kac CP uretti, kaci ROBOT olcutunu
(lateral <=2mm, ISARETLI angle <=10 derece, axial <=40mm) gecti, and tier
dagilimi ne.

Parcalar `_ara_test_parcalar.txt`ten okunur. Hepsi VAL kumesinden, i.e.
**brand egitimde VAR, this PARCALAR absent** -- real kullanim senaryosu.
Zorluk yelpazesi kasten genis secildi (GT 24/20/8/4/1) ki easy part
secip test sisirilmesin.

Kullanilan path `robot_cp.extract`, i.e. ihracatcilarin (GLB) BUGUN
cagirdigi path. Yani here gorulen number, GLB'de gorulecek seyin full
karsiligidir.
"""
import io 
import json 
import os 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

import canonical_d7 as K # noqa: E402
import d6_record # noqa: E402
from sina_cluster import match_hungarian # noqa: E402

LISTE =os .environ .get ("AT_LISTE","_ara_test_parcalar.txt")


def _birim (v ):
    v =np .asarray (v ,float ).reshape (-1 ,3 )
    return v /np .maximum (np .linalg .norm (v ,axis =1 ,keepdims =True ),1e-12 )


def main ():
    import torch 
    import robot_cp 
    from infer_step_cp import load_any 

    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    cfg =json .load (io .open ("cp_config.json",encoding ="utf-8"))
    ca =float (cfg .get ("robot_conf_auto",0.5 ))
    mav =int (cfg .get ("robot_min_auto_votes",3 ))
    models =[load_any (c ,dev =dev )[:2 ]
    for c in cfg ["robot_vote2_checkpoints"]]
    print (f"cihaz {dev } | {len (models )} kontrol noktasi "
    f"| threshold {ca } / min_oy {mav }\n",flush =True )

    kay =K .yukle ()
    kay .update ({str (p ):r for p ,r in d6_record .yukle ().items ()
    if str (p )not in kay })
    STEP =K .step_map ()
    pids =[x .strip ()for x in io .open (LISTE ,encoding ="utf-8")
    if x .strip ()]

    top ={k :[0 ,0 ,0 ]for k in ("tespit","rob","rbi")}
    satirlar =[]
    for pid in pids :
        if pid not in STEP or pid not in kay :
            print (f"  {pid }: STEP ya da GT yok -- ATLANDI")
            continue 
        r =kay [pid ]
        G =np .asarray (r ["G"],float ).reshape (-1 ,3 )
        Gd =_birim (r ["Gd"])
        dg =float (r ["diag"])
        cps =robot_cp .extract (models ,STEP [pid ],dev ,ca ,mav )
        P =np .asarray ([c ["point"]for c in cps ],
        float ).reshape (-1 ,3 )if cps else np .zeros ((0 ,3 ))
        D =(_birim ([c ["direction"]for c in cps ])if cps 
        else np .zeros ((0 ,3 )))
        tier ={}
        for c in cps :
            tier [c .get ("tier","?")]=tier .get (c .get ("tier","?"),0 )+1 
        s ={"pid":pid ,"gt":len (G ),"cp":len (cps ),"tier":tier }
        for ad ,tol ,am ,pct ,isr in (("tespit",0.0 ,180.0 ,True ,False ),
        ("rob",2.0 ,10.0 ,False ,False ),
        ("rbi",2.0 ,10.0 ,False ,True )):
            tp ,fp ,fn ,_ =match_hungarian (P ,D ,G ,Gd ,dg ,tol ,am ,pct ,
            signed =isr )
            top [ad ][0 ]+=tp 
            top [ad ][1 ]+=fp 
            top [ad ][2 ]+=fn 
            s [ad ]=tp 
        satirlar .append (s )
        print (f"  {pid :12s} GT={len (G ):3d}  uretilen={len (cps ):3d}  "
        f"tespit_dogru={s ['tespit']:3d}  robot_eksen={s ['rob']:3d}  "
        f"ROBOT_ISARETLI={s ['rbi']:3d}  tier={tier }",flush =True )

    def f1 (t ):
        return 2 *t [0 ]/max (2 *t [0 ]+t [1 ]+t [2 ],1 )

    print (f"\n{'part':13s} {'GT':>4s} {'uretilen':>9s} {'tespit':>7s} "
    f"{'robot':>6s} {'ISARETLI':>9s}")
    for s in satirlar :
        print (f"{s ['pid']:13s} {s ['gt']:4d} {s ['cp']:9d} "
        f"{s ['tespit']:7d} {s ['rob']:6d} {s ['rbi']:9d}")
    print (f"\n--- 5 PARCA TOPLAMI (mikro F1) ---")
    for ad ,isim in (("tespit","tespit"),("rob","robot-axis"),
    ("rbi","robot-ISARETLI")):
        tp ,fp ,fn =top [ad ]
        print (f"  {isim :16s} F1 {f1 (top [ad ]):.4f}   "
        f"(TP {tp } / FP {fp } / FN {fn })")
    print ("\nNOT: 5 part KUCUK bir ornek -- bu sayilar headline DEGILDIR.")
    print ("Manset VAL 100 parcadir (tespit 0.7878 / robot-ISARETLI 0.4839).")
    json .dump ({"parts":satirlar ,
    "toplam":{k :f1 (v )for k ,v in top .items ()}},
    io .open ("results/ara_test_glb.json","w",encoding ="utf-8"),
    indent =1 )
    print ("-> results/ara_test_glb.json")
    return 0 


if __name__ =="__main__":
    sys .exit (main ())
