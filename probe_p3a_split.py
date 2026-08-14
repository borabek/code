# -*- coding: utf-8 -*-
"""P3-a: `split_ratio` taramasi (birlesmis komsu agizlari ayirma).

`cp_openings._split_elongated` urun yolunda HIC gecirilmiyordu (default 0.0 =
KAPALI, olu kod). `robot_cp.py` 2026-08-09'da config'ten gecirilebilir yapildi.

KAPI (plan): genel candidate tavani +0.02 VEYA kalabalik (very-CP) recall +0.05
getirmezse OLDUR.

Olcum COK-CP parcalarda: split oralarda anlamli (komsu kutuplar birlesiyor).
"""
import glob ,json ,os ,sys 
import numpy as np 
import receipt_hash 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,".")
import d6_record ,robot_cp ,cp_openings 
from sina_cluster import match_hungarian ,f1w 
from corpus_identity import step_kimlik as SK 

OB ="results/_p1_olasilik_g10"
sv =d6_record .exam ();rec_ =d6_record .yukle (set (sv ["pidler"]))
S ={SK (s ):s for s in glob .glob ("all_wscad_stp/*.stp")}
# COK-CP parts (>=8 GT) -- bolmenin anlamli oldugu regime
pidler =sorted ([p for p in {f [:-4 ]for f in os .listdir (OB )if f .endswith (".npz")}
&set (rec_ )if len (rec_ [p ].get ("G",[]))>=8 ])
print (f"very-CP part {len (pidler )}",flush =True )

cfg =robot_cp ._load_cfg ()
res_ ={}
for sr in (0.0 ,0.4 ,0.5 ,0.6 ):
    cfg .setdefault ("prediction_postproc",{})["split_ratio"]=sr 
    T =[]
    for pid in pidler :
        r =rec_ [pid ]
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        d =np .load (f"{OB }/{pid }.npz")
        V =np .ascontiguousarray (d ["V"],np .float64 )
        F =np .ascontiguousarray (d ["F"],np .int64 )
        cps ,_o ,_c ,_p =robot_cp .derive_candidates (
        V ,F ,[np .asarray (q ,float )for q in d ["pbs"]],S .get (pid ),cfg =cfg )
        P =np .asarray ([c ["point"]for c in cps ],float )if cps else np .zeros ((0 ,3 ))
        D =np .asarray ([c ["direction"]for c in cps ],float )if cps else np .zeros ((0 ,3 ))
        T .append ((len (G ),)+match_hungarian (P ,D ,G ,Gd ,r ["diag"],0. ,180. ,True )[:3 ])
    n_aday =sum (int (t [1 ]+t [2 ])for t in T )
    tp =sum (int (t [1 ])for t in T );fn =sum (int (t [3 ])for t in T )
    rec =tp /max (tp +fn ,1 )
    res_ [str (sr )]={"oracle":f1w (T ),"recall":rec ,"candidate":n_aday }
    print (f"  split_ratio={sr }: oracle {f1w (T ):.4f} | recall {rec :.4f} | candidate {n_aday }",
    flush =True )
t0 =res_ ["0.0"]
en =max ((k for k in res_ if k !="0.0"),key =lambda k :res_ [k ]["oracle"])
dk =res_ [en ]["oracle"]-t0 ["oracle"];dr =res_ [en ]["recall"]-t0 ["recall"]
print (f"\nEN IYI split_ratio={en }: oracle {dk :+.4f} | recall {dr :+.4f}")
print (f"KARAR: {'ACIK'if (dk >=0.02 or dr >=0.05 )else 'OLDUR (<+0.02 oracle ve <+0.05 recall)'}")
json .dump ({"damga":receipt_hash .damga (),"sonuc":res_ ,"en_iyi":en ,
"kahin_fark":dk ,"recall_fark":dr ,"n_parca":len (pidler ),
"not":"COK-CP rejimi (>=8 GT). Kapi: oracle +0.02 VEYA recall +0.05"},
open ("results/p3a_split_ratio.json","w"),indent =1 )
print ("receipt -> results/p3a_split_ratio.json")
