# -*- coding: utf-8 -*-
"""FN TAKSONOMISI -- TAM zincir + YENI yigin (g10 + gate v7).

Hata bankasi g5 doneminden kalma (GATE_REDDI %43 / ADAY_YOK %41 / KALABALIK %15).
Yigin degisti; kovalarin YENIDEN olculmesi lazim ki remaining kollar correct yere baksin.

KOVALAR (oncelik sirasiyla):
  ADAY_YOK    : GT'nin detection toleransinda HIC candidate absent (temsil)
  GATE_REDDI  : candidate VAR but gate elemis (karar)
  KALABALIK   : candidate present, gate gecmis, but Macar baska GT'ye vermis (rekabet)
  POZ         : eslesmis but robot toleransini gecemiyor (lateral/angle)
"""
import collections ,glob ,json ,os ,pickle ,sys 
import numpy as np 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,".")
import d6_record ,robot_cp ,wire_gate ,product_chain 
from p1c_threshold import maske 
from sina_cluster import match_hungarian 
from corpus_identity import step_kimlik as SK 

OB ="results/_p1_olasilik_g10";GATE ="results/wire_gate_v7.pkl"
sv =d6_record .exam ();rec_ =d6_record .yukle (set (sv ["pidler"]))
gate =pickle .load (open (GATE ,"rb"))
S ={SK (s ):s for s in glob .glob ("all_wscad_stp/*.stp")}
pidler =sorted ({f [:-4 ]for f in os .listdir (OB )if f .endswith (".npz")}&set (rec_ ))

bucket =collections .Counter ();n_gt =0 
for pid in pidler :
    r =rec_ [pid ]
    G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
    if not len (G ):continue 
    d =np .load (f"{OB }/{pid }.npz")
    V =np .ascontiguousarray (d ["V"],np .float64 );F =np .ascontiguousarray (d ["F"],np .int64 )
    cps ,_o ,_c ,_p =robot_cp .derive_candidates (V ,F ,[np .asarray (q ,float )for q in d ["pbs"]],
    S .get (pid ))
    n_gt +=len (G )
    if not cps :
        bucket ["ADAY_YOK"]+=len (G );continue 
    P0 =np .asarray ([c ["point"]for c in cps ],float )
    D0 =np .asarray ([c ["direction"]for c in cps ],float )
    avg =np .asarray (d ["pbs"],float ).mean (0 )
    Xp =np .asarray (wire_gate .feats_for (V ,F ,avg ,cps ,robot_cp .CE ,robot_cp .CT ,
    step_path =S .get (pid )),float )
    k =maske (np .asarray (wire_gate .decision_score (gate ,Xp ),float ),0.40 ,0.30 )
    P1 ,D1 =(P0 [k ],D0 [k ])if k .any ()else (P0 [:0 ],D0 [:0 ])
    if len (P1 ):
        P1 ,D1 =product_chain .tam_poz (V ,F ,avg ,P1 ,D1 ,step_path =S .get (pid ))
    tol =max (3.0 ,0.06 *r ["diag"])
    es_rob =set ()
    if len (P1 ):
        _t ,_f ,_n ,bi =match_hungarian (P1 ,D1 ,G ,Gd ,r ["diag"],2.0 ,10.0 ,False ,signed =True )
        es_rob ={gi for (_pi ,gi ,*_x )in bi ["eslesme"]}
        _t2 ,_f2 ,_n2 ,bi2 =match_hungarian (P1 ,D1 ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True )
        es_tes ={gi for (_pi ,gi ,*_x )in bi2 ["eslesme"]}
    else :
        es_tes =set ()
    for i in range (len (G )):
        if i in es_rob :continue # robot-TP, FN not
        yakin_ham =np .min (np .linalg .norm (P0 -G [i ],axis =1 ))<=tol if len (P0 )else False 
        yakin_gate =np .min (np .linalg .norm (P1 -G [i ],axis =1 ))<=tol if len (P1 )else False 
        if not yakin_ham :bucket ["ADAY_YOK"]+=1 
        elif not yakin_gate :bucket ["GATE_REDDI"]+=1 
        elif i not in es_tes :bucket ["KALABALIK"]+=1 
        else :bucket ["POZ"]+=1 # eslesti but robot toleransi absent
print (f"part {len (pidler )} | GT {n_gt } | robot-FN {sum (bucket .values ())}\n")
for k_ ,v in bucket .most_common ():
    print (f"  {k_ :<12} {v :>5}  %{100 *v /max (sum (bucket .values ()),1 ):.1f}")
json .dump (dict (bucket ),open ("results/fn_taksonomi_g10v7.json","w"),indent =1 )
print ("\nmakbuz -> results/fn_taksonomi_g10v7.json")
