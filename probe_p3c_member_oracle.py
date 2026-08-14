# -*- coding: utf-8 -*-
"""P3-c: UYE-ONERI KAHINI -- ek seed egitmeye value mi?

SORU: g10 s1/s2 egitmek (saatler) kazanc getirir mi? Kapi: uye havuzunu
BIRLESTIRMEK candidate kahinini +0.05 artiriyorsa EGIT, <+0.02 whereas KAPAT.

VEKIL: elimizde g7 and g10 present (different egitimler). Bunlarin BIRLESIMI, two
seed'in birlesimi for a ALT SINIRDIR (g7 and g10 birbirinden g10-s0/s1'den
DAHA different, i.e. real seed kazanci bundan KUCUK becomes).
"""
import glob ,json ,os ,sys 
import numpy as np 
import receipt_hash 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,".")
import d6_record ,robot_cp 
from sina_cluster import match_hungarian ,f1w 
from corpus_identity import step_kimlik as SK 

A ,B ="results/_p1_olasilik_g10","results/_p1_olasilik_g7"
sv =d6_record .exam ();rec_ =d6_record .yukle (set (sv ["pidler"]))
S ={SK (s ):s for s in glob .glob ("all_wscad_stp/*.stp")}
ortak =sorted ({f [:-4 ]for f in os .listdir (A )if f .endswith (".npz")}
&{f [:-4 ]for f in os .listdir (B )if f .endswith (".npz")}&set (rec_ ))
print (f"ortak part {len (ortak )}",flush =True )


def candidates (ob ,pid ):
    d =np .load (f"{ob }/{pid }.npz")
    V =np .ascontiguousarray (d ["V"],np .float64 );F =np .ascontiguousarray (d ["F"],np .int64 )
    cps ,_o ,_c ,_p =robot_cp .derive_candidates (
    V ,F ,[np .asarray (q ,float )for q in d ["pbs"]],S .get (pid ))
    if not cps :
        return np .zeros ((0 ,3 )),np .zeros ((0 ,3 ))
    return (np .asarray ([c ["point"]for c in cps ],float ),
    np .asarray ([c ["direction"]for c in cps ],float ))


T_a ,T_b ,T_u =[],[],[]
for pid in ortak :
    r =rec_ [pid ]
    G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
    if not len (G ):
        continue 
    Pa ,Da =candidates (A ,pid );Pb ,Db =candidates (B ,pid )
    Pu =np .vstack ([Pa ,Pb ])if len (Pa )or len (Pb )else np .zeros ((0 ,3 ))
    Du =np .vstack ([Da ,Db ])if len (Da )or len (Db )else np .zeros ((0 ,3 ))
    for L ,(P ,D )in ((T_a ,(Pa ,Da )),(T_b ,(Pb ,Db )),(T_u ,(Pu ,Du ))):
        L .append ((len (G ),)+match_hungarian (P ,D ,G ,Gd ,r ["diag"],0. ,180. ,True )[:3 ])
a ,b ,u =f1w (T_a ),f1w (T_b ),f1w (T_u )
print (f"g10 tek       : {a :.4f}")
print (f"g7  tek       : {b :.4f}")
print (f"BIRLESIM      : {u :.4f}")
print (f"BIRLESIM KAZANCI (en iyi tekten): {u -max (a ,b ):+.4f}")
k =u -max (a ,b )
print (f"KARAR: {'EGIT (>=+0.05)'if k >=0.05 else ('KAPAT (<+0.02)'if k <0.02 else 'BELIRSIZ (0.02-0.05)')}")
json .dump ({"damga":receipt_hash .damga (),"g10":a ,"g7":b ,"birlesim":u ,
"kazanc":k ,"n_parca":len (T_a ),
"not":"g7+g10 birlesimi, IKI SEED birlesimi for ALT SINIR"},
open ("results/p3c_member_oracle.json","w"),indent =1 )
print ("receipt -> results/p3c_member_oracle.json")
