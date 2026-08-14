# -*- coding: utf-8 -*-
"""KANONIK ZINCIRLE D7: 0.2344 uretilebiliyor mu? Sonra p5-v2 AYNI zincirde.

Once VERIFICATION: kanonik zincir + old network (D7 onbellegi) ~0.2344 vermeli.
Vermezse measurement yolu HALA urunun yaptigi sey DEGILDIR and devam edilmez.
"""
import glob ,json ,os ,pickle ,sys 
import numpy as np 
import receipt_hash 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,".")
import d6_record ,product_chain ,canonical_chain as KZ 
from sina_cluster import match_hungarian ,f1w 
from corpus_identity import step_kimlik as SK 

YANAL ,ACI =2.0 ,10.0 
d7p =set (map (str ,json .load (open ("results/d7_exam_set.json"))["pidler"]))
rec_ =d6_record .yukle (d7p )
S ={SK (s ):s for s in glob .glob ("all_wscad_stp/*.stp")}
OB ="results/_p1_olasilik_d7"
pidler =sorted ({f [:-4 ]for f in os .listdir (OB )if f .endswith (".npz")}&set (rec_ ))
print (f"part {len (pidler )}",flush =True )

T ,R =[],[]
for pid in pidler :
    r =rec_ [pid ]
    G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
    if not len (G ):
        continue 
    d =np .load (f"{OB }/{pid }.npz")
    V =np .ascontiguousarray (d ["V"],np .float64 )
    F =np .ascontiguousarray (d ["F"],np .int64 )
    pbs =[np .asarray (q ,float )for q in d ["pbs"]]
    cps =KZ .product_output (V ,F ,pbs ,S .get (pid ))
    P ,D =KZ .poz_ver (cps )
    if len (P ):
        P ,D =product_chain .tam_poz (V ,F ,np .asarray (d ["pbs"],float ).mean (0 ),
        P ,D ,step_path =S .get (pid ))
    T .append ((len (G ),)+match_hungarian (P ,D ,G ,Gd ,r ["diag"],0. ,180. ,True )[:3 ])
    R .append ((len (G ),)+match_hungarian (P ,D ,G ,Gd ,r ["diag"],YANAL ,ACI ,
    False ,signed =True )[:3 ])
t ,rr =f1w (T ),f1w (R )
print (f"\nKANONIK ZINCIR (eski ag): detection {t :.4f} | robot {rr :.4f}")
print (f"MANSET BEKLENTISI       : detection 0.4847 | robot 0.2344")
print (f"FARK                    : detection {t -0.4847 :+.4f} | robot {rr -0.2344 :+.4f}")
json .dump ({"damga":receipt_hash .damga (),"detection":t ,"robot":rr ,
"manset_tespit":0.4847 ,"manset_robot":0.2344 ,"n_parca":len (T )},
open ("results/canonical_d7_dogrulama.json","w"),indent =1 )
print ("receipt -> results/canonical_d7_dogrulama.json")
