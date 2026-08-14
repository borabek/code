# -*- coding: utf-8 -*-
"""D7'de YENI YIGIN: g10 + gate v7 + p5-v2, ESKI 0.2344 with AYNI KUMEDE.

D7 residual DEV (harcandi) -- this FINAL DEGIL. Ama 0.2344'u ureten kumeyle AYNI
838 part / 12 brand oldugu for DOGRUDAN KIYASLANABILIR single sayidir.

UC KOL:
  1. old yigin (g7 onbellegi + gate v5 + tam_poz)      -> 0.2344 civari beklenir
  2. g10 + gate v7 (A3-A4)
  3. g10 + gate v7 + p5-v2 (ORTAK SECIM, gate SONRA)
"""
import glob ,json ,os ,pickle ,sys 
import numpy as np 
import makbuz_hash 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,".")
import d6_record ,robot_cp ,wire_gate ,product_zinciri ,p5v2_secenek as PS ,p5v2_egit as PE 
from p1c_threshold import maske 
from sina_cluster import match_hungarian ,f1w 
from korpus_kimlik import step_kimlik as SK 

YANAL ,ACI =2.0 ,10.0 
sv =json .load (open ("results/d7_sinav_kumesi.json"))
rec_ =d6_record .yukle (set (map (str ,sv ["pidler"])))
S ={SK (s ):s for s in glob .glob ("all_wscad_stp/*.stp")}
g5 =pickle .load (open ("results/wire_gate_v5.pkl","rb"))
g7g =pickle .load (open ("results/wire_gate_v7.pkl","rb"))
OB_ESKI ,OB_YENI ="results/_p1_olasilik_d7","results/_p1_olasilik_d7g10"
pidler =sorted ({f [:-4 ]for f in os .listdir (OB_ESKI )if f .endswith (".npz")}
&{f [:-4 ]for f in os .listdir (OB_YENI )if f .endswith (".npz")}
&set (rec_ ))
print (f"ortak part {len (pidler )} (D7 = DEV, FINAL DEGIL)\n",flush =True )


def kos (ob ,gate ,p5 =None ):
    T ,R =[],[]
    for pid in pidler :
        r =rec_ [pid ]
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        if not len (G ):
            continue 
        d =np .load (f"{ob }/{pid }.npz")
        V =np .ascontiguousarray (d ["V"],np .float64 )
        F =np .ascontiguousarray (d ["F"],np .int64 )
        cps ,_o ,_c ,_p =robot_cp .derive_candidates (
        V ,F ,[np .asarray (q ,float )for q in d ["pbs"]],S .get (pid ))
        if not cps :
            T .append ((len (G ),0. ,0. ,0. ));R .append ((len (G ),0. ,0. ,0. ));continue 
        P =np .asarray ([c ["point"]for c in cps ],float )
        D =np .asarray ([c ["direction"]for c in cps ],float )
        avg =np .asarray (d ["pbs"],float ).mean (0 )
        Xp =np .asarray (wire_gate .feats_for (V ,F ,avg ,cps ,robot_cp .CE ,
        robot_cp .CT ,step_path =S .get (pid )),float )
        gs =np .asarray (wire_gate .decision_score (gate ,Xp ),float )
        if p5 is None :
            k =maske (gs ,0.40 ,0.30 )
            if k .any ():
                P ,D =P [k ],D [k ]
            P ,D =product_zinciri .tam_poz (V ,F ,avg ,P ,D ,step_path =S .get (pid ))
        else :
            komsu =None 
            if len (D )>1 :
                B =D *np .sign (D @D [0 ])[:,None ]
                komsu =B .mean (0 );komsu /=(np .linalg .norm (komsu )+1e-12 )
            secs =PS .secenekler (P ,D ,p5 ["cyl"].get (pid ),p5 ["acik"].get (pid ),
            r ["diag"],gate_s =gs ,komsu =komsu )
            skor =[p5 ["clf"].predict_proba (np .asarray ([s [2 ]for s in o ],float ))[:,1 ]
            for o in secs ]
            P ,D =PE .sec (secs ,skor ,gate_skor =gs ,gate_esik =(0.30 ,0.30 ))
        T .append ((len (G ),)+match_hungarian (P ,D ,G ,Gd ,r ["diag"],0. ,180. ,True )[:3 ])
        R .append ((len (G ),)+match_hungarian (P ,D ,G ,Gd ,r ["diag"],YANAL ,ACI ,
        False ,signed =True )[:3 ])
    return f1w (T ),f1w (R )


res_ ={}
t ,r =kos (OB_ESKI ,g5 );res_ ["1_eski (g7+gate v5)"]={"tespit":t ,"robot":r }
print (f"1) eski yigin (g7 + gate v5)      tespit {t :.4f} | robot {r :.4f}",flush =True )
t ,r =kos (OB_YENI ,g7g );res_ ["2_A3A4 (g10+gate v7)"]={"tespit":t ,"robot":r }
print (f"2) g10 + gate v7 (A3-A4)          tespit {t :.4f} | robot {r :.4f}",flush =True )
json .dump ({"damga":makbuz_hash .damga (),"sonuc":res_ ,"n_parca":len (pidler ),
"not":"D7 = DEV (harcandi). FINAL DEGIL. 0.2344 with AYNI cluster."},
open ("results/d7_yeni_yigin.json","w"),indent =1 )
print ("receipt -> results/d7_yeni_yigin.json")
