# -*- coding: utf-8 -*-
"""K2.2+K2.1 UCTAN UCA: periyodik spread tespit/robot'u oynatiyor mu?

Yigin: g10 + gate v7 (bugunku most iyi). Iki arm, AYNI parts:
  A) spread YOK   B) spread VAR (+-1 step, row bazinda)
Yayilan points gate'ten GECMEZ (sentetik noktanin oznitelikleri absent) --
i.e. this kolun KESINLIK BEDELI full as olculur.
"""
import json ,os ,pickle ,sys ,glob 
import numpy as np 
import makbuz_hash 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,".")
import d6_record ,robot_cp ,wire_gate ,product_zinciri ,periyodik_yayilim as PYA 
from p1c_threshold import maske 
from sina_cluster import match_hungarian ,f1w 
from korpus_kimlik import step_kimlik as SK 

OB ="results/_p1_olasilik_g10";GATE ="results/wire_gate_v7.pkl"
ROBOT_YANAL ,ROBOT_ACI =2.0 ,10.0 

sv =d6_record .exam ();kayit =d6_record .yukle (set (sv ["pidler"]))
gate =pickle .load (open (GATE ,"rb"))
S ={SK (s ):s for s in glob .glob ("all_wscad_stp/*.stp")}
pidler =sorted ({f [:-4 ]for f in os .listdir (OB )if f .endswith (".npz")}&set (kayit ))
print (f"part {len (pidler )} | yigin g10 + gate v7\n",flush =True )

sonuc ={}
for ad ,yayilim in (("yayilim YOK",0 ),("yayilim VAR (+-1)",1 )):
    T ,R =[],[]
    ek_toplam =0 
    for pid in pidler :
        r =kayit [pid ]
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        if not len (G ):
            continue 
        d =np .load (f"{OB }/{pid }.npz")
        V =np .ascontiguousarray (d ["V"],np .float64 )
        F =np .ascontiguousarray (d ["F"],np .int64 )
        pbs =[np .asarray (q ,float )for q in d ["pbs"]]
        cps ,_o ,_c ,_p =robot_cp .derive_candidates (V ,F ,pbs ,S .get (pid ))
        if not cps :
            T .append ((len (G ),0.0 ,0.0 ,0.0 ));R .append ((len (G ),0.0 ,0.0 ,0.0 ));continue 
        P =np .asarray ([c ["point"]for c in cps ],float )
        D =np .asarray ([c ["direction"]for c in cps ],float )
        avg =np .asarray (d ["pbs"],float ).mean (0 )
        Xp =np .asarray (wire_gate .feats_for (V ,F ,avg ,cps ,robot_cp .CE ,
        robot_cp .CT ,step_path =S .get (pid )),float )
        k =maske (np .asarray (wire_gate .decision_score (gate ,Xp ),float ),0.40 ,0.30 )
        if k .any ():
            P ,D =P [k ],D [k ]
        P ,D =product_zinciri .tam_poz (V ,F ,avg ,P ,D ,step_path =S .get (pid ))
        if yayilim :
            U ,W =PYA .yay (P ,D ,adim_n =yayilim )
            if len (U ):
                ek_toplam +=len (U )
                P =np .vstack ([P ,U ]);D =np .vstack ([D ,W ])
        T .append ((len (G ),)+match_hungarian (P ,D ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True )[:3 ])
        R .append ((len (G ),)+match_hungarian (P ,D ,G ,Gd ,r ["diag"],ROBOT_YANAL ,
        ROBOT_ACI ,False ,signed =True )[:3 ])
    t ,rr =f1w (T ),f1w (R )
    sonuc [ad ]={"tespit":t ,"robot":rr ,"eklenen_nokta":ek_toplam }
    print (f"{ad :<20} tespit {t :.4f} | robot {rr :.4f} | eklenen {ek_toplam }",flush =True )

a ,b =sonuc ["yayilim YOK"],sonuc ["yayilim VAR (+-1)"]
print (f"\nFARK: tespit {b ['tespit']-a ['tespit']:+.4f} | robot {b ['robot']-a ['robot']:+.4f}")
json .dump (sonuc ,open ("results/k22_periyodik_yayilim.json","w"),indent =1 )
print ("receipt -> results/k22_periyodik_yayilim.json")
