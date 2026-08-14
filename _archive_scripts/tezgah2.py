# -*- coding: utf-8 -*-
"""TEZGAH2: remaining kollari AYNI ADIL TABANA karsi olcen ortak altyapi.

WHY: G3'te tabani dagitilan gate'i yukleyerek olctum and 0.8686 output -- because that gate'in
training verisi measurement kumesinin ~391 parcasini iceriyor (SIZINTI). Duzeltince baseline full
as mansetin 0.7584'u became. Bu tezgah that duzeltilmis tabani TEK YERDE kurar ki each arm
same zeminde olculsun and error tekrarlanmasin.

TABAN: old turetme (_der_tam.pkl) + old corpus (zengin_parite_w2.npz) but OLCUM
GRUPLARI CIKARILARAK egitilmis gate -> detection 0.7584 / robot 0.5893 (headline with birebir).
"""
import io 
import json 
import os 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

ESKI_NPZ ="results/zengin_parite_w2.npz"
DER_YOL ="results/_der_tam.pkl"


def yukle ():
    """(DER, gate, name) -- adil baseline gate'i and measurement kumesi."""
    import measure_set 
    import wire_gate 
    from sklearn .ensemble import RandomForestClassifier 
    DER ,rap =measure_set .cluster (DER_YOL )
    measure_set .rapor_bas (rap )
    d =np .load (ESKI_NPZ ,allow_pickle =True )
    X =(np .hstack ([np .asarray (d ["X22"],float ),np .asarray (d ["XR"],float )])
    if "X22"in d .files else np .asarray (d ["X"],float ))
    y =np .asarray (d ["y"]);pid =np .array ([str (x )for x in d ["pids"]])
    mfg =np .array ([str (x )for x in d ["mfg"]])
    with io .open ("results/_strict_geometry_keys.json",encoding ="utf-8")as f :
        gk =json .load (f )
    tg ={r ["geo"]for r in DER }
    keep =~np .isin (np .array ([gk .get (x ,"absent:"+x )for x in pid ]),list (tg ))
    Z =np .zeros ((len (X ),X .shape [1 ]*2 ))
    for u in np .unique (pid ):
        i =np .where (pid ==u )[0 ]
        Z [i ]=wire_gate .within_part (X [i ],"zskor")
    gate ={"clf":RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (Z [keep ],y [keep ]),
    "n_feat":Z .shape [1 ],"donusum":"zskor"}
    try :
        import gate_bench as T 
        ad =T .yukle ()["name"]
    except Exception :
        ad =None 
    return DER ,gate ,{"Z":Z ,"y":y ,"pid":pid ,"mfg":mfg ,"keep":keep ,"name":ad }


def puanla (DER ,gate ,karar =None ,duzelt =True ):
    """Urunun karar yolu. `karar(score, r, X)` verilirse default maskenin YERINE gecer."""
    import wire_gate 
    from sina_cluster import esle 
    det ,rob ,gg =[],[],[]
    for r in DER :
        P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
        if r ["X"]is not None and r .get ("XR")is not None :
            X =np .hstack ([r ["X"],r ["XR"]])
            if X .shape [1 ]*2 ==gate ["n_feat"]:
                sk =wire_gate .decision_score (gate ,X )
                k =(wire_gate .decision_mask (sk )if karar is None else karar (sk ,r ,X ))
                if k .any ():
                    P =np .asarray (r ["P"],float )[k ].copy ()
                    Pd =np .asarray (r ["Pd"],float )[k ].copy ()
                    if duzelt :
                        c =[{"point":P [i ],"direction":Pd [i ]}for i in range (len (P ))]
                        c =wire_gate .pose_correct (X [k ],c )
                        c =wire_gate .angle_correct (X [k ],c )
                        if r .get ("UYE"):
                            c =wire_gate .pick_member_direction (X [k ],c ,r ["UYE"])
                        P =np .array ([x ["point"]for x in c ],float )
                        Pd =np .array ([x ["direction"]for x in c ],float )
        rj ="very"if r ["n"]>=8 else "low"
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        det .append ((rj ,)+esle (P ,Pd ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True ))
        rob .append ((rj ,)+esle (P ,Pd ,G ,Gd ,r ["diag"],2.0 ,10.0 ,False ))
        gg .append (r ["geo"])
    return det ,rob ,gg 


def ga (taban_rows ,aday_rows ,gg ,n =3000 ):
    import measure_set 
    from sina_cluster import f1w 
    fn =lambda rows :f1w ([q for _ ,q in rows ])-f1w ([p for p ,_ in rows ])
    _ ,lo ,hi =measure_set .grup_bootstrap (list (zip (taban_rows ,aday_rows )),gg ,fn ,n =n )
    return lo ,hi 


def bas (ad ,det ,rob ,baseline =None ,gg =None ):
    from sina_cluster import f1w 
    s =f"{ad :<30}{f1w (det ):>9.4f}{f1w (rob ):>9.4f}"
    if baseline is not None and gg is not None :
        lo ,hi =ga (baseline [0 ],det ,gg )
        s +=f"   {f1w (det )-f1w (baseline [0 ]):+.4f}  GA[{lo :+.4f},{hi :+.4f}] " f"{'GERCEK'if (lo >0 or hi <0 )else 'noise'}"
    print (s ,flush =True )
    return f1w (det ),f1w (rob )
