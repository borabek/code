# -*- coding: utf-8 -*-
"""T9: HAVUZA ACI-DUZELTMESI ONCESI YONU DE KOY (single variable).

FINDING: dagitilan zincir this sirayla calisiyor
    pose -> angle_correct (YONU DEGISTIRIR) -> pick_member_direction (pool = degistirilmis direction + uyeler)
Yani uye secicinin "angle duzeltmesini GERI AL" secenegi YOK; duzeltilmis direction havuzda single
temsilci as duruyor.

T8'de havuza IKISI birden konmustu (correction oncesi + sonrasi) and kahinin %66'si yakalanmisti;
dagitilan zincirde %52. Ama T8'in tabani farkliydi, i.e. comparison KIRLI.

BU BETIK single degiskenle olcer:
    A  dagitilan zincir (pool = angle-duzeltilmis direction + uyeler)
    B  pool = angle-duzeltilmis direction + DUZELTME ONCESI direction + uyeler

Mantik: angle duzeltmesi sometimes YANLIS duzeltiyor may be; selector that durumda old yonu geri
secebilmeli. Bu, "kararsizsa dokunma" ilkesinin selector by uygulanmis hali.

KILL: robot >= +0.01 VE GA sifiri dislamali VE detection degismemeli.
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
YAKIN =5.0 


def main ():
    import measure_set 
    import wire_gate 
    from sina_cluster import esle ,f1w 
    from sklearn .ensemble import RandomForestClassifier 

    with io .open ("cp_config.json",encoding ="utf-8")as f :
        cfg =json .load (f )
    DER ,rap =measure_set .cluster ("results/_der_tam.pkl")
    gk =measure_set .geo_anahtarlari ();tg ={r ["geo"]for r in DER }
    g =[r ["geo"]for r in DER ]

    zen =np .load ("results/zengin_parite.npz",allow_pickle =True )
    Xt =np .hstack ([np .asarray (zen ["X22"],float ),np .asarray (zen ["XR"],float )])
    ytr =np .asarray (zen ["y"]);tpid =np .array ([str (x )for x in zen ["pids"]])
    keep =~np .isin (np .array ([gk .get (p ,"absent:"+p )for p in tpid ]),list (tg ))
    dag =wire_gate ._load (wire_gate .MODEL_PATH );DON =dag .get ("donusum")
    Z =np .zeros ((len (Xt ),Xt .shape [1 ]*2 ))
    for u in np .unique (tpid ):
        i =np .where (tpid ==u )[0 ]
        Z [i ]=wire_gate .within_part (Xt [i ],DON )
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (Z [keep ],ytr [keep ])
    gate ={"clf":clf ,"n_feat":Z .shape [1 ],"donusum":DON }
    uye_m =wire_gate ._load (wire_gate .UYE_PATH )
    print (f"gate + uye selector hazir ({uye_m ['n_feat']} sutun)",flush =True )

    def sec_yon (X58i ,pool ):
        DIR =np .array ([h [0 ]for h in pool ])
        DIR =DIR /(np .linalg .norm (DIR ,axis =1 ,keepdims =True )+1e-9 )
        CONF =np .array ([h [1 ]for h in pool ]);MES =np .array ([h [2 ]for h in pool ])
        ort =DIR .mean (0 );ort /=np .linalg .norm (ort )+1e-9 
        a_ort =np .degrees (np .arccos (np .clip (np .abs (DIR @ort ),0 ,1 )))
        a_bir =np .degrees (np .arccos (np .clip (np .abs (DIR @DIR [0 ]),0 ,1 )))
        rank_ =np .argsort (np .argsort (-CONF ))
        F =np .array ([[CONF [u ],MES [u ],a_ort [u ],a_bir [u ],float (len (DIR )),
        float (np .mean (a_ort )),float (rank_ [u ])]+X58i .tolist ()
        for u in range (len (DIR ))],float )
        if F .shape [1 ]!=uye_m ["n_feat"]:
            return DIR [0 ]
        return DIR [int (np .argmax (uye_m ["sec"].predict_proba (F )[:,1 ]))]

    def puanla (oncesini_ekle ):
        det ,rob =[],[]
        for r in DER :
            P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
            if r ["X"]is not None and r .get ("XR")is not None and r .get ("UYE"):
                X58 =np .hstack ([r ["X"],r ["XR"]])
                s =wire_gate .decision_score (gate ,X58 );k =wire_gate .decision_mask (s )
                if k .any ():
                    P =r ["P"][k ].copy ();Pd =r ["Pd"][k ].copy ()
                    cps =[{"point":P [i ],"direction":Pd [i ]}for i in range (len (P ))]
                    cps =wire_gate .pose_correct (X58 [k ],cps )
                    onc =np .array ([c ["direction"]for c in cps ],float )# ACI DUZELTMESI ONCESI
                    cps =wire_gate .angle_correct (X58 [k ],cps )
                    P =np .array ([c ["point"]for c in cps ],float )
                    Pd =np .array ([c ["direction"]for c in cps ],float )
                    Xk =X58 [k ]
                    for i in range (len (P )):
                        hav =[(Pd [i ],1.0 ,0.0 )]
                        if oncesini_ekle :
                            hav .append ((onc [i ],0.95 ,0.0 ))
                        for lst in r ["UYE"]:
                            for m in lst :
                                q =np .asarray (m ["point"],float )
                                dd =float (np .linalg .norm (q -P [i ]))
                                if dd <=YAKIN :
                                    hav .append ((np .asarray (m ["direction"],float ),
                                    float (m .get ("confidence",1.0 )),dd ))
                        if len (hav )>1 :
                            Pd [i ]=sec_yon (Xk [i ],hav )
            rj ="very"if r ["n"]>=8 else "low"
            det .append ((rj ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],0.0 ,180.0 ,True ))
            rob .append ((rj ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],2.0 ,10.0 ,False ))
        return det ,rob 

    dA ,rA =puanla (False )
    dB ,rB =puanla (True )
    print (f"\n{'arm':<34}{'detection':>10}{'ROBOT':>10}")
    print (f"{'A dagitilan (pool: duzeltilmis)':<34}{f1w (dA ):>10.4f}{f1w (rA ):>10.4f}")
    print (f"{'B +correction ONCESI direction de':<34}{f1w (dB ):>10.4f}{f1w (rB ):>10.4f}")
    dr =f1w (rB )-f1w (rA );dt =f1w (dB )-f1w (dA )
    fn =lambda rows :f1w ([y for _ ,y in rows ])-f1w ([x for x ,_ in rows ])
    _ ,lo ,hi =measure_set .grup_bootstrap (list (zip (rA ,rB )),g ,fn ,n =2000 )
    print (f"\nrobot {dr :+.4f} [{lo :+.4f}, {hi :+.4f}] | detection {dt :+.4f}")
    gecti =dr >=0.01 and lo >0 and abs (dt )<1e-9 
    print (f"KILL: robot >= +0.01 VE GA sifiri dislamali VE detection degismemeli -> "
    f"{'GECTI'if gecti else 'GECMEDI'}")
    with io .open ("results/t9_havuza_ikisi.json","w",encoding ="utf-8")as f :
        json .dump ({"A_robot":float (f1w (rA )),"B_robot":float (f1w (rB )),
        "difference":float (dr ),"ga":[float (lo ),float (hi )],
        "gecti":bool (gecti )},f ,indent =1 )
    print ("receipt -> results/t9_havuza_ikisi.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
