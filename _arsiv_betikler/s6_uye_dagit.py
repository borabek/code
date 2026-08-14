# -*- coding: utf-8 -*-
"""S6: UYE SECICIYI dagit and UCTAN UCA olc (full corpus verisiyle).

R7 (kahin):   four modelin ATILAN yonlerinde robot 0.5523 -> 0.6059 yetecek bilgi VAR (+0.0536)
R9 (ten-kestirim, YALNIZ 194 parcanin verisiyle): ogrenilmis selector +0.0301, kahinin %56'si,
    GA [+0.0070, +0.0560] KANITLI

Bu betik TAM KORPUS uye verisiyle (r8_uye_veri.py) secicinin last halini egitir, uctan uca
olcer and gecerse dagitir.

POSE HEAD DERSI: same fikir 1068 satirla KANITSIZDI (+0.0222), 6330 satirla KANITLI (+0.0428).
Burada 3215 -> ~30000 row bekleniyor.

KILL (onceden yazili): robot >= +0.02 VE tespit kaybi < 0.005 VE GA sifiri dislamali.
"""
import io 
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
UYE ="results/uye_veri.npz"
YAKIN =5.0 


def main ():
    import cp_openings 
    import measure_set 
    import wire_gate 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from sina_cluster import esle ,f1w 
    from sklearn .ensemble import RandomForestClassifier 

    if not os .path .exists (UYE ):
        raise SystemExit (f"YOK: {UYE } (r8_uye_veri.py bitmedi)")
    with io .open ("cp_config.json",encoding ="utf-8")as f :
        cfg =json .load (f )
    pp =cfg ["prediction_postproc"]
    MINV =int (pp ["min_vertices"]);VC =float (pp ["vertex_confidence_mask"])
    CL =float (pp ["cluster_mm"])
    DER ,rap =measure_set .cluster ("results/_der_zengin.pkl")
    measure_set .rapor_bas (rap )
    izin ={r ["pid"]for r in DER }
    gk =measure_set .geo_anahtarlari ();tg ={r ["geo"]for r in DER }

    d =np .load (UYE ,allow_pickle =True )
    UX =np .asarray (d ["X"],float );UY =np .asarray (d ["y"])
    UG =np .array ([str (x )for x in d ["geo"]])
    disi =~np .isin (UG ,list (tg ))
    print (f"uye verisi: {len (UY )} satir -> leakage disi {int (disi .sum ())} "
    f"| {len (set (UG [disi ]))} grup | 'dogru' orani {UY [disi ].mean ():.1%}",flush =True )
    sec =RandomForestClassifier (n_estimators =500 ,min_samples_leaf =5 ,n_jobs =-1 ,
    random_state =0 ).fit (UX [disi ],UY [disi ])

    zen =np .load ("results/zengin_parite.npz",allow_pickle =True )
    Xt =np .hstack ([np .asarray (zen ["X22"],float ),np .asarray (zen ["XR"],float )])
    ytr =np .asarray (zen ["y"]);tpid =np .array ([str (x )for x in zen ["pids"]])
    tgrp =np .array ([gk .get (p ,"absent:"+p )for p in tpid ]);keep =~np .isin (tgrp ,list (tg ))
    dag =wire_gate ._load (wire_gate .MODEL_PATH );DON =dag .get ("donusum")
    Z =np .zeros ((len (Xt ),Xt .shape [1 ]*2 ))
    for u in np .unique (tpid ):
        i =np .where (tpid ==u )[0 ]
        Z [i ]=wire_gate .within_part (Xt [i ],DON )
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (Z [keep ],ytr [keep ])
    gate ={"clf":clf ,"n_feat":Z .shape [1 ],"donusum":DON }

    VF ={}
    for cluster in ("dev","val"):
        cf =f"results/_probs_{cluster }.pkl"
        if not os .path .exists (cf )and cluster =="dev":
            cf ="results/_h_probs.pkl"
        with open (cf ,"rb")as f :
            for r in pickle .load (f ):
                pid =os .path .basename (r ["stp"]).split ("_")[1 ]
                if pid in izin :
                    VF [pid ]=r 
    DERM ={r ["pid"]:r for r in DER }

    det ={"A":[],"B":[]};rob ={"A":[],"B":[]};g =[]
    for i_ ,(pid ,r0 )in enumerate (VF .items (),1 ):
        if i_ %50 ==0 :
            print (f"  {i_ }/{len (VF )}",flush =True )
        rd =DERM .get (pid )
        if rd is None or rd ["X"]is None or rd .get ("XR")is None :
            continue 
        V =np .ascontiguousarray (r0 ["V"],np .float64 );F =np .ascontiguousarray (r0 ["F"],np .int64 )
        per =[cp_openings .connection_points (
        V ,F ,np .asarray (q ,float ).argmax (-1 ),min_v =MINV ,classes =(CE ,CT ),dedupe_mm =10.0 ,
        probs =np .asarray (q ,float ),vertex_conf =VC ,ct_depth_min_mm =1.0 ,cluster_mm =CL ,
        step_path =r0 ["stp"])for q in r0 ["pbs"]]
        X58 =np .hstack ([rd ["X"],rd ["XR"]])
        s =wire_gate .decision_score (gate ,X58 )
        k =wire_gate .decision_mask (s )
        P =rd ["P"][k ].copy ()if k .any ()else np .zeros ((0 ,3 ))
        Pd =rd ["Pd"][k ].copy ()if k .any ()else np .zeros ((0 ,3 ))
        PdB =Pd .copy ()
        if len (P ):
            cps =[{"point":P [i ],"direction":Pd [i ]}for i in range (len (P ))]
            cps =wire_gate .pose_correct (X58 [k ],cps )
            if cfg .get ("robot_aci_secici"):
                cps =wire_gate .angle_correct (X58 [k ],cps )
            P =np .array ([c ["point"]for c in cps ],float )
            Pd =np .array ([c ["direction"]for c in cps ],float )
            PdB =Pd .copy ()
            Xk =X58 [k ]
            for i in range (len (P )):
                uy =[(Pd [i ],1.0 ,0.0 )]
                for lst in per :
                    for m in lst :
                        q =np .asarray (m ["point"],float )
                        dd =float (np .linalg .norm (q -P [i ]))
                        if dd <=YAKIN :
                            uy .append ((np .asarray (m ["direction"],float ),
                            float (m .get ("confidence",1.0 )),dd ))
                if len (uy )<2 :
                    continue 
                DIR =np .array ([u [0 ]for u in uy ])
                DIR =DIR /(np .linalg .norm (DIR ,axis =1 ,keepdims =True )+1e-9 )
                CONF =np .array ([u [1 ]for u in uy ]);MES =np .array ([u [2 ]for u in uy ])
                ort =DIR .mean (0 );ort /=np .linalg .norm (ort )+1e-9 
                a_ort =np .degrees (np .arccos (np .clip (np .abs (DIR @ort ),0 ,1 )))
                a_bir =np .degrees (np .arccos (np .clip (np .abs (DIR @Pd [i ]),0 ,1 )))
                rank_ =np .argsort (np .argsort (-CONF ))
                Fm =np .array ([[CONF [u_ ],MES [u_ ],a_ort [u_ ],a_bir [u_ ],float (len (DIR )),
                float (np .mean (a_ort )),float (rank_ [u_ ])]+Xk [i ].tolist ()
                for u_ in range (len (DIR ))],float )
                PdB [i ]=DIR [int (np .argmax (sec .predict_proba (Fm )[:,1 ]))]
        rj ="very"if rd ["n"]>=8 else "low"
        G =np .asarray (rd ["G"],float );Gd =np .asarray (rd ["Gd"],float )
        det ["A"].append ((rj ,)+esle (P ,Pd ,G ,Gd ,rd ["diag"],0.0 ,180.0 ,True ))
        rob ["A"].append ((rj ,)+esle (P ,Pd ,G ,Gd ,rd ["diag"],2.0 ,10.0 ,False ))
        det ["B"].append ((rj ,)+esle (P ,PdB ,G ,Gd ,rd ["diag"],0.0 ,180.0 ,True ))
        rob ["B"].append ((rj ,)+esle (P ,PdB ,G ,Gd ,rd ["diag"],2.0 ,10.0 ,False ))
        g .append (rd ["geo"])

    print (f"\n{'arm':<26}{'tespit':>10}{'ROBOT':>10}")
    print (f"{'A dagitilan':<26}{f1w (det ['A']):>10.4f}{f1w (rob ['A']):>10.4f}")
    print (f"{'B +uye selector':<26}{f1w (det ['B']):>10.4f}{f1w (rob ['B']):>10.4f}")
    dr =f1w (rob ["B"])-f1w (rob ["A"]);dt =f1w (det ["B"])-f1w (det ["A"])
    cift =list (zip (rob ["A"],rob ["B"]))
    fn =lambda rows :f1w ([y for _ ,y in rows ])-f1w ([x for x ,_ in rows ])
    _ ,lo ,hi =measure_set .grup_bootstrap (cift ,g ,fn ,n =2000 )
    print (f"\nrobot {dr :+.4f} [{lo :+.4f}, {hi :+.4f}] | tespit {dt :+.4f}")
    gecti =dr >=0.02 and dt >-0.005 and lo >0 
    print (f"KILL: robot >= +0.02 VE tespit kaybi < 0.005 VE GA sifiri dislamali -> "
    f"{'GECTI'if gecti else 'GECMEDI'}")
    with io .open ("results/s6_uye_dagit.json","w",encoding ="utf-8")as f :
        json .dump ({"A_tespit":float (f1w (det ["A"])),"A_robot":float (f1w (rob ["A"])),
        "B_tespit":float (f1w (det ["B"])),"B_robot":float (f1w (rob ["B"])),
        "d_robot":float (dr ),"ga":[float (lo ),float (hi )],
        "gecti":bool (gecti ),"n_satir":int (disi .sum ())},f ,indent =1 )
    if gecti :
        with open ("results/uye_secici.pkl","wb")as f :
            pickle .dump ({"sec":sec ,"n_feat":UX .shape [1 ],"yakin_mm":YAKIN ,
            "note":("UYE SECICI 2026-08-02: _vote2 birlestirmede ATILAN uye "
            "yonlerinden most iyisini selects. Kahin +0.0536 (R7); ogrenilmis "
            f"selector {dr :+.4f} (GA [{lo :+.4f},{hi :+.4f}]). Tespit bedeli "
            "YAPISAL SIFIR: tespit olcutu aciya bakmaz.")},f )
        print ("-> results/uye_secici.pkl")
    print ("receipt -> results/s6_uye_dagit.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
