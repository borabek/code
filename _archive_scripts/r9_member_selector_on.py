# -*- coding: utf-8 -*-
"""R9: UYE SECICI ten-kestirim -- kahinin ne kadarini yakalayabiliriz?

R7: oracle uye YONU robot-haziri 0.5523 -> 0.6059 does (+0.0536). Soru: OGRENILMIS a
selector bunun ne kadarini takes?

Bu betik measurement kumesinin KENDI parcalari ten, GEOMETRI GRUBUNA according to capraz dogrulamayla
a ten-kestirim gives. Dagitim for not, DECISION for: full corpus verisi (r8) arka planda
uretiliyor and pahali; before this kolun ne vaat ettigini gormek gerek.

DURUST WARNING: buradaki selector measurement parcalarinin GRUP-DISI kardesleriyle egitiliyor, i.e.
real dagitimdaki up to data gormuyor. Pose head'de data 5.9 fold artinca kazanc two katina
cikmisti -- this kestirim muhtemelen ALT SINIR.
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
YAKIN =5.0 


def main ():
    import cp_openings 
    import measure_set 
    import wire_gate 
    from big_arbiter import eligible 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from sina_cluster import esle ,f1w 
    from sklearn .ensemble import RandomForestClassifier 
    from sklearn .model_selection import GroupKFold 

    with io .open ("cp_config.json",encoding ="utf-8")as f :
        cfg =json .load (f )
    pp =cfg ["prediction_postproc"]
    MINV =int (pp ["min_vertices"]);VC =float (pp ["vertex_confidence_mask"])
    CL =float (pp ["cluster_mm"])
    DER ,rap =measure_set .cluster ("results/_der_zengin.pkl")
    izin ={r ["pid"]for r in DER }
    gk =measure_set .geo_anahtarlari ();tg ={r ["geo"]for r in DER }

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

    # --- each part for kabul edilen CP'ler + uye adaylari + feature/label
    KAY =[]
    for i_ ,(pid ,r0 )in enumerate (VF .items (),1 ):
        if i_ %50 ==0 :
            print (f"  turetme {i_ }/{len (VF )}",flush =True )
        rd =DERM .get (pid )
        if rd is None or rd ["X"]is None or rd .get ("XR")is None or not len (rd ["G"]):
            continue 
        V =np .ascontiguousarray (r0 ["V"],np .float64 );F =np .ascontiguousarray (r0 ["F"],np .int64 )
        per =[cp_openings .connection_points (
        V ,F ,np .asarray (q ,float ).argmax (-1 ),min_v =MINV ,classes =(CE ,CT ),dedupe_mm =10.0 ,
        probs =np .asarray (q ,float ),vertex_conf =VC ,ct_depth_min_mm =1.0 ,cluster_mm =CL ,
        step_path =r0 ["stp"])for q in r0 ["pbs"]]
        X58 =np .hstack ([rd ["X"],rd ["XR"]])
        s =wire_gate .decision_score (gate ,X58 )
        k =wire_gate .decision_mask (s )
        if not k .any ():
            KAY .append (dict (pid =pid ,geo =rd ["geo"],rj ="very"if rd ["n"]>=8 else "low",
            P =np .zeros ((0 ,3 )),Pd =np .zeros ((0 ,3 )),UY =[],G =rd ["G"],
            Gd =rd ["Gd"],diag =rd ["diag"]))
            continue 
        P =rd ["P"][k ].copy ();Pd =rd ["Pd"][k ].copy ()
        cps =[{"point":P [i ],"direction":Pd [i ]}for i in range (len (P ))]
        cps =wire_gate .pose_correct (X58 [k ],cps )
        if cfg .get ("robot_aci_secici"):
            cps =wire_gate .angle_correct (X58 [k ],cps )
        P =np .array ([c ["point"]for c in cps ],float )
        Pd =np .array ([c ["direction"]for c in cps ],float )
        Xk =X58 [k ]
        UY =[]
        for i in range (len (P )):
            uy =[(Pd [i ],1.0 ,0.0 )]
            for lst in per :
                for m in lst :
                    q =np .asarray (m ["point"],float )
                    dd =float (np .linalg .norm (q -P [i ]))
                    if dd <=YAKIN :
                        uy .append ((np .asarray (m ["direction"],float ),
                        float (m .get ("confidence",1.0 )),dd ))
            DIR =np .array ([u [0 ]for u in uy ])
            DIR =DIR /(np .linalg .norm (DIR ,axis =1 ,keepdims =True )+1e-9 )
            CONF =np .array ([u [1 ]for u in uy ]);MES =np .array ([u [2 ]for u in uy ])
            ort =DIR .mean (0 );ort /=np .linalg .norm (ort )+1e-9 
            a_ort =np .degrees (np .arccos (np .clip (np .abs (DIR @ort ),0 ,1 )))
            a_bir =np .degrees (np .arccos (np .clip (np .abs (DIR @Pd [i ]),0 ,1 )))
            rank_ =np .argsort (np .argsort (-CONF ))
            F_ =np .array ([[CONF [u_ ],MES [u_ ],a_ort [u_ ],a_bir [u_ ],float (len (DIR )),
            float (np .mean (a_ort )),float (rank_ [u_ ])]+Xk [i ].tolist ()
            for u_ in range (len (DIR ))],float )
            UY .append ((DIR ,F_ ))
        KAY .append (dict (pid =pid ,geo =rd ["geo"],rj ="very"if rd ["n"]>=8 else "low",
        P =P ,Pd =Pd ,UY =UY ,G =np .asarray (rd ["G"],float ),
        Gd =np .asarray (rd ["Gd"],float ),diag =float (rd ["diag"])))
    print (f"{len (KAY )} part hazir",flush =True )

    # --- training satirlari (hedef: uye yonu GT'ye <=10 derece)
    RX ,RY ,RG =[],[],[]
    for r in KAY :
        for i ,(DIR ,F_ )in enumerate (r ["UY"]):
            if not len (r ["G"]):
                continue 
            b =int (np .argmin (np .linalg .norm (r ["G"]-r ["P"][i ],axis =1 )))
            a =np .degrees (np .arccos (np .clip (np .abs (DIR @r ["Gd"][b ]),0 ,1 )))
            for u_ in range (len (DIR )):
                RX .append (F_ [u_ ]);RY .append (1 if a [u_ ]<=10 else 0 );RG .append (r ["geo"])
    RX =np .array (RX ,float );RY =np .array (RY );RG =np .array (RG )
    print (f"uye satiri {len (RY )} | 'correct' orani {RY .mean ():.1%} | {len (set (RG ))} grup",
    flush =True )

    oof =np .zeros (len (RY ))
    for tr ,te in GroupKFold (n_splits =5 ).split (RX ,RY ,RG ):
        oof [te ]=RandomForestClassifier (n_estimators =400 ,min_samples_leaf =5 ,n_jobs =-1 ,
        random_state =0 ).fit (RX [tr ],RY [tr ]).predict_proba (RX [te ])[:,1 ]

    def puanla (mod ):
        det ,rob =[],[]
        idx =0 
        for r in KAY :
            P ,Pd =r ["P"].copy (),r ["Pd"].copy ()
            for i ,(DIR ,F_ )in enumerate (r ["UY"]):
                n =len (DIR )
                if mod =="selector"and len (r ["G"]):
                    pr =oof [idx :idx +n ]
                    Pd [i ]=DIR [int (np .argmax (pr ))]
                if len (r ["G"]):
                    idx +=n 
            det .append ((r ["rj"],)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],0.0 ,180.0 ,True ))
            rob .append ((r ["rj"],)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],2.0 ,10.0 ,False ))
        return det ,rob 

    g =[r ["geo"]for r in KAY ]
    d0 ,r0 =puanla ("absent");d1 ,r1 =puanla ("selector")
    print (f"\n{'arm':<24}{'detection':>10}{'ROBOT':>10}")
    print (f"{'A mevcut':<24}{f1w (d0 ):>10.4f}{f1w (r0 ):>10.4f}")
    print (f"{'B ogrenilmis selector':<24}{f1w (d1 ):>10.4f}{f1w (r1 ):>10.4f}")
    dr =f1w (r1 )-f1w (r0 )
    cift =list (zip (r0 ,r1 ))
    fn =lambda rows :f1w ([y for _ ,y in rows ])-f1w ([x for x ,_ in rows ])
    _ ,lo ,hi =measure_set .grup_bootstrap (cift ,g ,fn ,n =2000 )
    print (f"\nrobot {dr :+.4f} [{lo :+.4f}, {hi :+.4f}] | oracle +0.0536 -> "
    f"yakalanan {dr /0.0536 :.0%}")
    with io .open ("results/r9_member_selector_on.json","w",encoding ="utf-8")as f :
        json .dump ({"baseline":float (f1w (r0 )),"selector":float (f1w (r1 )),"d_robot":float (dr ),
        "ga":[float (lo ),float (hi )],"oracle":0.0536 ,
        "yakalanan":float (dr /0.0536 )},f ,indent =1 )
    print ("receipt -> results/r9_member_selector_on.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
