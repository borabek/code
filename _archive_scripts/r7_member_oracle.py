# -*- coding: utf-8 -*-
"""R7: UYE-ORACLE -- four modelin ATILAN direction/konumlarinda ne up to bilgi present?

Denetimin P2 maddesi. `robot_cp._vote2` anlasan uyeleri birlestirirken KONUMU confidence-agirlikli
ortaliyor and YONU temsilciden aliyor; digerlerinin yonu ATILIYOR.

BU OLCUM WHY SIMDI DEGERLI: pose head'den after baglayici sart ACI'ya dondu
(lateral gecis %88.9, angle %80.4). Dort modelden BIRI correct yonu bulmus may be and biz onu
atiyor olabiliriz. Bu, angle kolunda never denenmemis TEK yerdir.

OLCULEN (oracle = GT'ye bakip most iyi uyeyi secmek; ULASILABILIR TARGET DEGIL, UST SINIR):
    K0  mevcut birlestirme (dagitilan)
    K1  + most iyi uye YONU secilseydi
    K2  + most iyi uye KONUMU secilseydi
    K3  + ikisi birden

KILL (denetimin yazdigi): oracle kazanci < 0.02 whereas arm OLDURULUR; buyukse ogrenilmis a
uye selector kurulur.
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


def main ():
    import cp_openings 
    import measure_set 
    import robot_cp 
    import wire_gate 
    from big_arbiter import eligible 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from sina_cluster import esle ,f1w 
    from sklearn .ensemble import RandomForestClassifier 

    with io .open ("cp_config.json",encoding ="utf-8")as f :
        cfg =json .load (f )
    pp =cfg ["prediction_postproc"]
    MINV =int (pp ["min_vertices"]);VC =float (pp ["vertex_confidence_mask"])
    CL =float (pp ["cluster_mm"])
    mfg_of ={p :m for m ,p ,jf ,s in eligible ()}
    DER ,rap =measure_set .cluster ("results/_der_zengin.pkl")
    measure_set .rapor_bas (rap )
    izin ={r ["pid"]for r in DER }
    gk =measure_set .geo_anahtarlari ();tg ={r ["geo"]for r in DER }

    # gate (sizintisiz)
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
    print ("gate hazir",flush =True )

    # UYE bazinda candidates: probability onbelleginden yeniden derive (network cikarimi YOK)
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
    print (f"{len (VF )}/{len (izin )} parcanin olasiligi present",flush =True )

    DERM ={r ["pid"]:r for r in DER }
    det ={k :[]for k in ("K0","K1","K2","K3")}
    rob ={k :[]for k in ("K0","K1","K2","K3")}
    gruplar =[]
    for i_ ,(pid ,r0 )in enumerate (VF .items (),1 ):
        if i_ %50 ==0 :
            print (f"  {i_ }/{len (VF )}",flush =True )
        rd =DERM .get (pid )
        if rd is None or rd ["X"]is None or rd .get ("XR")is None or not len (rd ["G"]):
            continue 
        V =np .ascontiguousarray (r0 ["V"],np .float64 );F =np .ascontiguousarray (r0 ["F"],np .int64 )
        plist =[np .asarray (pb ,np .float64 )for pb in r0 ["pbs"]]
        per =[cp_openings .connection_points (
        V ,F ,q .argmax (-1 ),min_v =MINV ,classes =(CE ,CT ),dedupe_mm =10.0 ,
        probs =q ,vertex_conf =VC ,ct_depth_min_mm =1.0 ,cluster_mm =CL ,
        step_path =r0 ["stp"])for q in plist ]

        X58 =np .hstack ([rd ["X"],rd ["XR"]])
        s =wire_gate .decision_score (gate ,X58 )
        k =wire_gate .decision_mask (s )
        P0 =rd ["P"][k ].copy ()if k .any ()else np .zeros ((0 ,3 ))
        Pd0 =rd ["Pd"][k ].copy ()if k .any ()else np .zeros ((0 ,3 ))
        if len (P0 ):
            cps =[{"point":P0 [i ],"direction":Pd0 [i ]}for i in range (len (P0 ))]
            cps =wire_gate .pose_correct (X58 [k ],cps )
            if cfg .get ("robot_aci_secici"):
                cps =wire_gate .angle_correct (X58 [k ],cps )
            P0 =np .array ([c ["point"]for c in cps ],float )
            Pd0 =np .array ([c ["direction"]for c in cps ],float )

        G =np .asarray (rd ["G"],float );Gd =np .asarray (rd ["Gd"],float )
        # each kabul edilen CP for, 5mm icindeki UYE adaylarini topla
        UP ,UD =[],[]
        for i in range (len (P0 )):
            up ,ud =[P0 [i ]],[Pd0 [i ]]
            for lst in per :
                for c in lst :
                    q =np .asarray (c ["point"],float )
                    if np .linalg .norm (q -P0 [i ])<=5.0 :
                        up .append (q );ud .append (np .asarray (c ["direction"],float ))
            UP .append (np .array (up ));UD .append (np .array (ud ))

            # KAHIN secim: each CP for GT'ye EN YAKIN uye (direction / konum)
        P1 ,Pd1 ,P2 ,Pd2 ,P3 ,Pd3 =(P0 .copy (),Pd0 .copy (),P0 .copy (),Pd0 .copy (),
        P0 .copy (),Pd0 .copy ())
        for i in range (len (P0 )):
            if not len (G ):
                break 
            b =int (np .argmin (np .linalg .norm (G -P0 [i ],axis =1 )))
            ac =np .degrees (np .arccos (np .clip (np .abs (UD [i ]@Gd [b ]),0 ,1 )))
            j =int (np .argmin (ac ))
            Pd1 [i ]=UD [i ][j ];Pd3 [i ]=UD [i ][j ]
            w =UP [i ]-G [b ]
            al =w @Gd [b ]
            pe =np .linalg .norm (w -al [:,None ]*Gd [b ],axis =1 )
            j2 =int (np .argmin (pe ))
            P2 [i ]=UP [i ][j2 ];P3 [i ]=UP [i ][j2 ]

        rj ="very"if rd ["n"]>=8 else "low"
        for ad ,(P ,Pd )in (("K0",(P0 ,Pd0 )),("K1",(P1 ,Pd1 )),
        ("K2",(P2 ,Pd2 )),("K3",(P3 ,Pd3 ))):
            det [ad ].append ((rj ,)+esle (P ,Pd ,G ,Gd ,rd ["diag"],0.0 ,180.0 ,True ))
            rob [ad ].append ((rj ,)+esle (P ,Pd ,G ,Gd ,rd ["diag"],2.0 ,10.0 ,False ))
        gruplar .append (rd ["geo"])

    print (f"\n{len (gruplar )} part puanlandi")
    print (f"\n{'arm':<28}{'detection':>10}{'ROBOT':>10}{'d_robot':>10}")
    ad_uzun ={"K0":"K0 mevcut birlestirme","K1":"K1 +oracle uye YONU",
    "K2":"K2 +oracle uye KONUMU","K3":"K3 +ikisi birden"}
    r0v =f1w (rob ["K0"])
    SON ={}
    for ad in ("K0","K1","K2","K3"):
        SON [ad ]=(float (f1w (det [ad ])),float (f1w (rob [ad ])))
        print (f"{ad_uzun [ad ]:<28}{SON [ad ][0 ]:>10.4f}{SON [ad ][1 ]:>10.4f}"
        f"{SON [ad ][1 ]-r0v :>+10.4f}")
    en =max (("K1","K2","K3"),key =lambda a :SON [a ][1 ])
    kaz =SON [en ][1 ]-r0v 
    print (f"\nEN IYI KAHIN: {ad_uzun [en ]} -> robot {kaz :+.4f}")
    print (f"KILL (denetimin yazdigi): oracle kazanci < 0.02 ise KOL OLU -> "
    f"{'CANLI, ogrenilmis selector kurulabilir'if kaz >=0.02 else 'OLU'}")
    with io .open ("results/r7_member_oracle.json","w",encoding ="utf-8")as f :
        json .dump ({"kollar":{k :list (v )for k ,v in SON .items ()},"en_iyi":en ,
        "oracle_kazanc":float (kaz ),"canli":bool (kaz >=0.02 )},f ,indent =1 )
    print ("receipt -> results/r7_member_oracle.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
