# -*- coding: utf-8 -*-
"""P2: KONSOLIDE ABLASYON TABLOSU -- tezin ihtiyaci which is TEK tablo.

SORUN: each blok ayri a gecede, ayri a tabana karsi, ayri a korpusla measured. Bugun
"ZENGIN +0.0233", "FIZ +0.0704", "TOPO +0.012" diye dolasan sayilarin none of them AYNI kosudan
not and therefore TOPLANAMAZLAR. Tez a ablasyon tablosu ister and that tablo TEK kosudan,
AYNI measurement kumesinden, AYNI korpustan cikmalidir.

BU BETIK ONU URETIR. Iki axis:

  OZNITELIK EKSENI (detection + robot):
     13 baseline -> +5 B-rep FIZIKSEL -> +4 icbukey TOPOLOJI -> +36 ZENGIN
     each biri hem HAM hem PARCA-ICI Z-SKOR halinde (i.e. donusumun katkisi da ayrisir)

  DECISION EKSENI (only robot; detection bunlardan YAPISAL as etkilenmez):
     gate -> +pose -> +angle -> +uye

Her row: havuzlanmis / PXC-disi / WEI-disi + grup bootstrap GA (a onceki satira according to).
Hicbir sey dagitilmaz.
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

BLOK =[("13 baseline",13 ),("+5 B-rep fiziksel",18 ),("+4 icbukey topoloji",22 ),
("+36 zengin",58 )]


def main ():
    import gate_bench as T 
    import measure_set 
    import wire_gate 
    from sina_cluster import esle 
    from sklearn .ensemble import RandomForestClassifier 
    from night_kilit import guard 

    guard ("p2")
    D =T .yukle ()
    measure_set .rapor_bas (D ["rap"])
    cfg =D ["cfg"]
    X ,y ,pid ,mfg ,keep =D ["X"],D ["y"],D ["pid"],D ["mfg"],D ["keep"]
    AD =D ["name"]
    print (f"\nsutun haritasi: baseline {AD [:3 ]}... | FIZ {AD [13 :18 ]} | TOPO {AD [18 :22 ]} "
    f"| ZENGIN {AD [22 :25 ]}...({len (AD )-22 } sutun)")

    _asil =wire_gate .decision_score 

    def skor_alt (m ,Xh ):
        if isinstance (m ,dict )and "_n"in m :
            Xh =np .asarray (Xh ,float )[:,:m ["_n"]]
        return _asil (m ,Xh )
    wire_gate .decision_score =skor_alt 

    ONB ={}

    def arm (n_sut ,donusumlu ):
        key_ =(n_sut ,donusumlu )

        def f (X_ ,y_ ,pid_ ,mfg_ ,kp ,th ):
            if key_ not in ONB :
                Xs =X_ [:,:n_sut ]
                if donusumlu :
                    Zz =np .zeros ((len (Xs ),n_sut *2 ))
                    for u in np .unique (pid_ ):
                        i =np .where (pid_ ==u )[0 ]
                        Zz [i ]=wire_gate .within_part (Xs [i ],D ["donusum"])
                else :
                    Zz =Xs 
                ONB [key_ ]=Zz 
            Zz =ONB [key_ ]
            clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
            random_state =th ).fit (Zz [kp ],y_ [kp ])
            return {"clf":clf ,"n_feat":Zz .shape [1 ],
            "donusum":D ["donusum"]if donusumlu else None ,"_n":n_sut }
        return f 

    print ("\n=== OZNITELIK EKSENI ===")
    T .baslik (D )
    SON ,PARCA ={},{}
    for don in (False ,True ):
        for ad ,n in BLOK :
            label_ =f"{ad }{' [z-score]'if don else ' [ham]'}"
            SON [label_ ],PARCA [label_ ]=T .calistir (D ,arm (n ,don ),label_ )
            guard ("p2 "+label_ )

    print (f"\n{'blok':<30}{'pool':>9}{'PXC-d':>9}{'WEI-d':>9}{'onceki farka GA':>24}")
    onceki =None 
    OZET ={}
    for don in (False ,True ):
        for ad ,n in BLOK :
            e =f"{ad }{' [z-score]'if don else ' [ham]'}"
            s =SON [e ]
            ga =""
            if onceki is not None :
                lo ,hi =T .ga (PARCA [onceki ],PARCA [e ],"havuzlanmis")
                ga =f"[{lo :+.4f},{hi :+.4f}] {'GERCEK'if (lo >0 or hi <0 )else 'noise'}"
            print (f"{e :<30}{s ['havuzlanmis']['detection']:>9.4f}"
            f"{s .get ('PXC-disi',{}).get ('detection',float ('nan')):>9.4f}"
            f"{s .get ('WEI-disi',{}).get ('detection',float ('nan')):>9.4f}{ga :>24}")
            OZET [e ]={"havuzlanmis":s ["havuzlanmis"]["detection"],
            "uretici_ort":s ["_URETICI_DISI_ORT"],
            "robot":s ["havuzlanmis"]["robot"]}
            onceki =e 

            # === DECISION EKSENI (only robot) ===
    print ("\n=== DECISION EKSENI (robot; detection bunlardan YAPISAL as etkilenmez) ===")
    Zt =ONB [(58 ,True )]
    gate ={"clf":RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (Zt [keep ],y [keep ]),
    "n_feat":Zt .shape [1 ],"donusum":D ["donusum"],"_n":58 }
    ADIM =[("gate (duzeltmesiz)",False ,False ,False ),("+pose",True ,False ,False ),
    ("+pose +angle",True ,True ,False ),("+pose +angle +uye",True ,True ,True )]
    RSON ={}
    onceki_det =None 
    for ad ,po ,ac ,uy in ADIM :
        rob ,rbi ,gg =[],[],[]
        for r in D ["DER"]:
            P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
            if r ["X"]is not None and r .get ("XR")is not None :
                Xr =np .hstack ([r ["X"],r ["XR"]])
                k =wire_gate .decision_mask (wire_gate .decision_score (gate ,Xr ))
                if k .any ():
                    P =r ["P"][k ].copy ();Pd =r ["Pd"][k ].copy ()
                    if po :
                        c =[{"point":P [i ],"direction":Pd [i ]}for i in range (len (P ))]
                        c =wire_gate .pose_correct (Xr [k ],c )
                        if ac :
                            c =wire_gate .angle_correct (Xr [k ],c )
                        if uy and r .get ("UYE"):
                            c =wire_gate .pick_member_direction (Xr [k ],c ,r ["UYE"])
                        P =np .array ([x ["point"]for x in c ],float )
                        Pd =np .array ([x ["direction"]for x in c ],float )
            rj ="very"if r ["n"]>=8 else "low"
            rob .append ((rj ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],2.0 ,10.0 ,False ))
            rbi .append ((rj ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],2.0 ,10.0 ,False ,
            signed =True ))
            gg .append (r ["geo"])
        ga =""
        if onceki_det is not None :
            fn =lambda rows :T .f1w ([q for _ ,q in rows ])-T .f1w ([p for p ,_ in rows ])
            _ ,lo ,hi =measure_set .grup_bootstrap (list (zip (onceki_det ,rob )),gg ,fn ,n =2000 )
            ga =f"[{lo :+.4f},{hi :+.4f}] {'GERCEK'if (lo >0 or hi <0 )else 'noise'}"
        RSON [ad ]={"robot":T .f1w (rob ),"robot_isaretli":T .f1w (rbi )}
        print (f"{ad :<24}robot {T .f1w (rob ):.4f}  signed {T .f1w (rbi ):.4f}  {ga }")
        onceki_det =rob 
    with io .open ("results/p2_ablasyon.json","w",encoding ="utf-8")as f :
        json .dump ({"feature":OZET ,"karar":RSON ,
        "not":("TEK run, AYNI measurement kumesi (194 part/174 grup), AYNI corpus "
        "(zengin_parite_w2). Onceki gecelerin blok sayilari FARKLI "
        "tabanlardan geldigi for TOPLANAMAZ; this tablo toplanabilir.")},
        f ,indent =1 )
    print ("receipt -> results/p2_ablasyon.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
