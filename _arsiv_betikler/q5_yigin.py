# -*- coding: utf-8 -*-
"""Q5: ALT-ESIK KOLLARINI YIGIN OLARAK OLC (bari indirmek instead of).

Elimde two ALT-ESIK arm present, ikisi de real but single baslarina +0.02'yi gecmiyor:
    ACI SECICI      robot  +0.0126  GA [+0.0002, +0.0325]   tespit bedeli 0.0000
    DUR/DEVAM KESIM tespit +0.0150  5/5 seed, GA sifiri kil payi iceriyor

Hafizadaki kayit ([[stacked-levers-2026-07-29]]) this state for net: bari INDIRME, kollari
YIGIN as uygula and yigini olc. Ikisi FARKLI metrige and different asamaya dokunuyor
(biri direction, digeri kesim) -- ortusme beklenmez but olculmeden varsayilmaz.

KOLLAR:
  A  dagitilan urun (zengin gate + pose head)
  B  + angle selector
  C  + dur/devam kesim
  D  + ikisi (YIGIN)

KILL (yigin for, onceden yazili): robot >= +0.02 VEYA tespit >= +0.02, digerinde loss
< 0.005, and kazanan metrigin GA'si sifiri dislamali.
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


def yerel (d ):
    d =np .asarray (d ,float );d =d /(np .linalg .norm (d )+1e-9 )
    a =np .array ([1.0 ,0.0 ,0.0 ])
    if abs (float (d @a ))>0.9 :
        a =np .array ([0.0 ,1.0 ,0.0 ])
    u =np .cross (d ,a );u /=np .linalg .norm (u )+1e-9 
    return d ,u ,np .cross (d ,u )


def kesim_ozellik (s ,diag ,is_hi ):
    v =np .sort (np .asarray (s ,float ))[::-1 ]
    n =len (v )
    pad =lambda i :float (v [i ])if i <n else 0.0 
    dif =np .diff (v )*-1.0 if n >1 else np .array ([0.0 ])
    return [float (n ),float (v .max ()),float (v .mean ()),float (v .std ()),float (np .median (v )),
    pad (0 )-pad (1 ),pad (0 )-pad (2 ),pad (1 )-pad (2 ),
    float ((v >=0.25 ).sum ()),float ((v >=0.35 ).sum ()),
    float ((v >=0.50 ).sum ()),float ((v >=0.70 ).sum ()),
    float (dif .max ()),float (int (np .argmax (dif ))+1 )if n >1 else 1.0 ,
    float (diag ),float (bool (is_hi ))]


def main ():
    import measure_set 
    import wire_gate 
    from big_arbiter import eligible 
    from sina_cluster import esle ,f1w 
    from sklearn .ensemble import RandomForestClassifier ,RandomForestRegressor 

    mfg_of ={p :m for m ,p ,jf ,s in eligible ()}
    DER ,rap =measure_set .cluster ("results/_der_zengin.pkl")
    for r in DER :
        r ["mfg"]=mfg_of .get (r ["pid"],"?")
    gk =measure_set .geo_anahtarlari ()
    tg ={r ["geo"]for r in DER }
    g =[x ["geo"]for x in DER ]

    zen =np .load ("results/zengin_parite.npz",allow_pickle =True )
    Xt =np .hstack ([np .asarray (zen ["X22"],float ),np .asarray (zen ["XR"],float )])
    ytr =np .asarray (zen ["y"]);tpid =np .array ([str (x )for x in zen ["pids"]])
    tgrp =np .array ([gk .get (p ,"yok:"+p )for p in tpid ]);keep =~np .isin (tgrp ,list (tg ))
    dag =wire_gate ._load (wire_gate .MODEL_PATH );DON =dag .get ("donusum")
    Z =np .zeros ((len (Xt ),Xt .shape [1 ]*2 ))
    for u in np .unique (tpid ):
        i =np .where (tpid ==u )[0 ]
        Z [i ]=wire_gate .within_part (Xt [i ],DON )
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (Z [keep ],ytr [keep ])
    gate ={"clf":clf ,"n_feat":Z .shape [1 ],"donusum":DON }

    pv =np .load ("results/pose_veri.npz",allow_pickle =True )
    RX =np .asarray (pv ["X"],float );RY =np .asarray (pv ["Y"],float )
    RG =np .array ([str (x )for x in pv ["geo"]])
    disi =~np .isin (RG ,list (tg ))
    RX ,RY =RX [disi ],RY [disi ]
    aci =np .degrees (np .arcsin (np .clip (np .linalg .norm (RY [:,2 :],axis =1 ),0 ,1 )))
    sec =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =5 ,n_jobs =-1 ,
    random_state =0 ).fit (RX ,(aci >10.0 ).astype (int ))
    reg =RandomForestRegressor (n_estimators =400 ,min_samples_leaf =5 ,n_jobs =-1 ,
    random_state =0 ).fit (RX ,RY )
    print (f"aci selector + regresor hazir ({len (RY )} satir)",flush =True )

    # --- DUR/DEVAM kesim modeli: training korpusunun KENDI skorlarindan
    oof_pid =tpid [keep ]
    sc =clf .predict_proba (Z [keep ])[:,1 ]
    KX ,KY ,KG =[],[],[]
    for u in np .unique (oof_pid ):
        i =np .where (oof_pid ==u )[0 ]
        v =sc [i ];o =np .argsort (-v );vs =v [o ];ys =ytr [keep ][i ][o ]
        # kahin K: F1'i at most yapan onek
        en ,enk =-1 ,0 
        for K in range (0 ,len (vs )+1 ):
            tp =int (ys [:K ].sum ());fp =K -tp ;fn =int (ys .sum ())-tp 
            f =2 *tp /max (2 *tp +fp +fn ,1 )
            if f >en :
                en ,enk =f ,K 
        kum =0.0 
        for k2 in range (len (vs )):
            kum +=float (vs [k2 ])
            KX .append ([float (vs [k2 ]),float (vs [k2 ]/max (vs [0 ],1e-9 )),float (k2 +1 ),
            float ((k2 +1 )/len (vs )),
            float (vs [k2 -1 ]-vs [k2 ])if k2 >0 else 0.0 ,
            float (vs [k2 ]-vs [k2 +1 ])if k2 +1 <len (vs )else 0.0 ,
            float (kum ),float (len (vs )),float (vs [0 ]),float (vs .mean ()),
            float (vs .std ()),0.0 ,0.0 ])
            KY .append (1 if (k2 +1 )<=enk else 0 )
            KG .append (u )
    KX =np .array (KX ,float );KY =np .array (KY )
    kes =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =5 ,n_jobs =-1 ,
    random_state =0 ).fit (KX ,KY )
    print (f"kesim modeli hazir ({len (KY )} satir)",flush =True )

    def puanla (aci_on =False ,kes_on =False ):
        det ,rob =[],[]
        for r in DER :
            P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
            if r ["X"]is not None and r .get ("XR")is not None :
                X58 =np .hstack ([r ["X"],r ["XR"]])
                s =wire_gate .decision_score (gate ,X58 )
                k =wire_gate .decision_mask (s )
                if kes_on and k .any ():
                    v =np .asarray (s ,float );o =np .argsort (-v );vs =v [o ]
                    kum =0.0 ;feats =[]
                    for k2 in range (len (vs )):
                        kum +=float (vs [k2 ])
                        feats .append ([float (vs [k2 ]),float (vs [k2 ]/max (vs [0 ],1e-9 )),
                        float (k2 +1 ),float ((k2 +1 )/len (vs )),
                        float (vs [k2 -1 ]-vs [k2 ])if k2 >0 else 0.0 ,
                        float (vs [k2 ]-vs [k2 +1 ])if k2 +1 <len (vs )else 0.0 ,
                        float (kum ),float (len (vs )),float (vs [0 ]),
                        float (vs .mean ()),float (vs .std ()),0.0 ,0.0 ])
                    pr =kes .predict_proba (np .array (feats ,float ))[:,1 ]
                    K =0 
                    while K <len (pr )and pr [K ]>=0.60 :
                        K +=1 
                    yeni =np .zeros (len (v ),bool )
                    if K :
                        yeni [o [:K ]]=True 
                        # YIGIN KURALI: kesim only DARALTIR (mevcut kabulun lower kumesi)
                    k =k &yeni if yeni .any ()else k 
                if k .any ():
                    P =r ["P"][k ].copy ();Pd =r ["Pd"][k ].copy ()
                    cps =[{"point":P [i ],"direction":Pd [i ]}for i in range (len (P ))]
                    cps =wire_gate .pose_correct (X58 [k ],cps )
                    P =np .array ([c ["point"]for c in cps ],float )
                    if aci_on :
                        pr =reg .predict (X58 [k ]);ps =sec .predict_proba (X58 [k ])[:,1 ]
                        for i in range (len (P )):
                            if ps [i ]<0.10 :
                                continue 
                            d ,u ,v2 =yerel (Pd [i ])
                            gg =d +float (pr [i ,2 ])*u +float (pr [i ,3 ])*v2 
                            Pd [i ]=gg /(np .linalg .norm (gg )+1e-9 )
            rj ="cok"if r ["n"]>=8 else "dusuk"
            det .append ((rj ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],0.0 ,180.0 ,True ))
            rob .append ((rj ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],2.0 ,10.0 ,False ))
        return det ,rob 

    KOL ={"A dagitilan":(False ,False ),"B +aci selector":(True ,False ),
    "C +kesim":(False ,True ),"D YIGIN (ikisi)":(True ,True )}
    print (f"\n{'arm':<20}{'tespit':>10}{'ROBOT':>10}{'d_tespit':>10}{'d_robot':>10}")
    SON ,PAR ={},{}
    for ad ,(a_ ,k_ )in KOL .items ():
        det ,rob =puanla (a_ ,k_ )
        SON [ad ]=(float (f1w (det )),float (f1w (rob )))
        PAR [ad ]=(det ,rob )
        b =SON ["A dagitilan"]
        print (f"{ad :<20}{SON [ad ][0 ]:>10.4f}{SON [ad ][1 ]:>10.4f}"
        f"{SON [ad ][0 ]-b [0 ]:>+10.4f}{SON [ad ][1 ]-b [1 ]:>+10.4f}",flush =True )

    b =SON ["A dagitilan"]
    print ("\n=== GA (grup bootstrap) ===")
    OUT ={}
    for ad in KOL :
        if ad .startswith ("A"):
            continue 
        for mi ,ad2 in ((1 ,"robot"),(0 ,"tespit")):
            cift =list (zip (PAR ["A dagitilan"][mi ],PAR [ad ][mi ]))
            fn =lambda rows :f1w ([y for _ ,y in rows ])-f1w ([x for x ,_ in rows ])
            _ ,lo ,hi =measure_set .grup_bootstrap (cift ,g ,fn ,n =2000 )
            OUT [f"{ad }|{ad2 }"]=[float (lo ),float (hi )]
            print (f"  {ad :<20}{ad2 :<8}[{lo :+.4f}, {hi :+.4f}] "
            f"{'GERCEK'if (lo >0 or hi <0 )else 'noise'}")
    with io .open ("results/q5_yigin.json","w",encoding ="utf-8")as f :
        json .dump ({"kollar":{k :list (v )for k ,v in SON .items ()},"ga":OUT },f ,indent =1 )
    print ("receipt -> results/q5_yigin.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
