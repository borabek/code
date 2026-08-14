# -*- coding: utf-8 -*-
"""Q6: POSE duzeltmesini YINELE (duzelt -> ozellikleri yeniden hesapla -> tekrar duzelt).

Pose head lateral gecisi %74.8 -> %88.9 yapti. Kalan %11.1'in a kismi, TEK ADIMDA full
duzeltilemeyecek up to large sapmalar may be: kafa duzeltmeyi MEVCUT konumun ozelliklerinden
prediction ediyor; point duzeldikce ozellikler de degisir and ikinci a prediction more isabetli may be.

Ayni fikir axis olcumunde ISE YARAMISTI (cp_openings: "IKI TUR: rafine edilen direction new seed
becomes... 1 kind 0.4904 -> 2 kind 0.4955, 3./4. turda DEGISMIYOR").

MALIYET: each kind ozelliklerin YENIDEN hesaplanmasini gerektirir (zengin blok mesh'e bagli).
Bu yuzden here YAKLASIK a yineleme olculuyor: same ozelliklerle ikinci times prediction
(ozellikler degismedigi for correction AYNI cikar) YERINE, correction buyuklugunu OLCEKLEYEREK
"missing correction" hipotezi sinaniyor.

  A  correction absent
  B  1.0x (dagitilan)
  C  1.25x / 1.5x / 2.0x  -- kafa SISTEMATIK as AZ mi duzeltiyor?

RF regresyonu ortalamaya kacar (shrinkage), i.e. sistematik as AZ duzeltmesi BEKLENIR.
Olcekleme bunu telafi eder. Kill: robot >= +0.01 ek kazanc VE GA sifiri dislamali.
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


def main ():
    import measure_set 
    import wire_gate 
    from big_arbiter import eligible 
    from sina_cluster import esle ,f1w 
    from sklearn .ensemble import RandomForestClassifier ,RandomForestRegressor 

    mfg_of ={p :m for m ,p ,jf ,s in eligible ()}
    DER ,rap =measure_set .cluster ("results/_der_zengin.pkl")
    gk =measure_set .geo_anahtarlari ();tg ={r ["geo"]for r in DER }
    g =[x ["geo"]for x in DER ]
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

    pv =np .load ("results/pose_veri.npz",allow_pickle =True )
    RX =np .asarray (pv ["X"],float );RY =np .asarray (pv ["Y"],float )
    RG =np .array ([str (x )for x in pv ["geo"]])
    disi =~np .isin (RG ,list (tg ))
    RX ,RY =RX [disi ],RY [disi ]
    reg =RandomForestRegressor (n_estimators =400 ,min_samples_leaf =5 ,n_jobs =-1 ,
    random_state =0 ).fit (RX ,RY )
    aci =np .degrees (np .arcsin (np .clip (np .linalg .norm (RY [:,2 :],axis =1 ),0 ,1 )))
    sec =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =5 ,n_jobs =-1 ,
    random_state =0 ).fit (RX ,(aci >10.0 ).astype (int ))

    # SHRINKAGE OLCUMU: tahminin buyuklugu real sapmanin buyuklugune according to ne up to small?
    pr_tr =reg .predict (RX )
    ger =np .linalg .norm (RY [:,:2 ],axis =1 );tah =np .linalg .norm (pr_tr [:,:2 ],axis =1 )
    print (f"shrinkage: gercek deviation medyan {np .median (ger ):.3f}mm | tahmin medyan "
    f"{np .median (tah ):.3f}mm | ratio {np .median (tah )/max (np .median (ger ),1e-9 ):.3f}",
    flush =True )

    def yerel (d ):
        d =np .asarray (d ,float );d =d /(np .linalg .norm (d )+1e-9 )
        a =np .array ([1.0 ,0.0 ,0.0 ])
        if abs (float (d @a ))>0.9 :
            a =np .array ([0.0 ,1.0 ,0.0 ])
        u =np .cross (d ,a );u /=np .linalg .norm (u )+1e-9 
        return d ,u ,np .cross (d ,u )

    def puanla (olcek ):
        det ,rob =[],[]
        for r in DER :
            P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
            if r ["X"]is not None and r .get ("XR")is not None :
                X58 =np .hstack ([r ["X"],r ["XR"]])
                s =wire_gate .decision_score (gate ,X58 )
                k =wire_gate .decision_mask (s )
                if k .any ():
                    P =r ["P"][k ].copy ();Pd =r ["Pd"][k ].copy ()
                    pr =reg .predict (X58 [k ]);ps =sec .predict_proba (X58 [k ])[:,1 ]
                    for i in range (len (P )):
                        d ,u ,v =yerel (Pd [i ])
                        if olcek >0 :
                            dw =(float (pr [i ,0 ])*u +float (pr [i ,1 ])*v )*olcek 
                            n =float (np .linalg .norm (dw ))
                            if n >1e-9 :
                                P [i ]=P [i ]+dw *(min (n ,3.0 )/n )
                        if ps [i ]>=0.10 :
                            gg =d +float (pr [i ,2 ])*u +float (pr [i ,3 ])*v 
                            Pd [i ]=gg /(np .linalg .norm (gg )+1e-9 )
            rj ="very"if r ["n"]>=8 else "low"
            det .append ((rj ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],0.0 ,180.0 ,True ))
            rob .append ((rj ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],2.0 ,10.0 ,False ))
        return det ,rob 

    print (f"\n{'scale':<10}{'detection':>10}{'ROBOT':>10}{'d_robot':>10}")
    SON ,PAR ={},{}
    for o in (0.0 ,1.0 ,1.25 ,1.5 ,2.0 ):
        det ,rob =puanla (o )
        SON [o ]=(float (f1w (det )),float (f1w (rob )))
        PAR [o ]=(det ,rob )
        print (f"{o :<10.2f}{SON [o ][0 ]:>10.4f}{SON [o ][1 ]:>10.4f}{SON [o ][1 ]-SON [1.0 ][1 ]:>+10.4f}"
        if 1.0 in SON else 
        f"{o :<10.2f}{SON [o ][0 ]:>10.4f}{SON [o ][1 ]:>10.4f}",flush =True )

    baseline =SON [1.0 ][1 ]
    en =max ((o for o in SON if o not in (0.0 ,1.0 )),key =lambda o :SON [o ][1 ])
    dr =SON [en ][1 ]-baseline 
    cift =list (zip (PAR [1.0 ][1 ],PAR [en ][1 ]))
    fn =lambda rows :f1w ([y for _ ,y in rows ])-f1w ([x for x ,_ in rows ])
    _ ,lo ,hi =measure_set .grup_bootstrap (cift ,g ,fn ,n =2000 )
    print (f"\nEN IYI OLCEK: {en } | robot {dr :+.4f} [{lo :+.4f}, {hi :+.4f}]")
    gecti =dr >=0.01 and lo >0 
    print (f"KILL: ek kazanc >= +0.01 VE GA sifiri dislamali -> {'GECTI'if gecti else 'GECMEDI'}")
    with io .open ("results/q6_pose_yineleme.json","w",encoding ="utf-8")as f :
        json .dump ({"kollar":{str (k ):list (v )for k ,v in SON .items ()},"en_iyi_olcek":float (en ),
        "d_robot":float (dr ),"ga":[float (lo ),float (hi )],"gecti":bool (gecti )},
        f ,indent =1 )
    print ("receipt -> results/q6_pose_yineleme.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
