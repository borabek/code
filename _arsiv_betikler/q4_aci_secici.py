# -*- coding: utf-8 -*-
"""Q4: ACI duzeltmesini SECICI yap -- robot-hazir 0.60'a giden last arm.

POSE SONRASI AYRISTIRMA (867 eslesme):
    lateral gecis  %74.8 -> %88.9   (pose head calisti)
    angle   gecis  %81.4 -> %80.4   (degismedi)
    only angle'dan kaybedilen %14.2 | only yanaldan %5.7
Yani BAGLAYICI SART residual ACI. Ve aritmetik: angle tamamen cozulse gecis %88.9 -> robot ~0.60.

ACI duzeltmesi why more before KAYBETTIRDI (q1/q2): yonlerin most ZATEN TAM DOGRU (medyan
0.00 derece) and correction HEPSINE uygulaniyordu -> medyan 0.00 -> 2.36 dereceye output.
Bu, this gecenin four olu kolunun ortak kusuru.

COZUM (this gece IKI KEZ whereas yaradi -- cokus yonlendirmesi and yarik secicisi): before
"BU ACIYA DOKUNULMALI MI" diye sor, after duzelt.

  1. SINIFLANDIRICI: "this adayin acisi 10 dereceden extra wrong mi?"  (manufacturer GT'sinden)
  2. Yalniz P(wrong) >= threshold olanlarda direction duzeltmesi uygulanir
  3. Esik taranir; selector no DOGRU yonu bozmamali

KILL (onceden yazili): robot-hazir >= +0.02 VE tespit kaybi < 0.005 VE GA sifiri dislamali.
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

    # POSE verisi -> angle hedefi + "wrong mi" etiketi
    pv =np .load ("results/pose_veri.npz",allow_pickle =True )
    RX =np .asarray (pv ["X"],float );RY =np .asarray (pv ["Y"],float )
    RG =np .array ([str (x )for x in pv ["geo"]])
    disi =~np .isin (RG ,list (tg ))
    RX ,RY ,RG =RX [disi ],RY [disi ],RG [disi ]
    aci =np .degrees (np .arcsin (np .clip (np .linalg .norm (RY [:,2 :],axis =1 ),0 ,1 )))
    yanlis =(aci >10.0 ).astype (int )
    print (f"aci selector egitimi: {len (RY )} satir | yanlis ratio {yanlis .mean ():.1%} "
    f"| {len (set (RG ))} grup",flush =True )

    sec =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =5 ,n_jobs =-1 ,
    random_state =0 ).fit (RX ,yanlis )
    reg =RandomForestRegressor (n_estimators =400 ,min_samples_leaf =5 ,n_jobs =-1 ,
    random_state =0 ).fit (RX ,RY )
    pose =wire_gate ._load (wire_gate .POSE_PATH )

    def yerel (d ):
        d =np .asarray (d ,float );d =d /(np .linalg .norm (d )+1e-9 )
        a =np .array ([1.0 ,0.0 ,0.0 ])
        if abs (float (d @a ))>0.9 :
            a =np .array ([0.0 ,1.0 ,0.0 ])
        u =np .cross (d ,a );u /=np .linalg .norm (u )+1e-9 
        return d ,u ,np .cross (d ,u )

    def puanla (aci_esik =None ):
        det ,rob =[],[]
        for r in DER :
            P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
            if r ["X"]is not None and r .get ("XR")is not None :
                X58 =np .hstack ([r ["X"],r ["XR"]])
                s =wire_gate .decision_score (gate ,X58 )
                k =wire_gate .decision_mask (s )
                if k .any ():
                    P =r ["P"][k ].copy ();Pd =r ["Pd"][k ].copy ()
                    cps =[{"point":P [i ],"direction":Pd [i ]}for i in range (len (P ))]
                    cps =wire_gate .pose_correct (X58 [k ],cps )
                    P =np .array ([c ["point"]for c in cps ],float )
                    if aci_esik is not None :
                        pr =reg .predict (X58 [k ]);ps =sec .predict_proba (X58 [k ])[:,1 ]
                        for i in range (len (P )):
                            if ps [i ]<aci_esik :
                                continue 
                            d ,u ,v =yerel (Pd [i ])
                            g =d +float (pr [i ,2 ])*u +float (pr [i ,3 ])*v 
                            Pd [i ]=g /(np .linalg .norm (g )+1e-9 )
            rj ="very"if r ["n"]>=8 else "low"
            det .append ((rj ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],0.0 ,180.0 ,True ))
            rob .append ((rj ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],2.0 ,10.0 ,False ))
        return det ,rob 

    g =[x ["geo"]for x in DER ]
    print (f"\n{'arm':<22}{'tespit':>10}{'ROBOT':>10}{'d_robot':>10}")
    det0 ,rob0 =puanla (None )
    t0 ,r0 =f1w (det0 ),f1w (rob0 )
    print (f"{'A angle duzeltmesi absent':<22}{t0 :>10.4f}{r0 :>10.4f}{0.0 :>+10.4f}")
    SON ={"A":(float (t0 ),float (r0 ))}
    PAR ={"A":(det0 ,rob0 )}
    for e in (0.05 ,0.10 ,0.15 ,0.20 ,0.25 ,0.30 ):
        det ,rob =puanla (e )
        SON [f"B threshold {e :.2f}"]=(float (f1w (det )),float (f1w (rob )))
        PAR [f"B threshold {e :.2f}"]=(det ,rob )
        print (f"{'B selector threshold '+f'{e :.2f}':<22}{f1w (det ):>10.4f}{f1w (rob ):>10.4f}"
        f"{f1w (rob )-r0 :>+10.4f}",flush =True )

    en =max ((k for k in SON if k !="A"),key =lambda k :SON [k ][1 ])
    dr =SON [en ][1 ]-r0 ;dt =SON [en ][0 ]-t0 
    cift =list (zip (PAR ["A"][1 ],PAR [en ][1 ]))
    fn =lambda rows :f1w ([y for _ ,y in rows ])-f1w ([x for x ,_ in rows ])
    _ ,lo ,hi =measure_set .grup_bootstrap (cift ,g ,fn ,n =2000 )
    print (f"\nEN IYI: {en } | robot {dr :+.4f} [{lo :+.4f}, {hi :+.4f}] | tespit {dt :+.4f}")
    gecti =dr >=0.02 and dt >-0.005 and lo >0 
    print (f"KILL: robot >= +0.02 VE tespit kaybi < 0.005 VE GA sifiri dislamali -> "
    f"{'GECTI'if gecti else 'GECMEDI'}")
    with io .open ("results/q4_aci_secici.json","w",encoding ="utf-8")as f :
        json .dump ({"kollar":{k :list (v )for k ,v in SON .items ()},"en_iyi":en ,
        "d_robot":float (dr ),"d_tespit":float (dt ),
        "ga":[float (lo ),float (hi )],"gecti":bool (gecti )},f ,indent =1 )
    if gecti :
        with open ("results/aci_secici.pkl","wb")as f :
            pickle .dump ({"sec":sec ,"reg":reg ,"threshold":float (en .split ()[-1 ]),
            "n_feat":RX .shape [1 ],
            "note":("ACI SECICI 2026-08-02: before 'this angle >10 derece wrong mi' "
            "diye sorar, only oyleyse yonu fixes. Duzeltmeyi HERKESE "
            "uygulamak already correct which is yonleri bozuyordu "
            "(medyan 0.00 -> 2.36 derece).")},f )
        print ("-> results/aci_secici.pkl")
    print ("receipt -> results/q4_aci_secici.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
