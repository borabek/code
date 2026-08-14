# -*- coding: utf-8 -*-
"""R1: ROBOTUN ACI ARTIGI -- listenin last kalemi.

x1 teshisi: tespit edilen 873 ciftin %80.3'u two olcutu de geciyor.
    only YANAL dusen  %6.4
    only ACI  dusen   %8.0
    ikisi de dusen      %5.3
Uye havuzunun ACI kahini only +%2.3 birakiyor -> uye seciciyi already bosalttik.

BU BETIK: kalan angle hatalarinin NE OLDUGUNU soyler.
  1. Hata dagilimi: 10-30 derece mi (small deviation) otherwise ~90 derece mi (YANLIS EKSEN sinifi)?
  2. ~90 derece olanlar hafizadaki "manufacturer yonu silindir eksenine DIK" sinifi mi?
  3. Bu sinif TAHMIN EDILEBILIR mi (ozniteliklerden ayirt edilebiliyor mu)? Edilebiliyorsa
     ogrenilmis a "dik mi" kafasi yazilabilir; edilemiyorsa arm kapanir.

Ayrica ROBOT TAVANI'ni sayiyla gives: angle MUKEMMEL olsa robot kac olurdu, lateral da
mukemmel olsa kac olurdu.
"""
import io 
import json 

import numpy as np 

import gate_bench as T 


def main ():
    import measure_set 
    import wire_gate 
    from sklearn .ensemble import RandomForestClassifier 
    from sklearn .model_selection import GroupKFold 

    D =T .yukle ()
    measure_set .rapor_bas (D ["rap"])
    cfg =D ["cfg"]
    X ,y ,pid ,keep =D ["X"],D ["y"],D ["pid"],D ["keep"]
    Z =np .zeros ((len (X ),X .shape [1 ]*2 ))
    for u in np .unique (pid ):
        i =np .where (pid ==u )[0 ]
        Z [i ]=wire_gate .within_part (X [i ],D ["donusum"])
    gate ={"clf":RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (Z [keep ],y [keep ]),
    "n_feat":Z .shape [1 ],"donusum":D ["donusum"]}

    ACI ,YANAL ,FEAT ,GRUP ,RJ =[],[],[],[],[]
    for r in D ["DER"]:
        if r ["X"]is None or r .get ("XR")is None or not len (r ["G"]):
            continue 
        Xr =np .hstack ([r ["X"],r ["XR"]])
        k =wire_gate .decision_mask (wire_gate .decision_score (gate ,Xr ))
        if not k .any ():
            continue 
        P =r ["P"][k ].copy ();Pd =r ["Pd"][k ].copy ();Xk =Xr [k ]
        if cfg .get ("robot_pose_head"):
            c =[{"point":P [i ],"direction":Pd [i ]}for i in range (len (P ))]
            c =wire_gate .pose_correct (Xk ,c )
            if cfg .get ("robot_aci_secici"):
                c =wire_gate .angle_correct (Xk ,c )
            if cfg .get ("robot_uye_secici")and r .get ("UYE"):
                c =wire_gate .pick_member_direction (Xk ,c ,r ["UYE"])
            P =np .array ([x ["point"]for x in c ],float )
            Pd =np .array ([x ["direction"]for x in c ],float )
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        diff =P [:,None ,:]-G [None ,:,:]
        al =(diff *Gd [None ,:,:]).sum (-1 )
        pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
        pe =np .where (np .abs (al )<=40.0 ,pe ,np .inf )
        ac =np .degrees (np .arccos (np .clip (np .abs (Pd @Gd .T ),0 ,1 )))
        tol =max (3.0 ,0.06 *float (r ["diag"]))
        up ,ug =set (),set ()
        for d_ ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )for a in range (len (P ))
        for b in range (len (G ))):
            if not np .isfinite (d_ )or d_ >tol or a_ in up or b_ in ug :
                continue 
            up .add (a_ );ug .add (b_ )
            ACI .append (ac [a_ ,b_ ]);YANAL .append (pe [a_ ,b_ ])
            FEAT .append (Xk [a_ ]);GRUP .append (r ["geo"])
            RJ .append ("cok"if r ["n"]>=8 else "dusuk")
    ACI =np .array (ACI );YANAL =np .array (YANAL );FEAT =np .array (FEAT )
    GRUP =np .array (GRUP )
    print (f"\n{len (ACI )} tespit edilmis cift")

    print (f"\nACI HATASI DAGILIMI")
    kova =[(0 ,10 ),(10 ,30 ),(30 ,60 ),(60 ,80 ),(80 ,100 ),(100 ,181 )]
    for lo ,hi in kova :
        m =(ACI >=lo )&(ACI <hi )
        print (f"  {lo :>3}-{hi :<3} derece : {m .sum ():>4} ({m .mean ():>5.1%})")
    dik =(ACI >=60 )
    print (f"\n  ACI HATASI olan {int ((ACI >10 ).sum ())} ciftin {int (dik .sum ())}'i >=60 derece "
    f"(= {dik .sum ()/max ((ACI >10 ).sum (),1 ):.0%}) -> 'YANLIS EKSEN' sinifi")
    print (f"  kucuk deviation (10-30 derece): {int (((ACI >10 )&(ACI <30 )).sum ())}")

    print (f"\nTAVANLAR (tespit edilen ciftler uzerinden)")
    y_ok =YANAL <=2.0 ;a_ok =ACI <=10.0 
    print (f"  su an ikisi de gecen        {(y_ok &a_ok ).mean ():6.1%}")
    print (f"  ACI mukemmel olsa           {y_ok .mean ():6.1%}")
    print (f"  YANAL mukemmel olsa         {a_ok .mean ():6.1%}")
    print (f"  IKISI de mukemmel (=tespit) {1.0 :6.1%}")
    hh =T .f1w 
    print (f"\n  su anki robot F1 0.5893 | tespit F1 0.7584")
    print (f"  aci mukemmel -> robot ~ {0.7584 *y_ok .mean ():.4f}")
    print (f"  lateral mukemmel -> robot ~ {0.7584 *a_ok .mean ():.4f}")

    # --- 'YANLIS EKSEN' sinifi OZNITELIKLERDEN prediction edilebilir mi?
    hedef =(ACI >=60 ).astype (int )
    if hedef .sum ()>=20 :
        oof =np .zeros (len (hedef ))
        for tr ,te in GroupKFold (n_splits =5 ).split (FEAT ,hedef ,GRUP ):
            oof [te ]=RandomForestClassifier (n_estimators =300 ,min_samples_leaf =3 ,n_jobs =-1 ,
            random_state =0 ).fit (
            FEAT [tr ],hedef [tr ]).predict_proba (FEAT [te ])[:,1 ]
        from scipy .stats import rankdata 
        rk =rankdata (oof );n1 =hedef .sum ();n0 =len (hedef )-n1 
        auc =float ((rk [hedef ==1 ].sum ()-n1 *(n1 +1 )/2 )/(n1 *n0 ))
        print (f"\n'YANLIS EKSEN' sinifi ({int (hedef .sum ())} cift) grup-capraz AUC = {auc :.3f}")
        print (f"  KARAR: {'TAHMIN EDILEBILIR -> ogrenilmis dik-kafa yazilabilir'if auc >=0.70 else 'AYIRT EDILEMIYOR -> R1 KAPANIR'}")
    else :
        auc =None 
        print ("\n'YANLIS EKSEN' sinifi cok kucuk -> istatistik yok")
    with io .open ("results/r1_aci_artigi.json","w",encoding ="utf-8")as f :
        json .dump ({"n":len (ACI ),"aci_ok":float (a_ok .mean ()),"yanal_ok":float (y_ok .mean ()),
        "ikisi":float ((y_ok &a_ok ).mean ()),
        "yanlis_eksen_n":int ((ACI >=60 ).sum ()),
        "yanlis_eksen_auc":auc ,
        "tavan_aci_mukemmel":float (0.7584 *y_ok .mean ()),
        "tavan_yanal_mukemmel":float (0.7584 *a_ok .mean ())},f ,indent =1 )
    print ("receipt -> results/r1_aci_artigi.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
