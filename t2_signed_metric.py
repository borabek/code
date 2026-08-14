# -*- coding: utf-8 -*-
"""T2: FIZIKSEL (ISARETLI) ROBOT METRIGI -- direction sozlesmesini olc and ayir.

FINDING (dogrulandi): `sina_cluster.esle` aciyi `abs(Pd . Gd)` with oluyor -> 180 derece TERS a
prediction 0 derece sayiliyor. Ayrica robot_cp docstring'i yonu "telin girdigi direction" diye
tanimliyor but MEASURED: candidate yonleri govdeden DISARI bakiyor, ureticinin InsertDirection'i
ICERI. Yani belge with data celisiyor and metrik bunu goremiyor.

BU BETIK UC METRIGI AYIRIR:
    axis-hazir      lateral<=2mm VE angle<=10 derece, ISARETSIZ  (bugune up to raporlanan)
    takma-hazir(-)   same but ISARETLI, takma_yonu = -disari_normal
    takma-hazir(+)   same but ISARETLI, takma_yonu = +disari_normal

OKUMA:
  takma(-) ~ axis-hazir  -> sozlesme "-disari" DOGRU; metrik already fiziksel, only
                             ilan edilmemis. Duzeltme = alan adlandirmasi.
  takma(-) COKERSE        -> yonler candidate BASINA tutarsiz; robot hedefi yeniden tabanlanmali.

Hicbir model does not change; this a OLCUM ayristirmasidir.
"""
import io 
import json 

import numpy as np 

import gate_bench as T 


def match_signed (P ,Pd ,G ,Gd ,diag ,lm ,am ,sign ):
    """sina_cluster.esle with AYNI greedy, but angle ISARETLI olcülür.

    sign = 0  -> abs (mevcut davranis)
    sign = -1 -> takma yonu = -Pd  (disari normalin tersi)
    sign = +1 -> takma yonu = +Pd
    """
    if not len (P )or not len (G ):
        return 0 ,len (P ),len (G )
    tol =lm if lm >0 else max (3.0 ,0.06 *float (diag ))
    diff =P [:,None ,:]-G [None ,:,:]
    al =(diff *Gd [None ,:,:]).sum (-1 )
    pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
    D =Pd if sign >=0 else -Pd 
    c =D @Gd .T 
    an =np .degrees (np .arccos (np .clip (np .abs (c )if sign ==0 else c ,-1 ,1 )))
    pe =np .where ((np .abs (al )>40 )|(an >am ),np .inf ,pe )
    up ,ug ,tp =set (),set (),0 
    for d_ ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )for a in range (len (P ))for b in range (len (G ))):
        if not np .isfinite (d_ )or d_ >tol or a_ in up or b_ in ug :
            continue 
        up .add (a_ );ug .add (b_ );tp +=1 
    return tp ,len (P )-tp ,len (G )-tp 


def main ():
    import measure_set 
    import wire_gate 
    from sklearn .ensemble import RandomForestClassifier 

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

    KOL ={"axis-hazir (unsigned)":0 ,"takma-hazir (-disari)":-1 ,"takma-hazir (+disari)":+1 }
    det ={k :[]for k in KOL }
    ISARET =[]# candidate basina: -Pd mi +Pd mi GT with same yone bakiyor
    for r in D ["DER"]:
        P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
        if r ["X"]is not None and r .get ("XR")is not None :
            Xr =np .hstack ([r ["X"],r ["XR"]])
            k =wire_gate .decision_mask (wire_gate .decision_score (gate ,Xr ))
            if k .any ():
                P =r ["P"][k ].copy ();Pd =r ["Pd"][k ].copy ()
                if cfg .get ("robot_pose_head"):
                    c =[{"point":P [i ],"direction":Pd [i ]}for i in range (len (P ))]
                    c =wire_gate .pose_correct (Xr [k ],c )
                    if cfg .get ("robot_aci_secici"):
                        c =wire_gate .angle_correct (Xr [k ],c )
                    if cfg .get ("robot_uye_secici")and r .get ("UYE"):
                        c =wire_gate .pick_member_direction (Xr [k ],c ,r ["UYE"])
                    P =np .array ([x ["point"]for x in c ],float )
                    Pd =np .array ([x ["direction"]for x in c ],float )
        rj ="very"if r ["n"]>=8 else "dusuk"
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        for ad ,s in KOL .items ():
            det [ad ].append ((rj ,)+match_signed (P ,Pd ,G ,Gd ,r ["diag"],2.0 ,10.0 ,s ))
            # ISARET SAYIMI: matched ciftlerde -Pd mi +Pd mi GT'ye yakin
        if len (P )and len (G ):
            diff =P [:,None ,:]-G [None ,:,:]
            al =(diff *Gd [None ,:,:]).sum (-1 )
            pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
            pe =np .where (np .abs (al )>40 ,np .inf ,pe )
            up ,ug =set (),set ()
            for d_ ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )for a in range (len (P ))
            for b in range (len (G ))):
                if not np .isfinite (d_ )or d_ >2.0 or a_ in up or b_ in ug :
                    continue 
                up .add (a_ );ug .add (b_ )
                ISARET .append (float (Pd [a_ ]@Gd [b_ ]))

    print (f"\n{'arm':<26}{'robot F1':>10}")
    S ={}
    for ad in KOL :
        S [ad ]=T .f1w (det [ad ])
        print (f"{ad :<26}{S [ad ]:>10.4f}")

    I =np .array (ISARET )
    print (f"\nISARET DAGILIMI ({len (I )} matched cift, Pd . Gd):")
    print (f"  NEGATIF (Pd disari, Gd iceri -> takma = -Pd): {float ((I <0 ).mean ()):.1%}")
    print (f"  POZITIF (same yone bakiyor)                 : {float ((I >0 ).mean ()):.1%}")
    print (f"  medyan {np .median (I ):+.3f}")

    e =S ["axis-hazir (unsigned)"]
    m =S ["takma-hazir (-disari)"];p =S ["takma-hazir (+disari)"]
    en_iyi ="-disari"if m >=p else "+disari"
    loss =e -max (m ,p )
    print (f"\nHUKUM: correct sozlesme takma_yonu = {en_iyi }")
    print (f"  signed metrik {max (m ,p ):.4f} vs unsigned {e :.4f} -> loss {loss :+.4f}")
    if loss <=0.01 :
        print ("  -> Sozlesme TUTARLI. Metrik already fiziksel; missing which is only ILAN.")
    else :
        print ("  -> Yonler candidate BASINA TUTARSIZ. Robot hedefi signed metrikle YENIDEN tabanlanmali.")
    with io .open ("results/t2_signed.json","w",encoding ="utf-8")as f :
        json .dump ({k :float (v )for k ,v in S .items ()}|
        {"negatif_pay":float ((I <0 ).mean ()),"medyan_dot":float (np .median (I )),
        "dogru_sozlesme":en_iyi ,"loss":float (loss )},f ,indent =1 )
    print ("receipt -> results/t2_signed.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
