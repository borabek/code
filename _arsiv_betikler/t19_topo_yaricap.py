# -*- coding: utf-8 -*-
"""T19: topoloji yaricapi (R) taramasi -- R=6.0 mm KEYFI secilmisti.

`kon_cevre` = icbukey kenarlarin axis etrafindaki acisal kapsamasi, R yaricapi inside. R'nin
kendisi never taranmadi: very kucukse mouth cemberi kacar, very buyukse komsu deliklerin kenarlari
karisir. Aday duzeyinde ucuz taranir.

KILL (onceden yazili, BUYUKLUK dahil -- this gecenin dersi): a R, mevcut 6.0'i candidate-duzeyi
artimli F1'de EN AZ +0.01 gecmezse full corpus yeniden uretilmez. (Rejim-ratio kolu "artmali"
dedigim for +0.0018 with teknik as gecmisti; residual buyukluk sart.)
"""
import os ,sys ,json ,pickle 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
from t1_manufacturer_out import auc_mw 

R_LISTE =[3.0 ,4.5 ,6.0 ,8.0 ,12.0 ]


def main ():
    import cp_openings ,robot_cp ,wire_gate ,topo_feats 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from sklearn .ensemble import RandomForestClassifier 
    from sklearn .model_selection import GroupKFold 
    from big_arbiter import eligible 

    cfg =json .load (open ("cp_config.json",encoding ="utf-8"))
    pp =cfg ["prediction_postproc"]
    MINV =int (pp ["min_vertices"]);VC =float (pp ["vertex_confidence_mask"]);CL =float (pp ["cluster_mm"])
    ORAN =float (cfg ["gate_goreli_oran"]);TABAN =float (cfg ["gate_goreli_taban"])
    mfg_of ={p :m for m ,p ,jf ,s in eligible ()}

    X18 ,XR ,Y ,GRP ,MF =[],{r :[]for r in R_LISTE },[],[],[]
    for cluster in ("dev","val"):
        cf =f"results/_probs_{cluster }.pkl"
        if not os .path .exists (cf )and cluster =="dev":
            cf ="results/_h_probs.pkl"
        cache =pickle .load (open (cf ,"rb"))
        for r in cache :
            r .setdefault ("pid",os .path .basename (r ["stp"]).split ("_")[1 ])
        for i ,r in enumerate (cache ,1 ):
            if i %50 ==0 :
                print (f"  {cluster } {i }/{len (cache )}",flush =True )
            V =np .ascontiguousarray (r ["V"],np .float64 )
            F =np .ascontiguousarray (r ["F"],np .int64 )
            plist =[np .asarray (pb ,np .float64 )for pb in r ["pbs"]]
            mk =lambda pr_ ,**kw :cp_openings .connection_points (
            V ,F ,pr_ .argmax (-1 ),min_v =MINV ,classes =(CE ,CT ),dedupe_mm =10.0 ,
            probs =pr_ ,vertex_conf =VC ,ct_depth_min_mm =1.0 ,cluster_mm =CL ,
            step_path =r ["stp"],**kw )
            merge =lambda L :robot_cp ._vote2 (L ,min_votes =1 )
            base =merge ([mk (pb )for pb in plist ])
            is_hi =robot_cp ._highcp_router (r ["stp"],V ,len (base ))
            cps =merge ([mk (pb ,conn_promote =0.25 )for pb in plist ])if is_hi else base 
            if not cps :
                continue 
            Xg =wire_gate .feats_for (V ,F ,sum (plist )/len (plist ),cps ,CE ,CT ,
            step_path =r ["stp"])[:,:18 ]
            onb =topo_feats .icbukey_kenarlar (V ,F )# edge yapisi BIR times
            P =np .array ([c ["point"]for c in cps ],float )
            D =np .array ([c ["direction"]for c in cps ],float )
            G ,Gd =r ["G"],r ["Gd"]
            lab =np .zeros (len (P ),bool )
            if len (G ):
                diff =P [:,None ,:]-G [None ,:,:]
                a_ =(diff *Gd [None ,:,:]).sum (-1 )
                pe =np .linalg .norm (diff -a_ [...,None ]*Gd [None ,:,:],axis =-1 )
                pe =np .where (np .abs (a_ )<=40.0 ,pe ,np .inf )
                t0 =max (3.0 ,0.06 *r ["diag"]);us ,ug =set (),set ()
                for dv ,x ,y_ in sorted ((pe [a ,b ],a ,b )
                for a in range (len (P ))for b in range (len (G ))):
                    if dv >t0 or x in us or y_ in ug :
                        continue 
                    us .add (x );ug .add (y_ );lab [x ]=True 
            for k in range (len (P )):
                X18 .append (Xg [k ]);Y .append (bool (lab [k ]));GRP .append (r ["pid"])
                MF .append (mfg_of .get (r ["pid"],"?"))
                for R in R_LISTE :
                    XR [R ].append (topo_feats .topo_ozellik (V ,F ,P [k ],D [k ],R =R ,cache =onb ))
    X18 =np .array (X18 ,float );Y =np .array (Y ,bool )
    GRP =np .array (GRP );MF =np .array (MF )
    for R in R_LISTE :
        XR [R ]=np .array (XR [R ],float )
    print (f"\n{len (Y )} candidate | TP {int (Y .sum ())} | {len (set (GRP ))} part")

    def olc (M ):
        o =np .zeros (len (Y ))
        for tr ,te in GroupKFold (n_splits =5 ).split (M ,Y ,GRP ):
            o [te ]=RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
            random_state =0 ).fit (M [tr ],Y [tr ]).predict_proba (M [te ])[:,1 ]
        m =np .zeros (len (o ),bool )
        for u in np .unique (GRP ):
            i =GRP ==u ;v =o [i ]
            m [i ]=(v >=ORAN *max (v .max (),1e-9 ))&(v >=TABAN )
        tp =int ((Y &m ).sum ());fp =int ((~Y &m ).sum ());fn =int ((Y &~m ).sum ())
        p_ =tp /max (tp +fp ,1 );r_ =tp /max (tp +fn ,1 )
        return 2 *p_ *r_ /max (p_ +r_ ,1e-9 )

    print (f"\n{'R (mm)':<10}{'kon_cevre AUC':>15}{'TP med':>9}{'FP med':>9}{'18+4 F1':>10}{'difference':>9}")
    taban_f1 =olc (X18 )
    print (f"{'(topolojisiz)':<10}{'-':>15}{'-':>9}{'-':>9}{taban_f1 :>10.4f}{'':>9}")
    out ={}
    ref =None 
    for R in R_LISTE :
        a =auc_mw (XR [R ][:,3 ],Y )
        f =olc (np .hstack ([X18 ,XR [R ]]))
        if R ==6.0 :
            ref =f 
        print (f"{R :<10.1f}{a :>15.3f}{np .median (XR [R ][Y ,3 ]):>9.3f}"
        f"{np .median (XR [R ][~Y ,3 ]):>9.3f}{f :>10.4f}"
        f"{(f -ref )if ref is not None else 0 :>+9.4f}",flush =True )
        out [str (R )]={"kon_cevre_auc":float (a ),"f1":float (f )}
    en_iyi =max (out ,key =lambda k :out [k ]["f1"])
    fark =out [en_iyi ]["f1"]-out ["6.0"]["f1"]
    print (f"\nEN IYI R = {en_iyi } | mevcut 6.0'a gore {fark :+.4f}")
    print (f"KILL (BUYUKLUK dahil): >= +0.01 gelmezse tam corpus YENIDEN URETILMEZ -> "
    f"{'URET'if fark >=0.01 else 'URETME'}")
    json .dump (out |{"en_iyi":en_iyi ,"difference":float (fark )},
    open ("results/t19_topo_yaricap.json","w"),indent =1 )
    print ("receipt -> results/t19_topo_yaricap.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
