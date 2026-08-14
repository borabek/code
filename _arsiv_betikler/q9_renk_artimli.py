# -*- coding: utf-8 -*-
"""Q9: RENK, 18 ozellige (13 mevcut + 5 fiziksel) ARTIMLI a sey katiyor mu?

Q8 OLCTU (62 order-dogrulanmis part, 1038 candidate): `metal_mesafe` AUC 0.639 (ters yonde),
`c_govde` 0.622 -- ikisi de null'un ten. Yani "renk ayirmaz" varsayimim YANLISTI.
Ama two uyari:
  * adayin UZERINDE durdugu silindir neredeyse never metal not (c_metal AUC 0.496, OLU) --
    i.e. "kanalin dibinde metal" sinyali dogrudan not, UZAKLIK uzerinden geliyor.
  * `c_govde` large olcude "renk cozulebildi mi" vekili may be (c_metal ~0 oldugundan
    c_metal + c_govde ~= renk present mi). Tek basina AUC yeterli evidence DEGIL.

Bu yuzden single criterion ARTIMLI degerdir: 18 feature vs 18+3 renk, AYNI adaylarda, grup-capraz OOF.

KILL (onceden yazili): candidate-duzeyi OOF F1 artmazsa RENK listeye GIRMEZ. Artarsa madde as
eklenir and uctan uca olcume alinir.
"""
import os ,sys ,json ,pickle 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
from q7_colour_degeri import auc_mw 


def main ():
    import cp_openings ,robot_cp ,wire_gate 
    import step_face_colors as SC 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from sklearn .ensemble import RandomForestClassifier 
    from sklearn .model_selection import GroupKFold 
    from j_position_mean import vote_avg 

    assert len (wire_gate .FEAT_NAMES )==18 
    cfg =json .load (open ("cp_config.json",encoding ="utf-8"))
    pp =cfg ["prediction_postproc"]
    MINV =int (pp ["min_vertices"]);VC =float (pp ["vertex_confidence_mask"]);CL =float (pp ["cluster_mm"])
    THR ={"low":float (cfg ["robot_wire_gate_threshold"]),
    "very":float (cfg ["robot_wire_gate_threshold_highcp"])}
    nmax =int (sys .argv [1 ])if len (sys .argv )>1 else 120 

    parts =[]
    for cluster in ("dev","val"):
        cf =f"results/_probs_{cluster }.pkl"
        if not os .path .exists (cf )and cluster =="dev":
            cf ="results/_h_probs.pkl"
        for r in pickle .load (open (cf ,"rb")):
            r .setdefault ("pid",os .path .basename (r ["stp"]).split ("_")[1 ])
            parts .append (r )
    rng =np .random .RandomState (0 )
    parts =[parts [i ]for i in rng .choice (len (parts ),min (nmax ,len (parts )),replace =False )]

    X18 ,XC ,Y ,GRP ,HI =[],[],[],[],[]
    kabul =red =0 
    for r in parts :
        try :
            cy ,ok =SC .read_by_order (r ["stp"])
        except Exception :
            red +=1 ;continue 
        cyl =[f for f in cy if ok and f ["axis"]is not None and f ["rgb"]is not None ]
        if len (cyl )<2 :
            red +=1 ;continue 
        kabul +=1 
        mcom =np .array ([f ["com"]for f in cyl if f ["is_metal"]],float )
        C =np .array ([f ["com"]for f in cyl ],float )
        A =np .array ([f ["axis"]/(np .linalg .norm (f ["axis"])+1e-12 )for f in cyl ],float )
        MET =np .array ([f ["is_metal"]for f in cyl ],bool )

        V =np .ascontiguousarray (r ["V"],np .float64 )
        F =np .ascontiguousarray (r ["F"],np .int64 )
        plist =[np .asarray (pb ,np .float64 )for pb in r ["pbs"]]
        mk =lambda pr_ ,**kw :cp_openings .connection_points (
        V ,F ,pr_ .argmax (-1 ),min_v =MINV ,classes =(CE ,CT ),dedupe_mm =10.0 ,
        probs =pr_ ,vertex_conf =VC ,ct_depth_min_mm =1.0 ,cluster_mm =CL ,
        step_path =r ["stp"],**kw )
        merge =lambda L :vote_avg (L ,min_votes =1 ,mode ="wmean")
        base =merge ([mk (pb )for pb in plist ])
        is_hi =robot_cp ._highcp_router (r ["stp"],V ,len (base ))
        cps =merge ([mk (pb ,conn_promote =0.25 )for pb in plist ])if is_hi else base 
        if not cps :
            continue 
        Xg =wire_gate .feats_for (V ,F ,sum (plist )/len (plist ),cps ,CE ,CT ,step_path =r ["stp"])
        P =np .array ([c ["point"]for c in cps ],float )
        D =np .array ([c ["direction"]for c in cps ],float )
        G_ ,Gd =r ["G"],r ["Gd"]
        lab =np .zeros (len (P ),bool )
        if len (G_ ):
            diff =P [:,None ,:]-G_ [None ,:,:]
            al =(diff *Gd [None ,:,:]).sum (-1 )
            pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
            pe =np .where (np .abs (al )<=40.0 ,pe ,np .inf )
            t0 =max (3.0 ,0.06 *r ["diag"])
            us ,ug =set (),set ()
            for dv ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )
            for a in range (len (P ))for b in range (len (G_ ))):
                if dv >t0 or a_ in us or b_ in ug :
                    continue 
                us .add (a_ );ug .add (b_ );lab [a_ ]=True 
        for k in range (len (P )):
            p =P [k ];d =D [k ]
            c_govde =0.0 ;kanal =0.0 
            rel =p -C 
            al =(rel *A ).sum (1 )
            off =np .linalg .norm (rel -al [:,None ]*A ,axis =1 )
            near =off <=3.0 
            if near .any ():
                cand =np .where (near )[0 ]
                j =int (cand [int (np .argmax (np .abs (A [cand ]@d )))])
                c_govde =float (not MET [j ])
                par =np .abs (A @A [j ])>0.99 
                dd =C -C [j ]
                coax =par &(np .linalg .norm (dd -(dd @A [j ])[:,None ]*A [j ],axis =1 )<1.5 )
                kanal =float (bool ((MET &coax ).any ()))
            md =float (np .min (np .linalg .norm (mcom -p ,axis =1 )))if len (mcom )else 99.0 
            X18 .append (Xg [k ]);XC .append ([md ,c_govde ,kanal ])
            Y .append (bool (lab [k ]));GRP .append (r ["pid"]);HI .append (is_hi )

    X18 =np .array (X18 ,float );XC =np .array (XC ,float )
    Y =np .array (Y ,bool );GRP =np .array (GRP );HI =np .array (HI ,bool )
    print (f"sira-dogrulanan part {kabul } | atlanan {red } | {len (Y )} candidate | TP {int (Y .sum ())}")
    if kabul <20 or len (Y )<300 :
        print ("OLCULEMEDI");return 
    X21 =np .hstack ([X18 ,XC ])

    def oof (X ):
        o =np .zeros (len (Y ))
        for tr ,te in GroupKFold (n_splits =5 ).split (X ,Y ,GRP ):
            o [te ]=RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
            random_state =0 ).fit (X [tr ],Y [tr ]).predict_proba (X [te ])[:,1 ]
        return o 

    print (f"\n{'gate':<26}{'OOF AUC':>10}{'F1':>9}{'conclusive':>9}{'recall':>9}")
    res ={}
    for lab_ ,X in (("18 (13+fiziksel)",X18 ),("21 (+3 renk)",X21 )):
        o =oof (X )
        thr =np .where (HI ,THR ["very"],THR ["low"])
        s =o >=thr 
        tp =int ((Y &s ).sum ());fp =int ((~Y &s ).sum ());fn =int ((Y &~s ).sum ())
        p_ =tp /max (tp +fp ,1 );r_ =tp /max (tp +fn ,1 )
        f1 =2 *p_ *r_ /max (p_ +r_ ,1e-9 )
        a =auc_mw (o ,Y )
        print (f"{lab_ :<26}{a :>10.4f}{f1 :>9.4f}{p_ :>9.3f}{r_ :>9.3f}",flush =True )
        res [lab_ ]={"auc":float (a ),"f1":float (f1 ),"precision":float (p_ ),"recall":float (r_ )}

    d =res ["21 (+3 renk)"]["f1"]-res ["18 (13+fiziksel)"]["f1"]
    da =res ["21 (+3 renk)"]["auc"]-res ["18 (13+fiziksel)"]["auc"]
    print (f"\nARTIMLI: AUC {da :+.4f} | candidate-duzeyi F1 {d :+.4f}")
    print (f"KILL: candidate-duzeyi F1 artmazsa RENK listeye GIRMEZ -> {'GIRER'if d >0 else 'GIRMEZ'}")
    json .dump (res |{"artimli_f1":float (d ),"artimli_auc":float (da ),
    "kabul_parca":kabul ,"n_aday":int (len (Y )),
    "karar":"GIRER"if d >0 else "GIRMEZ"},
    open ("results/q9_renk_artimli.json","w"),indent =1 )
    print ("receipt -> results/q9_renk_artimli.json")


if __name__ =="__main__":
    main ()
