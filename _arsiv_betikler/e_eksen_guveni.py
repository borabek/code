# -*- coding: utf-8 -*-
"""E: gate'e EKSEN GUVENI ozelligi. Ucuz measurement -- gate korpusunu bastan kurmadan.

ON-TARAMA (yapildi): axis kaynagi TP/FP'yi ayiriyor mu?
    B-rep SILINDIR eslesti   n=393  TP %94.7
    B-rep DUZLEM eslesti     n= 74  TP %89.2
    none of them (ambiguous)       n= 80  TP %85.0
    -> axis BELIRLI %93.8 vs BELIRSIZ %85.0 = +8.8 score. Gercek but zayif.

Zayif diye ATMIYORUZ: RF 13 ozelligi already kullaniyor, biri more marj katabilir. Ama pahali
yolu (gate_regrow, saatler) kosmadan before UCUZ and DOGRU measurement is done: _h_probs.pkl inside
V/F/olasilik present; candidates turetilir, 13 feature + axis ozellikleri cikarilir, PARCA-GRUPLU
capraz dogrulamayla RF egitilir and two arm same bolmede karsilastirilir.

KILL: OOF CP-F1 farki < +0.005 whereas E duser (gate korpusunu bastan kurmaya degmez).
"""
import os ,sys ,json ,pickle 
import numpy as np 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
W ={"dusuk":0.895 ,"cok":0.105 }


def main ():
    import cp_openings ,robot_cp ,wire_gate 
    import brep_axes as B 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from sklearn .ensemble import RandomForestClassifier 
    from sklearn .model_selection import GroupKFold 

    cfg =json .load (open ("cp_config.json"));pp =cfg ["prediction_postproc"]
    MINV =int (pp ["min_vertices"]);VC =float (pp ["vertex_confidence_mask"]);CL =float (pp ["cluster_mm"])
    TL =float (cfg ["robot_wire_gate_threshold"]);TH =float (cfg ["robot_wire_gate_threshold_highcp"])
    cache =pickle .load (open ("results/_h_probs.pkl","rb"))
    print (f"{len (cache )} part",flush =True )

    X13 ,XE ,y ,grp ,reg =[],[],[],[],[]
    for gi ,r in enumerate (cache ):
        V =np .ascontiguousarray (r ["V"],np .float64 );F =np .ascontiguousarray (r ["F"],np .int64 )
        try :
            cyl =B .cylinders (r ["stp"]);pl =B .planes (r ["stp"])
        except Exception :
            cyl =(np .zeros ((0 ,3 )),)*3 ;pl =(np .zeros ((0 ,3 )),)*3 
        der =[]
        for pb in r ["pbs"]:
            p0 =np .asarray (pb ,np .float64 )
            der .append (cp_openings .connection_points (
            V ,F ,p0 .argmax (-1 ),min_v =MINV ,classes =(CE ,CT ),dedupe_mm =10.0 ,
            probs =p0 ,vertex_conf =VC ,ct_depth_min_mm =1.0 ,cluster_mm =CL ,
            step_path =r ["stp"]))
        base =robot_cp ._vote2 (der ,min_votes =1 )
        is_hi =robot_cp ._highcp_router (r ["stp"],V ,len (base ))
        cps =base 
        if not cps :
            continue 
        probs =sum (np .asarray (p_ ,np .float64 )for p_ in r ["pbs"])/len (r ["pbs"])
        f13 =wire_gate .feats_for (V ,F ,probs ,cps ,CE ,CT )
        P =np .array ([c ["point"]for c in cps ],float )
        D =np .array ([c ["direction"]for c in cps ],float )
        ex =[]
        for i in range (len (cps )):
            a_c =B .axis_at (P [i ],D [i ],cyl ,max_off_mm =5.0 ,max_turn_deg =60.0 ,want_radius =True )
            c_ok =a_c [0 ]is not None 
            rad =float (a_c [1 ])if (c_ok and a_c [1 ])else 0.0 
            p_ok =(not c_ok )and (B .axis_from_planes (P [i ],D [i ],pl ,max_dist_mm =6.0 ,
            min_faces =4 ,flat_ratio =0.20 ,
            max_turn_deg =45.0 )is not None )
            # 4 feature: silindir mi, duzlem mi, never mi, and eslesen silindirin yaricapi
            ex .append ([float (c_ok ),float (p_ok ),float (not (c_ok or p_ok )),rad ])
            # label
        G ,Gd =r ["G"],r ["Gd"]
        lab =np .zeros (len (cps ),int )
        if len (G ):
            diff =P [:,None ,:]-G [None ,:,:];al =(diff *Gd [None ,:,:]).sum (-1 )
            pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
            t =max (3.0 ,0.06 *r ["diag"]);used =set ();hit =np .zeros (len (G ),bool )
            for d_ ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )for a in range (len (P ))for b in range (len (G ))):
                if d_ >t or abs (al [a_ ,b_ ])>40 or a_ in used or hit [b_ ]:continue 
                hit [b_ ]=True ;used .add (a_ );lab [a_ ]=1 
        X13 .append (f13 );XE .append (np .array (ex ,float ));y .append (lab )
        grp +=[gi ]*len (cps );reg +=["cok"if r ["n"]>=8 else "dusuk"]*len (cps )
        if (gi +1 )%25 ==0 :
            print (f"  {gi +1 } part",flush =True )
    X13 =np .vstack (X13 );XE =np .vstack (XE );y =np .concatenate (y )
    grp =np .array (grp );reg =np .array (reg )
    ngt ={gi :len (r ["G"])for gi ,r in enumerate (cache )}
    print (f"\n{len (y )} candidate, {int (y .sum ())} TP\n",flush =True )

    def oof (X ):
        o =np .zeros (len (y ))
        for tr ,te in GroupKFold (n_splits =5 ).split (X ,y ,grp ):
            o [te ]=RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
            random_state =0 ).fit (X [tr ],y [tr ]).predict_proba (X [te ])[:,1 ]
        return o 

    def f1_of (o ):
        agg ={"dusuk":[0 ,0 ,0 ],"cok":[0 ,0 ,0 ]}
        tot ={"dusuk":0 ,"cok":0 }
        for gi in np .unique (grp ):
            k =reg [grp ==gi ][0 ]
            tot [k ]+=ngt [gi ]
        for k in ("dusuk","cok"):
            m =reg ==k 
            thr =TH if k =="cok"else TL 
            selm =(o >=thr )&m 
            tp =int ((y [selm ]==1 ).sum ());fp =int (selm .sum ())-tp 
            p =tp /max (tp +fp ,1 );rc =tp /max (tot [k ],1 )
            agg [k ]=[2 *p *rc /max (p +rc ,1e-9 ),p ,rc ]
        return sum (W [k ]*agg [k ][0 ]for k in W ),agg 

    a13 ,g13 =f1_of (oof (X13 ))
    aE ,gE =f1_of (oof (np .hstack ([X13 ,XE ])))
    print (f"{'arm':<28}{'CP-F1':>9}{'dusuk':>9}{'cok':>9}")
    print (f"{'13 ozellik (mevcut)':<28}{a13 :>9.4f}{g13 ['dusuk'][0 ]:>9.4f}{g13 ['cok'][0 ]:>9.4f}")
    print (f"{'13 + EKSEN GUVENI (4)':<28}{aE :>9.4f}{gE ['dusuk'][0 ]:>9.4f}{gE ['cok'][0 ]:>9.4f}")
    print (f"\nFARK: {aE -a13 :+.4f}")
    print (f"KARAR (>= +0.005): {'E URUNE ALINIR'if aE -a13 >=0.005 else 'E DUSER'}")
    json .dump ({"f1_13":a13 ,"f1_13_eksen":aE ,"fark":aE -a13 },
    open ("results/e_eksen_guveni.json","w"),indent =1 )


if __name__ =="__main__":
    main ()
