# -*- coding: utf-8 -*-
"""E'yi TAM KORPUS gate'inde dogrula -- inference YAPMADAN.

KISAYOL: results/gate_regrow_data_rt2.npz candidate NOKTA (pts), YON (dirs) and PARCA NO (pids)
tasiyor. Eksen guveni ozellikleri only bunlara + STEP'in B-rep'ine baglidir; agin
olasiliklarina DEGIL. Yani 23551 candidate for ozellikler offline hesaplanip X'e eklenebilir --
1903 parcalik yeniden inference (saatler) gereksiz.

WHY IMPORTANT: E, 100 parcalik zayif a gate'te +0.0208 verdi. Ama this projede a kaldirac
"bayat/zayif gate"te kazanip full corpus gate'inde KAYBETTI (P1 uzlasma yukseltici: +0.006 zayif
gate, -0.0225 full gate). Bu yuzden full korpusta dogrulanmadan urune GIRMEZ.

KILL: full corpus OOF CP-F1 farki < +0.005 whereas E duser.
"""
import os ,sys ,json ,glob 
import numpy as np 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
W ={"low":0.895 ,"very":0.105 }
NPZ ="results/gate_regrow_data_rt2.npz"
OUT ="results/gate_regrow_data_rt2_axis.npz"


def main ():
    import brep_axes as B 
    from sklearn .ensemble import RandomForestClassifier 
    from sklearn .model_selection import GroupKFold 

    d =np .load (NPZ ,allow_pickle =True )
    X ,y ,groups ,fams =d ["X"],d ["y"],d ["groups"],d ["fams"]
    pts ,dirs ,pids =d ["pts"],d ["dirs"],np .array ([str (x )for x in d ["pids"]])
    ngt =dict (zip (d ["grp_ids"].tolist (),d ["ngt"].tolist ()))
    print (f"{X .shape [0 ]} candidate / {len (set (pids ))} part / {X .shape [1 ]} ozellik",flush =True )

    if os .path .exists (OUT ):
        XE =np .load (OUT )["XE"]
        print (f"axis ozellikleri onbellekten: {XE .shape }",flush =True )
    else :
        step ={os .path .basename (s ).split ("_")[1 ]:s for s in glob .glob ("all_wscad_stp/*.stp")}
        XE =np .zeros ((len (X ),4 ),float )
        miss =0 
        for k ,pid in enumerate (sorted (set (pids ))):
            stp =step .get (pid )
            m =pids ==pid 
            if not stp :
                miss +=1 ;continue 
            try :
                cyl =B .cylinders (stp );pl =B .planes (stp )
            except Exception :
                miss +=1 ;continue 
            idx =np .where (m )[0 ]
            for i in idx :
                a ,rad =B .axis_at (pts [i ],dirs [i ],cyl ,max_off_mm =5.0 ,max_turn_deg =60.0 ,
                want_radius =True )
                c_ok =a is not None 
                p_ok =(not c_ok )and (B .axis_from_planes (pts [i ],dirs [i ],pl ,max_dist_mm =6.0 ,
                min_faces =4 ,flat_ratio =0.20 ,
                max_turn_deg =45.0 )is not None )
                XE [i ]=[float (c_ok ),float (p_ok ),float (not (c_ok or p_ok )),
                float (rad )if (c_ok and rad )else 0.0 ]
            if (k +1 )%100 ==0 :
                print (f"  {k +1 } part",flush =True )
        np .savez (OUT ,XE =XE )
        print (f"axis ozellikleri yazildi ({miss } part STEP/B-rep okunamadi)",flush =True )

    print (f"\neksen kaynagi dagilimi: silindir %{100 *XE [:,0 ].mean ():.1f} | "
    f"duzlem %{100 *XE [:,1 ].mean ():.1f} | belirsiz %{100 *XE [:,2 ].mean ():.1f}")
    det =XE [:,2 ]==0 
    print (f"  axis BELIRLI  TP orani %{100 *y [det ].mean ():.1f} (n={int (det .sum ())})")
    print (f"  axis BELIRSIZ TP orani %{100 *y [~det ].mean ():.1f} (n={int ((~det ).sum ())})\n")

    reg =np .array ([("very"if int (ngt .get (int (g ),0 ))>=8 else "low")for g in groups ])
    tot ={"low":0 ,"very":0 }
    for g in {int (g )for g in groups }:
        n =int (ngt .get (g ,0 ))
        if n >0 :
            tot ["very"if n >=8 else "low"]+=n 
    gk =fams .astype (str )

    def oof (M ):
        o =np .zeros (len (y ))
        for tr ,te in GroupKFold (n_splits =5 ).split (M ,y ,gk ):
            o [te ]=RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
            random_state =0 ).fit (M [tr ],y [tr ]).predict_proba (M [te ])[:,1 ]
        return o 

    def best_f1 (o ):
        out ={}
        for k in ("low","very"):
            b =(0.0 ,0.40 )
            for thr in np .arange (0.20 ,0.71 ,0.05 ):
                m =reg ==k ;s =(o >=thr )&m 
                tp =int ((y [s ]==1 ).sum ());fp =int (s .sum ())-tp 
                p =tp /max (tp +fp ,1 );r =tp /max (tot [k ],1 )
                f =2 *p *r /max (p +r ,1e-9 )
                if f >b [0 ]:b =(f ,float (thr ))
            out [k ]=b 
        return sum (W [k ]*out [k ][0 ]for k in W ),out 

    a13 ,d13 =best_f1 (oof (X ))
    aE ,dE =best_f1 (oof (np .hstack ([X ,XE ])))
    print (f"{'arm':<26}{'CP-F1':>9}{'low':>9}{'very':>9}{'esikler':>18}")
    print (f"{'13 feature (dagitilan)':<26}{a13 :>9.4f}{d13 ['low'][0 ]:>9.4f}{d13 ['very'][0 ]:>9.4f}"
    f"{str ((d13 ['low'][1 ],d13 ['very'][1 ])):>18}")
    print (f"{'13 + EKSEN GUVENI':<26}{aE :>9.4f}{dE ['low'][0 ]:>9.4f}{dE ['very'][0 ]:>9.4f}"
    f"{str ((dE ['low'][1 ],dE ['very'][1 ])):>18}")
    print (f"\nTAM KORPUS FARKI: {aE -a13 :+.4f}")
    print (f"KARAR (>= +0.005): {'E URUNE ALINIR'if aE -a13 >=0.005 else 'E DUSER'}")
    json .dump ({"f1_13":a13 ,"f1_axis":aE ,"difference":aE -a13 ,
    "thr_13":{k :v [1 ]for k ,v in d13 .items ()},
    "thr_axis":{k :v [1 ]for k ,v in dE .items ()}},
    open ("results/e_full_gate.json","w"),indent =1 )


if __name__ =="__main__":
    main ()
