# -*- coding: utf-8 -*-
"""D1.4: renk ozelligi TAM korpusta (1903 part) gate'e ne kazandiriyor?

Erken sinyal (64 part): +0.0195. Simdi real measurement.
Ozellikler cikarilmis adaylarin KONUMLARINDAN offline is computed (GPU gerekmez).
NaN = renk absent; 0 YAZILMAZ ("metal absent" yalani produces). Eksiklik bayragi ayri feature.
KILL: corpus-agirlikli CP-F1 katkisi < +0.02.
"""
import os ,sys ,json 
import numpy as np 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
W ={"dusuk":0.895 ,"cok":0.105 }
NPZ ="results/gate_regrow_data_rt2.npz"


def main ():
    import step_face_color_link as L 
    from cad_eval import align_frames 
    from big_arbiter import eligible 
    from infer_step_cp import step_to_mesh 
    from k7_dip_metal import dip_features 
    from sklearn .ensemble import RandomForestClassifier 
    from sklearn .model_selection import GroupKFold 

    d =np .load (NPZ ,allow_pickle =True )
    X ,y ,groups ,fams =d ["X"],d ["y"],d ["groups"],d ["fams"]
    pts ,dirs ,pids =d ["pts"],d ["dirs"],d ["pids"].astype (str )
    ngt =dict (zip (d ["grp_ids"].tolist (),d ["ngt"].tolist ()))
    fam ={}
    for g ,f in zip (groups ,fams ):
        fam .setdefault (int (g ),str (f ))
    gk =np .array ([fam [int (g )]for g in groups ])
    reg =np .array ([("cok"if int (ngt .get (int (g ),0 ))>=8 else "dusuk")for g in groups ])
    tot ={k :sum (v for g ,v in ngt .items ()if (int (v )>=8 )==(k =="cok"))
    for k in ("dusuk","cok")}
    stp_of ={p :s for _ ,p ,_ ,s in eligible ()}

    XC =np .full ((len (y ),3 ),np .nan )
    done =ok =0 
    for pid in np .unique (pids ):
        m =pids ==pid 
        if pid not in stp_of :
            continue 
        done +=1 
        try :
            faces =L .face_vertices_from_text (stp_of [pid ])
            met =[x for x in faces if x ["is_metal"]]
            if not faces or not met :
                continue 
            Vr ,_ =step_to_mesh (stp_of [pid ])
            allp =np .vstack ([x ["pts"]for x in faces ])
            flag =np .concatenate ([np .full (len (x ["pts"]),1.0 if x ["is_metal"]else 0.0 )
            for x in faces ])
            # KAPSAMA DUZELTMESI (2026-07-30, measured):
            #  (a) hizalama RENKLI yuzeylerle not, katinin TUM koseleriyle is done --
            #      renkli yuzeyler katinin a kismini kapladiginda oncekisi cakiyordu
            #  (b) residual esigi 1.0 -> 2.0mm; 1-2mm bandindaki 7 parcada donusturulmus
            #      koselerin mesh yuzeyine medyan uzakligi 0.8-1.3mm, i.e. pratikte is used
            allv =L .all_vertex_points (stp_of [pid ])
            R ,t ,res =align_frames (Vr ,allv if len (allv )else allp )
            if res >1.0 :# 2.0 was tried: kapsama %36->%58 but katki +0.0078 -> -0.0044
                continue 
            allm =(allp -t )@R 
            mp_all =np .column_stack ([allm ,flag ])
            mpts =allm [flag >0.5 ]
            idx =np .where (m )[0 ]
            for i in idx :
                XC [i ]=dip_features (mp_all ,mpts ,pts [i ],dirs [i ])
            ok +=1 
        except Exception :
            continue 
        if done %200 ==0 :
            print (f"  {done } part ({ok } renkli)",flush =True )
    print (f"\n{done } part islendi, {ok } tanesinde renk (%{100 *ok /max (done ,1 ):.0f})")
    print (f"renk ozelligi olan candidate: {int ((~np .isnan (XC ).any (1 )).sum ())}/{len (y )}\n",flush =True )

    def oof (Xm ):
        s =np .zeros (len (y ))
        for tr ,te in GroupKFold (n_splits =5 ).split (Xm ,y ,gk ):
            s [te ]=RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
            random_state =0 ).fit (Xm [tr ],y [tr ]).predict_proba (Xm [te ])[:,1 ]
        return s 

    def best (s ):
        out ={}
        for k in ("dusuk","cok"):
            bb =0 
            for thr in np .arange (0.20 ,0.71 ,0.05 ):
                mm =reg ==k ;sel =(s >=thr )&mm 
                tp =int ((y [sel ]==1 ).sum ());fp =int (sel .sum ())-tp 
                p =tp /max (tp +fp ,1 );r =tp /max (tot [k ],1 )
                bb =max (bb ,2 *p *r /max (p +r ,1e-9 ))
            out [k ]=bb 
        out ["w"]=sum (W [k ]*out [k ]for k in W )
        return out 

    miss =np .isnan (XC ).any (axis =1 ).astype (float )[:,None ]
    XCf =np .where (np .isnan (XC ),-1.0 ,XC )
    a =best (oof (X ))
    b =best (oof (np .hstack ([X ,XCf ,miss ])))
    print (f"{'ozellik seti':<26}{'dusuk':>9}{'cok':>9}{'agirlikli':>11}")
    print (f"{'13 mevcut':<26}{a ['dusuk']:>9.4f}{a ['cok']:>9.4f}{a ['w']:>11.4f}")
    print (f"{'13 + RENK':<26}{b ['dusuk']:>9.4f}{b ['cok']:>9.4f}{b ['w']:>11.4f}")
    print (f"\nRENK katkisi: {b ['w']-a ['w']:+.4f}")
    print (f"KAPI (>= +0.02): {'GECTI'if b ['w']-a ['w']>=0.02 else 'OLU'}")
    json .dump ({"base":a ,"color":b ,"delta":b ["w"]-a ["w"],
    "parts_with_color":ok ,"parts":done },
    open ("results/d1_4_renk.json","w"),indent =1 )
    print ("receipt -> results/d1_4_renk.json")


if __name__ =="__main__":
    main ()
