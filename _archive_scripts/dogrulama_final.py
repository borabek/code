# -*- coding: utf-8 -*-
"""E + I sonrasi TAM BORU HATTI dogrulamasi -- new headline count.

WHY SEPARATE BIR BETIK: `tolerans_gercegi.py` gate'i npz'den 13 ozellikle kuruyor and onbellegi
I'dan ONCE uretilmisti. Yeni urun two degisiklik tasiyor:
  I -- silindir susunca DUZLEM ciftlerinden axis (cp_openings inside, step_path with)
  E -- gate'e 4 EKSEN GUVENI ozelligi (17 feature) + low-CP esigi 0.40 -> 0.45
Ikisi de however real hatta birlikte olculebilir.

SIZINTI: gate test AILELERI cikarilarak egitilir; esikler BUYUK veride same sekilde secilir.
(Uyari: this korpusta family_key part numarasinin kendisi -- see YOL_075_ROBOT.md F maddesi.
Yani koruma fiilen PARCA-DISI. F maddesi bunu geometri gruplamasiyla yeniden olcecek.)
"""
import os ,sys ,json ,pickle 
import numpy as np 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
W ={"low":0.895 ,"very":0.105 }
NPZ ="results/gate_regrow_data_rt2.npz"
NPZ_AX ="results/gate_regrow_data_rt2_axis.npz"
CACHE ="results/_final_cache.pkl"


def main ():
    import torch ,thesis_remesh ,cp_openings ,robot_cp ,wire_gate ,diffusionnet as D 
    from cad_eval import align_frames 
    from infer_step_cp import step_to_mesh ,load_any 
    from big_arbiter import eligible 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from json_dataset import family_key 
    from sklearn .ensemble import RandomForestClassifier 
    from sklearn .model_selection import GroupKFold 

    cfg =json .load (open ("cp_config.json"));pp =cfg ["prediction_postproc"]
    MINV =int (pp ["min_vertices"]);VC =float (pp ["vertex_confidence_mask"]);CL =float (pp ["cluster_mm"])
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    LOCK =set (json .load (open ("results/split_lock.json"))["locked_parts"])

    parts =[]
    for m ,p ,jf ,s in eligible ():
        if p in LOCK :continue 
        try :n =len (json .load (open (jf ,encoding ="utf-8-sig"))["ConnectionPoints"])
        except Exception :continue 
        if n >0 :parts .append ((m ,p ,jf ,s ,n ))
    rng =np .random .RandomState (202 )
    lo =[x for x in parts if x [4 ]<8 ];hi =[x for x in parts if x [4 ]>=8 ]
    sel =([lo [i ]for i in rng .choice (len (lo ),70 ,replace =False )]+
    [hi [i ]for i in rng .choice (len (hi ),30 ,replace =False )])
    test_fams ={family_key (p [1 ])for p in sel }

    d =np .load (NPZ ,allow_pickle =True )
    XE =np .load (NPZ_AX )["XE"]
    X13 =d ["X"];y =d ["y"];fams =d ["fams"].astype (str );groups =d ["groups"]
    ngt =dict (zip (d ["grp_ids"].tolist (),d ["ngt"].tolist ()))
    keep =~np .isin (fams ,list (test_fams ))
    X17 =np .hstack ([X13 ,XE ])
    print (f"{len (sel )} test part | gate {int (keep .sum ())} candidate x 17 ozellik",flush =True )

    # esikler: large veride, test aileleri disarida, 17 ozellikle
    reg =np .array ([("very"if int (ngt .get (int (g ),0 ))>=8 else "low")for g in groups ])[keep ]
    tot ={"low":0 ,"very":0 }
    for g in {int (g )for g in groups [keep ]}:
        n =int (ngt .get (g ,0 ))
        if n >0 :tot ["very"if n >=8 else "low"]+=n 
    oof =np .zeros (int (keep .sum ()))
    Xk ,yk ,fk =X17 [keep ],y [keep ],fams [keep ]
    for tr ,te in GroupKFold (n_splits =5 ).split (Xk ,yk ,fk ):
        oof [te ]=RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
        random_state =0 ).fit (Xk [tr ],yk [tr ]).predict_proba (Xk [te ])[:,1 ]
    thr ={}
    for k in ("low","very"):
        b =(0.0 ,0.40 )
        for t in np .arange (0.20 ,0.71 ,0.05 ):
            m =reg ==k ;s_ =(oof >=t )&m 
            tp =int ((yk [s_ ]==1 ).sum ());fp =int (s_ .sum ())-tp 
            p =tp /max (tp +fp ,1 );r =tp /max (tot [k ],1 )
            f =2 *p *r /max (p +r ,1e-9 )
            if f >b [0 ]:b =(f ,float (t ))
        thr [k ]=b [1 ]
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (Xk ,yk )
    print (f"esikler (aile-disi OOF, 17 ozellik): {thr }",flush =True )

    models =[load_any (c ,dev =dev )[:2 ]for c in cfg ["current_product"]["checkpoints"]]
    cache =[]
    for mfg ,pid ,jf ,stp ,n in sel :
        try :
            Vr ,Fr =step_to_mesh (stp )
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
            pbs =[]
            for model ,meta in models :
                _ ,pb =D .predict (model ,meta ,V ,F ,device =dev ,
                op_cache_dir =f"results/step_infer/ops_k{int (meta .get ('k_eig',64 ))}",
                return_probs =True )
                pbs .append (np .asarray (pb ,float ))
            der =lambda pr :[cp_openings .connection_points (
            V ,F ,pb .argmax (-1 ),min_v =MINV ,classes =(CE ,CT ),dedupe_mm =10.0 ,
            probs =pb ,vertex_conf =VC ,ct_depth_min_mm =1.0 ,cluster_mm =CL ,
            conn_promote =pr ,step_path =stp )for pb in pbs ]# <- I here
            base =robot_cp ._vote2 (der (0.0 ),min_votes =1 )
            is_hi =robot_cp ._highcp_router (stp ,V ,len (base ))
            cps =robot_cp ._vote2 (der (0.25 ),min_votes =1 )if is_hi else base 
            X17c =X13c =None 
            if cps :
                probs =sum (pbs )/len (pbs )
                X17c =wire_gate .feats_for (V ,F ,probs ,cps ,CE ,CT ,step_path =stp )
                X13c =X17c [:,:13 ]# same candidates, ESKI feature kumesi -> E izole edilir
            j =json .load (open (jf ,encoding ="utf-8-sig"))
            Vj =np .array ([[q [c ]for c in "XYZ"]for q in j ["Graphic3d"]["Points"]],float )
            G =np .array ([[c ["Point"][q ]for q in "XYZ"]for c in j ["ConnectionPoints"]],float )
            Gd =np .array ([[c ["InsertDirection"][q ]for q in "XYZ"]for c in j ["ConnectionPoints"]],float )
            Gd /=np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 
            R ,t ,_ =align_frames (Vr ,Vj )
            cache .append (dict (
            P =np .array ([c ["point"]for c in cps ],float )if cps else np .zeros ((0 ,3 )),
            Pd =np .array ([c ["direction"]for c in cps ],float )if cps else np .zeros ((0 ,3 )),
            X17 =X17c ,X13 =X13c ,is_hi =bool (is_hi ),
            G =(G -t )@R ,Gd =Gd @R ,n =n ,
            diag =float (np .linalg .norm (V .max (0 )-V .min (0 )))))
        except Exception :
            continue 
        if len (cache )%25 ==0 :
            print (f"  {len (cache )} part",flush =True )
    pickle .dump (cache ,open (CACHE ,"wb"))
    print (f"cache {len (cache )} part\n",flush =True )

    # IKI GATE, AYNI ADAYLAR: E'nin uctan uca katkisi so IZOLE becomes.
    clf13 =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (X13 [keep ],y [keep ])
    reg13 =np .array ([("very"if int (ngt .get (int (g ),0 ))>=8 else "low")for g in groups ])[keep ]
    oof13 =np .zeros (int (keep .sum ()))
    for tr ,te in GroupKFold (n_splits =5 ).split (X13 [keep ],y [keep ],fams [keep ]):
        oof13 [te ]=RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
        random_state =0 ).fit (X13 [keep ][tr ],y [keep ][tr ]
        ).predict_proba (X13 [keep ][te ])[:,1 ]
    thr13 ={}
    for k in ("low","very"):
        b =(0.0 ,0.40 )
        for t in np .arange (0.20 ,0.71 ,0.05 ):
            m =reg13 ==k ;s_ =(oof13 >=t )&m 
            tp =int ((y [keep ][s_ ]==1 ).sum ());fp =int (s_ .sum ())-tp 
            p =tp /max (tp +fp ,1 );r =tp /max (tot [k ],1 )
            f =2 *p *r /max (p +r ,1e-9 )
            if f >b [0 ]:b =(f ,float (t ))
        thr13 [k ]=b [1 ]
    print (f"13-ozellik esikleri: {thr13 }",flush =True )

    def score (model ,feat_key ,TH ,tol ,am ,pct =False ):
        agg ={"low":[0 ,0 ,0 ],"very":[0 ,0 ,0 ]};ANG =[]
        for r in cache :
            Xk =r [feat_key ]
            if Xk is None or not len (r ["P"]):
                Q =np .zeros ((0 ,3 ));Qd =np .zeros ((0 ,3 ))
            else :
                sc =model .predict_proba (Xk )[:,1 ]
                t_ =TH ["very"]if r ["is_hi"]else TH ["low"]
                m =sc >=t_ 
                Q =r ["P"][m ];Qd =r ["Pd"][m ]
            Gt ,Gd =r ["G"],r ["Gd"]
            hit =np .zeros (len (Gt ),bool );used =set ()
            if len (Q )and len (Gt ):
                diff =Q [:,None ,:]-Gt [None ,:,:];al =(diff *Gd [None ,:,:]).sum (-1 )
                pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
                an =np .degrees (np .arccos (np .clip (np .abs (Qd @Gd .T ),0 ,1 )))
                tt =max (3.0 ,0.06 *r ["diag"])if pct else tol 
                pe =np .where ((np .abs (al )>40 )|(an >am ),np .inf ,pe )
                for d_ ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )for a in range (len (Q ))for b in range (len (Gt ))):
                    if d_ >tt or a_ in used or hit [b_ ]:continue 
                    hit [b_ ]=True ;used .add (a_ )
                    ANG .append (np .degrees (np .arccos (min (1.0 ,abs (float (np .dot (Qd [a_ ],Gd [b_ ])))))))
            tp =int (hit .sum ());k ="very"if r ["n"]>=8 else "low"
            agg [k ][0 ]+=tp ;agg [k ][1 ]+=len (Q )-tp ;agg [k ][2 ]+=len (Gt )-tp 
        o ={};TP =FP =FN =0 
        for k ,(T ,Fp ,Fn )in agg .items ():
            p =T /max (T +Fp ,1 );rc =T /max (T +Fn ,1 )
            o [k ]=2 *p *rc /max (p +rc ,1e-9 );TP +=T ;FP +=Fp ;FN +=Fn 
        return (sum (W [k ]*o [k ]for k in W ),o ["low"],o ["very"],
        TP /max (TP +FP ,1 ),TP /max (TP +FN ,1 ),np .array (ANG )if ANG else np .array ([0.0 ]))

    print ()
    print (f"{'arm':<24}{'detection':>9}{'yanal2':>9}{'ROBOT':>9}{'ratio':>8}{'>15d':>8}")
    res ={}
    for lab ,mdl ,fk ,TH in (("13 feature (E YOK)",clf13 ,"X13",thr13 ),
    ("17 feature (E VAR)",clf ,"X17",thr )):
        d_ =score (mdl ,fk ,TH ,0 ,180 ,True )
        l_ =score (mdl ,fk ,TH ,2.0 ,180 )
        r_ =score (mdl ,fk ,TH ,2.0 ,10 )
        res [lab ]=dict (detection =d_ [0 ],yanal2 =l_ [0 ],robot =r_ [0 ],ratio =r_ [0 ]/max (d_ [0 ],1e-9 ),
        dusuk =r_ [1 ],very =r_ [2 ],precision =d_ [3 ],recall =d_ [4 ],
        eksen15 =float ((d_ [5 ]>15 ).mean ()))
        print (f"{lab :<24}{d_ [0 ]:>9.4f}{l_ [0 ]:>9.4f}{r_ [0 ]:>9.4f}"
        f"{r_ [0 ]/max (d_ [0 ],1e-9 ):>8.3f}{100 *(d_ [5 ]>15 ).mean ():>7.1f}%",flush =True )
    a =res ["17 feature (E VAR)"];b =res ["13 feature (E YOK)"]
    print (f"E'nin UCTAN UCA katkisi: detection {a ['detection']-b ['detection']:+.4f}  "
    f"robot {a ['robot']-b ['robot']:+.4f}")
    json .dump (res ,open ("results/dogrulama_final.json","w"),indent =1 )
    print ("receipt -> results/dogrulama_final.json")


if __name__ =="__main__":
    main ()
