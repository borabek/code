# -*- coding: utf-8 -*-
"""H2: H'nin OLUM BICIMINI hedefleyen two correction.

H olctu: oncul duzeltmesi recall'i ARTIRMADI (0.747 -> 0.722), precision'i yikti (0.923 -> 0.791).
Yani fazladan atesnlenen koseler opening not, DAGINIK KUCUK LEKELER. Iki hedefli care:

  A) KAPILI oncul  : oncul only already CE/CT sinyali which is koselerde uygulanir
                     (p_CE+p_CT >= floor). Esigin under kalmis GERCEK acikliklar kurtulur,
                     no sinyali olmayan koseler never girmez.
  B) min_v with birlikte: fazladan gelen lekeler KUCUK. min_v yukseltmek onlari eler,
                     real acikliklar (more large) kalir. H single basina not, CIFT as.

Onbellekten runs, GPU'ya dokunmaz.
KILL: detection F1 mevcut 0.7783'u gecmezse H tamamen kapanir.
"""
import os ,sys ,json ,pickle 
import numpy as np 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
W ={"low":0.895 ,"very":0.105 }
NPZ ="results/gate_regrow_data_rt2.npz"
CACHE ="results/_h_probs.pkl"


def main ():
    import cp_openings ,robot_cp ,wire_gate 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from json_dataset import family_key 
    from sklearn .ensemble import RandomForestClassifier 
    from h_sinif_onculu import big_thr 

    cfg =json .load (open ("cp_config.json"));pp =cfg ["prediction_postproc"]
    VC =float (pp ["vertex_confidence_mask"]);CL =float (pp ["cluster_mm"])
    MINV0 =int (pp ["min_vertices"])
    cache =pickle .load (open (CACHE ,"rb"))
    from big_arbiter import eligible 
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
    thr =big_thr (NPZ ,test_fams )
    d =np .load (NPZ ,allow_pickle =True )
    keep =~np .isin (d ["fams"].astype (str ),list (test_fams ))
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (d ["X"][keep ],d ["y"][keep ])
    acc =np .zeros (5 )
    for r in cache :
        for pb in r ["pbs"]:acc +=np .asarray (pb ,np .float64 ).mean (0 )
    prior =acc /acc .sum ()
    print (f"{len (cache )} part | esikler {thr }\n",flush =True )

    def score (alpha ,floor ,minv ):
        agg ={"low":[0 ,0 ,0 ],"very":[0 ,0 ,0 ]}
        w =1.0 /np .maximum (prior ,1e-9 )**alpha 
        for r in cache :
            V =np .ascontiguousarray (r ["V"],np .float64 );F =np .ascontiguousarray (r ["F"],np .int64 )
            def der (pr_ ):
                out =[]
                for pb in r ["pbs"]:
                    p0 =np .asarray (pb ,np .float64 )
                    q =p0 *w [None ,:]
                    lab =q .argmax (-1 )
                    if floor >0 :# KAPI: sinyalsiz koselerde ESKI karari koru
                        weak =(p0 [:,CE ]+p0 [:,CT ])<floor 
                        lab [weak ]=p0 [weak ].argmax (-1 )
                    out .append (cp_openings .connection_points (
                    V ,F ,lab ,min_v =minv ,classes =(CE ,CT ),dedupe_mm =10.0 ,
                    probs =p0 ,vertex_conf =VC ,ct_depth_min_mm =1.0 ,cluster_mm =CL ,
                    conn_promote =pr_ ,step_path =r ["stp"]))
                return out 
            base =robot_cp ._vote2 (der (0.0 ),min_votes =1 )
            is_hi =robot_cp ._highcp_router (r ["stp"],V ,len (base ))
            cps =robot_cp ._vote2 (der (0.25 ),min_votes =1 )if is_hi else base 
            kept =[]
            if cps :
                probs =sum (np .asarray (p_ ,np .float64 )for p_ in r ["pbs"])/len (r ["pbs"])
                sc =clf .predict_proba (wire_gate .feats_for (V ,F ,probs ,cps ,CE ,CT ))[:,1 ]
                t_ =thr ["very"]if is_hi else thr ["low"]
                kept =[c for c ,s_ in zip (cps ,sc )if s_ >=t_ ]
            Q =np .array ([c ["point"]for c in kept ],float )if kept else np .zeros ((0 ,3 ))
            G ,Gd =r ["G"],r ["Gd"];hit =np .zeros (len (G ),bool );used =set ()
            if len (Q )and len (G ):
                diff =Q [:,None ,:]-G [None ,:,:];al =(diff *Gd [None ,:,:]).sum (-1 )
                pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
                tol =max (3.0 ,0.06 *r ["diag"])
                pe =np .where (np .abs (al )<=40.0 ,pe ,np .inf )
                for d_ ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )for a in range (len (Q ))for b in range (len (G ))):
                    if d_ >tol or a_ in used or hit [b_ ]:continue 
                    hit [b_ ]=True ;used .add (a_ )
            tp =int (hit .sum ());k ="very"if r ["n"]>=8 else "low"
            agg [k ][0 ]+=tp ;agg [k ][1 ]+=len (Q )-tp ;agg [k ][2 ]+=len (G )-tp 
        o ={};TP =FP =FN =0 
        for k ,(T ,Fp ,Fn )in agg .items ():
            p =T /max (T +Fp ,1 );rc =T /max (T +Fn ,1 )
            o [k ]=2 *p *rc /max (p +rc ,1e-9 );TP +=T ;FP +=Fp ;FN +=Fn 
        return (sum (W [k ]*o [k ]for k in W ),TP /max (TP +FP ,1 ),TP /max (TP +FN ,1 ))

    print (f"{'setting':<34}{'detection F1':>11}{'precision':>10}{'recall':>9}{'difference':>9}")
    base =None 
    CFG =[("MEVCUT (alpha 0)",0.0 ,0.0 ,MINV0 ),
    ("A: alpha .3 + gate .10",0.3 ,0.10 ,MINV0 ),
    ("A: alpha .3 + gate .25",0.3 ,0.25 ,MINV0 ),
    ("A: alpha .5 + gate .25",0.5 ,0.25 ,MINV0 ),
    ("B: alpha .3 + min_v 10",0.3 ,0.0 ,10 ),
    ("B: alpha .3 + min_v 20",0.3 ,0.0 ,20 ),
    ("A+B: alpha .3 gate .25 min_v 10",0.3 ,0.25 ,10 ),
    ("A+B: alpha .5 gate .25 min_v 10",0.5 ,0.25 ,10 )]
    res ={}
    for lab ,al ,fl ,mv in CFG :
        f1_ ,p_ ,r_ =score (al ,fl ,mv )
        if base is None :base =f1_ 
        res [lab ]=dict (f1 =f1_ ,precision =p_ ,recall =r_ )
        print (f"{lab :<34}{f1_ :>11.4f}{p_ :>10.3f}{r_ :>9.3f}{f1_ -base :>+9.4f}",flush =True )
    json .dump (res ,open ("results/h2_hedefli.json","w"),indent =1 )
    print ("\nmakbuz -> results/h2_hedefli.json")


if __name__ =="__main__":
    main ()
