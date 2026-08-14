# -*- coding: utf-8 -*-
"""H: argmax instead of SINIF-ONCULU karar. Bir inference, after bedava suepurme.

WHY: label this an 5 sinifli softmax'in ARGMAX'i. Baglanti siniflari (CableEntry/Contact)
NADIR; argmax that is why yapisal as Housing lehine yanli. Bir kose %35 CableEntry / %40
Housing oldugunda Housing kazanir and that opening HIC atesnlenmez. Kayitli bulgu: low-CP'de
"network atesledikten after no sey kaybedilmiyor, ceiling dogrudan ATESLEME ORANI" -- i.e. this
axis full da tavana bakiyor and bugune up to HIC taranmadi (vertex_confidence_mask a MASKE,
oncul duzeltmesi not).

YONTEM: p_duzeltilmis = p / oncul**alpha, after argmax. alpha=0 mevcut davranis.
Cikarim BIR KEZ runs, olasiliklar saklanir, alpha bedava suepurulur -- difference so modelden
not YALNIZ karar kuralindan gelir.

KILL (olcumden before): tespit F1 gerilerse duser. Recall artip precision'i gate geri
alamiyorsa duser.
"""
import os ,sys ,json ,pickle 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
W ={"low":0.895 ,"very":0.105 }
NPZ ="results/gate_regrow_data_rt2.npz"
CACHE ="results/_h_probs.pkl"


def big_thr (npz ,test_fams ):
    from sklearn .ensemble import RandomForestClassifier 
    from sklearn .model_selection import GroupKFold 
    d =np .load (npz ,allow_pickle =True )
    X ,y ,groups ,fams =d ["X"],d ["y"],d ["groups"],d ["fams"]
    keep =~np .isin (fams .astype (str ),list (test_fams ))
    X ,y ,groups ,fams =X [keep ],y [keep ],groups [keep ],fams [keep ]
    ngt =dict (zip (d ["grp_ids"].tolist (),d ["ngt"].tolist ()))
    reg =np .array ([("very"if int (ngt .get (int (g ),0 ))>=8 else "low")for g in groups ])
    tot ={"low":0 ,"very":0 }
    for g in {int (g )for g in groups }:
        n =int (ngt .get (g ,0 ))
        if n >0 :
            tot ["very"if n >=8 else "low"]+=n 
    oof =np .zeros (len (y ))
    for tr ,te in GroupKFold (n_splits =5 ).split (X ,y ,fams .astype (str )):
        oof [te ]=RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
        random_state =0 ).fit (X [tr ],y [tr ]).predict_proba (X [te ])[:,1 ]
    out ={}
    for k in ("low","very"):
        best =(0.0 ,0.40 )
        for thr in np .arange (0.20 ,0.71 ,0.05 ):
            m =reg ==k 
            sel =(oof >=thr )&m 
            tp =int ((y [sel ]==1 ).sum ())
            fp =int (sel .sum ())-tp 
            p =tp /max (tp +fp ,1 )
            r =tp /max (tot [k ],1 )
            f1 =2 *p *r /max (p +r ,1e-9 )
            if f1 >best [0 ]:
                best =(f1 ,float (thr ))
        out [k ]=best [1 ]
    return out 


def main ():
    import torch ,thesis_remesh ,cp_openings ,robot_cp ,wire_gate ,diffusionnet as D 
    from cad_eval import align_frames 
    from infer_step_cp import step_to_mesh ,load_any 
    from big_arbiter import eligible 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from json_dataset import family_key 
    from sklearn .ensemble import RandomForestClassifier 

    cfg =json .load (open ("cp_config.json"));pp =cfg ["prediction_postproc"]
    MINV =int (pp ["min_vertices"]);VC =float (pp ["vertex_confidence_mask"]);CL =float (pp ["cluster_mm"])
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    LOCK =set (json .load (open ("results/split_lock.json"))["locked_parts"])

    parts =[]
    for m ,p ,jf ,s in eligible ():
        if p in LOCK :
            continue 
        try :
            n =len (json .load (open (jf ,encoding ="utf-8-sig"))["ConnectionPoints"])
        except Exception :
            continue 
        if n >0 :
            parts .append ((m ,p ,jf ,s ,n ))
    rng =np .random .RandomState (202 )
    lo =[x for x in parts if x [4 ]<8 ]
    hi =[x for x in parts if x [4 ]>=8 ]
    sel =([lo [i ]for i in rng .choice (len (lo ),70 ,replace =False )]+
    [hi [i ]for i in rng .choice (len (hi ),30 ,replace =False )])
    test_fams ={family_key (p [1 ])for p in sel }
    thr =big_thr (NPZ ,test_fams )
    d =np .load (NPZ ,allow_pickle =True )
    keep =~np .isin (d ["fams"].astype (str ),list (test_fams ))
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (d ["X"][keep ],d ["y"][keep ])
    print (f"{len (sel )} part | esikler {thr } | gate {int (keep .sum ())} candidate",flush =True )

    if os .path .exists (CACHE ):
        cache =pickle .load (open (CACHE ,"rb"))
        print (f"cache diskten: {len (cache )} part (inference tekrar kosmuyor)",flush =True )
    else :
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
                    pbs .append (np .asarray (pb ,np .float32 ))
                j =json .load (open (jf ,encoding ="utf-8-sig"))
                Vj =np .array ([[q [c ]for c in "XYZ"]for q in j ["Graphic3d"]["Points"]],float )
                G =np .array ([[c ["Point"][q ]for q in "XYZ"]for c in j ["ConnectionPoints"]],float )
                Gd =np .array ([[c ["InsertDirection"][q ]for q in "XYZ"]for c in j ["ConnectionPoints"]],float )
                Gd /=np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 
                R ,t ,_ =align_frames (Vr ,Vj )
                cache .append (dict (V =V .astype (np .float32 ),F =F .astype (np .int32 ),
                pbs =[p_ .astype (np .float16 )for p_ in pbs ],stp =stp ,n =n ,
                G =(G -t )@R ,Gd =Gd @R ,
                diag =float (np .linalg .norm (V .max (0 )-V .min (0 )))))
            except Exception :
                continue 
            if len (cache )%25 ==0 :
                print (f"  {len (cache )} part",flush =True )
        pickle .dump (cache ,open (CACHE ,"wb"))
        print (f"cache yazildi: {len (cache )} part",flush =True )

        # sinif oncullerini EGITIM verisinden not, inference olasiliklarinin ortalamasindan al
    acc =np .zeros (5 )
    for r in cache :
        for pb in r ["pbs"]:
            acc +=np .asarray (pb ,np .float64 ).mean (0 )
    prior =acc /acc .sum ()
    print (f"sinif oncul (ortalama olasilik): {np .round (prior ,4 ).tolist ()}",flush =True )
    print (f"  CE={CE } CT={CT } -> baglanti siniflarinin payi %{100 *(prior [CE ]+prior [CT ]):.1f}\n",
    flush =True )

    def score (alpha ):
        agg ={"low":[0 ,0 ,0 ],"very":[0 ,0 ,0 ]}
        w =1.0 /np .maximum (prior ,1e-9 )**alpha 
        for r in cache :
            V =np .ascontiguousarray (r ["V"],np .float64 );F =np .ascontiguousarray (r ["F"],np .int64 )
            der =[]
            for pb in r ["pbs"]:
                q =np .asarray (pb ,np .float64 )*w [None ,:]
                der .append (cp_openings .connection_points (
                V ,F ,q .argmax (-1 ),min_v =MINV ,classes =(CE ,CT ),dedupe_mm =10.0 ,
                probs =np .asarray (pb ,np .float64 ),vertex_conf =VC ,ct_depth_min_mm =1.0 ,
                cluster_mm =CL ,step_path =r ["stp"]))
            base =robot_cp ._vote2 (der ,min_votes =1 )
            is_hi =robot_cp ._highcp_router (r ["stp"],V ,len (base ))
            # URUNLE AYNI must be: router very-CP derse promote with YENIDEN turet.
            # Ilk surumde is_hi hesaplanip KULLANILMIYORDU -> alpha=0 tabani 0.7518 output,
            # oysa urunun real tabani 0.7784. Goreli etki for zararsizdi but mutlak
            # sayilari karsilastirilamaz yapiyordu.
            if is_hi :
                pr =0.25 
                der2 =[cp_openings .connection_points (
                V ,F ,(np .asarray (pb ,np .float64 )*w [None ,:]).argmax (-1 ),
                min_v =MINV ,classes =(CE ,CT ),dedupe_mm =10.0 ,
                probs =np .asarray (pb ,np .float64 ),vertex_conf =VC ,ct_depth_min_mm =1.0 ,
                cluster_mm =CL ,conn_promote =pr ,step_path =r ["stp"])for pb in r ["pbs"]]
                cps =robot_cp ._vote2 (der2 ,min_votes =1 )
            else :
                cps =base 
            kept =[]
            if cps :
                probs =sum (np .asarray (p_ ,np .float64 )for p_ in r ["pbs"])/len (r ["pbs"])
                sc =clf .predict_proba (wire_gate .feats_for (V ,F ,probs ,cps ,CE ,CT ))[:,1 ]
                t_ =thr ["very"]if is_hi else thr ["low"]
                kept =[c for c ,s_ in zip (cps ,sc )if s_ >=t_ ]
            Q =np .array ([c ["point"]for c in kept ],float )if kept else np .zeros ((0 ,3 ))
            G ,Gd =r ["G"],r ["Gd"]
            hit =np .zeros (len (G ),bool );used =set ()
            if len (Q )and len (G ):
                diff =Q [:,None ,:]-G [None ,:,:]
                al =(diff *Gd [None ,:,:]).sum (-1 )
                pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
                tol =max (3.0 ,0.06 *r ["diag"])
                pe =np .where (np .abs (al )<=40.0 ,pe ,np .inf )
                for d_ ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )for a in range (len (Q ))for b in range (len (G ))):
                    if d_ >tol or a_ in used or hit [b_ ]:
                        continue 
                    hit [b_ ]=True ;used .add (a_ )
            tp =int (hit .sum ());k ="very"if r ["n"]>=8 else "low"
            agg [k ][0 ]+=tp ;agg [k ][1 ]+=len (Q )-tp ;agg [k ][2 ]+=len (G )-tp 
        o ={}
        TP =FP =FN =0 
        for k ,(T ,Fp ,Fn )in agg .items ():
            p =T /max (T +Fp ,1 );rc =T /max (T +Fn ,1 )
            o [k ]=2 *p *rc /max (p +rc ,1e-9 )
            TP +=T ;FP +=Fp ;FN +=Fn 
        return (sum (W [k ]*o [k ]for k in W ),o ["low"],o ["very"],
        TP /max (TP +FP ,1 ),TP /max (TP +FN ,1 ))

    print (f"{'alpha':>7}{'tespit F1':>11}{'low':>9}{'very':>9}{'precision':>10}{'recall':>9}")
    res ={}
    for al in (0.0 ,0.15 ,0.3 ,0.5 ,0.7 ,1.0 ):
        w_ ,lo_ ,hi_ ,p_ ,r_ =score (al )
        res [al ]=dict (f1 =w_ ,dusuk =lo_ ,cok =hi_ ,precision =p_ ,recall =r_ )
        star ="  <- mevcut"if al ==0.0 else ""
        print (f"{al :>7.2f}{w_ :>11.4f}{lo_ :>9.4f}{hi_ :>9.4f}{p_ :>10.3f}{r_ :>9.3f}{star }",flush =True )
    json .dump ({"prior":prior .tolist (),"sonuc":{str (k ):v for k ,v in res .items ()}},
    open ("results/h_sinif_onculu.json","w"),indent =1 )
    print ("\nmakbuz -> results/h_sinif_onculu.json")


if __name__ =="__main__":
    main ()
