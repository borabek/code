# -*- coding: utf-8 -*-
"""D1 ADIL KARSILASTIRMA: two gate de AYNI kosulda (test aileleri haric) egitilir.
Tek difference: hangi MANTIKLA uretilmis adaylarla egitildikleri.

  A) ESKI mantik verisi  (gate_regrow_data_minv4) -- promote YOK, bbox yonlendirmesi
  B) YENI mantik verisi  (gate_regrow_data_rt2)   -- promote uyumlu, geometrik yonlendirme

Calisma anindaki candidates HER IKI kolda da YENI mantikla uretilir (urun residual boyle calisiyor).
Boylece measured_path sey saf: gate'in EGITIM dagilimi with CALISMA dagiliminin uyusmasi ne kazandiriyor.
"""
import os ,sys ,json ,copy ,pickle 
import numpy as np 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
W ={"low":0.895 ,"very":0.105 }


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
    models =[load_any (c ,dev =dev )[:2 ]for c in cfg ["robot_vote2_checkpoints"]]

    parts =[]
    for m ,p ,jf ,s in eligible ():
        if p in LOCK :continue 
        try :n =len (json .load (open (jf ,encoding ="utf-8-sig"))["ConnectionPoints"])
        except Exception :continue 
        if n >0 :parts .append ((m ,p ,jf ,s ,n ))
    rng =np .random .RandomState (21 )
    lo =[x for x in parts if x [4 ]<8 ];hi =[x for x in parts if x [4 ]>=8 ]
    sel =([lo [i ]for i in rng .choice (len (lo ),30 ,replace =False )]+
    [hi [i ]for i in rng .choice (len (hi ),18 ,replace =False )])
    tf ={family_key (p [1 ])for p in sel }
    print (f"{len (sel )} test parcasi, {len (tf )} aile each IKI gate'in egitiminden de cikarildi",
    flush =True )

    gates ={}
    for tag ,npz in (("A_eski_mantik","results/gate_regrow_data_minv4.npz"),
    ("B_yeni_mantik","results/gate_regrow_data_rt2.npz")):
        d =np .load (npz ,allow_pickle =True )
        keep =~np .isin (d ["fams"].astype (str ),list (tf ))
        clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
        random_state =0 ).fit (d ["X"][keep ],d ["y"][keep ])
        path =f"results/_d1_{tag }.pkl"
        pickle .dump ({"clf":clf ,"feat_names":wire_gate .FEAT_NAMES },open (path ,"wb"))
        gates [tag ]=path 
        print (f"  {tag }: {int (keep .sum ())} candidate with egitildi",flush =True )

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
                # CALISMA mantigi: promote'suz derive -> router -> very-CP whereas promote'lu
            def der (pr ):
                return [cp_openings .connection_points (
                V ,F ,q .argmax (-1 ),min_v =MINV ,classes =(CE ,CT ),dedupe_mm =10.0 ,
                probs =q ,vertex_conf =VC ,ct_depth_min_mm =1.0 ,cluster_mm =CL ,
                conn_promote =pr )for q in pbs ]
            base =robot_cp ._vote2 (der (0.0 ),min_votes =1 )
            is_hi =robot_cp ._highcp_router (stp ,V ,len (base ))
            cps =robot_cp ._vote2 (der (0.25 ),min_votes =1 )if is_hi else base 
            j =json .load (open (jf ,encoding ="utf-8-sig"))
            Vj =np .array ([[q [c ]for c in "XYZ"]for q in j ["Graphic3d"]["Points"]],float )
            G =np .array ([[c ["Point"][q ]for q in "XYZ"]for c in j ["ConnectionPoints"]],float )
            Gd =np .array ([[c ["InsertDirection"][q ]for q in "XYZ"]for c in j ["ConnectionPoints"]],float )
            Gd /=np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 
            R ,t ,_ =align_frames (Vr ,Vj )
            cache .append (dict (V =V ,F =F ,probs =sum (pbs )/len (pbs ),cps =cps ,is_hi =is_hi ,
            G =(G -t )@R ,Gd =Gd @R ,n =n ,
            tol =max (3.0 ,0.06 *float (np .linalg .norm (V .max (0 )-V .min (0 ))))))
        except Exception :
            continue 
    print (f"cache {len (cache )} part\n",flush =True )

    def score (path ,tl ,th ):
        agg ={"low":[0 ,0 ,0 ],"very":[0 ,0 ,0 ]}
        for r in cache :
            thr =th if r ["is_hi"]else tl 
            s_ =wire_gate .apply (r ["V"],r ["F"],r ["probs"],copy .deepcopy (r ["cps"]),CE ,CT ,
            threshold =thr ,model_path =path ,top_n =None )if r ["cps"]else []
            Q =np .array ([c ["point"]for c in s_ ],float )if s_ else np .zeros ((0 ,3 ))
            G ,Gd =r ["G"],r ["Gd"];hit =np .zeros (len (G ),bool );used =set ()
            if len (Q )and len (G ):
                diff =Q [:,None ,:]-G [None ,:,:];al =(diff *Gd [None ,:,:]).sum (-1 )
                pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
                pe =np .where (np .abs (al )<=40.0 ,pe ,np .inf )
                for d_ ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )for a in range (len (Q ))for b in range (len (G ))):
                    if d_ >r ["tol"]or a_ in used or hit [b_ ]:continue 
                    hit [b_ ]=True ;used .add (a_ )
            tp =int (hit .sum ());k ="very"if r ["n"]>=8 else "low"
            agg [k ][0 ]+=tp ;agg [k ][1 ]+=len (Q )-tp ;agg [k ][2 ]+=len (G )-tp 
        out ={}
        for k ,(T ,Fp ,Fn )in agg .items ():
            p =T /max (T +Fp ,1 );rc =T /max (T +Fn ,1 )
            out [k ]=2 *p *rc /max (p +rc ,1e-9 )
        out ["w"]=sum (W [k ]*out [k ]for k in W )
        return out 

    print (f"{'gate training dagilimi':<24}{'threshold':>10}{'low':>9}{'very':>9}{'agirlikli':>11}")
    res ={}
    for tag ,path in gates .items ():
        best =None 
        for tl in (0.35 ,0.40 ,0.45 ,0.50 ):
            for th in (0.20 ,0.25 ,0.30 ):
                r =score (path ,tl ,th )
                if best is None or r ["w"]>best [0 ]["w"]:
                    best =(r ,tl ,th )
        r ,tl ,th =best ;res [tag ]={"r":r ,"thr":[tl ,th ]}
        print (f"{tag :<24}{f'{tl :.2f}/{th :.2f}':>10}{r ['low']:>9.4f}{r ['very']:>9.4f}{r ['w']:>11.4f}",
        flush =True )
    dlt =res ["B_yeni_mantik"]["r"]["w"]-res ["A_eski_mantik"]["r"]["w"]
    print (f"\nB - A = {dlt :+.4f}   (ADIL: ikisi de test aileleri haric egitildi)")
    print (f"KAPI (>= +0.02): {'GECTI'if dlt >=0.02 else 'OLU'}")
    json .dump ({k :{"w":v ["r"]["w"],"low":v ["r"]["low"],"very":v ["r"]["very"],
    "thr":v ["thr"]}for k ,v in res .items ()}|{"delta":dlt },
    open ("results/d1_fair.json","w"),indent =1 )
    print ("receipt -> results/d1_fair.json")


if __name__ =="__main__":
    main ()
