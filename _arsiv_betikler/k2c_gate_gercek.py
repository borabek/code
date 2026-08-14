# -*- coding: utf-8 -*-
"""K2c: new gate'i GERCEK boru hattinda dogrula -- SIZINTISIZ.

SORUN: results/wire_gate_regrow_minv4.pkl TUM 1903 parcayla egitildi, i.e. test edecegim
parcalari gordu. Dogrulama sizintili olurdu.

COZUM: test parcalarinin AILELERINI disarida birakan ayri a gate egit, after real
4-model union boru hattinda two gate'i karsilastir. Kilitli holdout'a DOKUNULMAZ.
"""
import os ,sys ,json ,copy ,pickle 
import numpy as np 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
W ={"dusuk":0.895 ,"cok":0.105 }
HOLD_GATE ="results/_d1_gate_heldout.pkl"


def main ():
    import torch ,thesis_remesh ,cp_openings ,robot_cp ,wire_gate ,diffusionnet as D 
    from cad_eval import align_frames 
    from infer_step_cp import step_to_mesh ,load_any 
    from big_arbiter import eligible 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from json_dataset import family_key 
    from sklearn .ensemble import RandomForestClassifier 

    cfg =json .load (open ("cp_config.json"));pp =cfg ["prediction_postproc"]
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    LOCK =set (json .load (open ("results/split_lock.json"))["locked_parts"])

    parts =[]
    for m ,p ,jf ,s in eligible ():
        if p in LOCK :continue 
        try :n =len (json .load (open (jf ,encoding ="utf-8-sig"))["ConnectionPoints"])
        except Exception :continue 
        if n >0 :parts .append ((m ,p ,jf ,s ,n ))
    rng =np .random .RandomState (1 )
    lo =[x for x in parts if x [4 ]<8 ];hi =[x for x in parts if x [4 ]>=8 ]
    sel =([lo [i ]for i in rng .choice (len (lo ),26 ,replace =False )]+
    [hi [i ]for i in rng .choice (len (hi ),18 ,replace =False )])
    test_fams ={family_key (p [1 ])for p in sel }
    print (f"{len (sel )} test parcasi, {len (test_fams )} aile disarida birakiliyor",flush =True )

    # --- test ailelerini HARIC tutarak gate egit ---
    d =np .load ("results/gate_regrow_data_rt2.npz",allow_pickle =True )
    X ,y ,fams =d ["X"],d ["y"],d ["fams"]
    keep =~np .isin (fams .astype (str ),list (test_fams ))
    print (f"gate egitimi: {int (keep .sum ())} candidate ({len (y )-int (keep .sum ())} tanesi test ailesi, cikarildi)",
    flush =True )
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (X [keep ],y [keep ])
    pickle .dump ({"clf":clf ,"feat_names":wire_gate .FEAT_NAMES },open (HOLD_GATE ,"wb"))

    models =[load_any (c ,dev =dev )[:2 ]for c in cfg ["robot_vote2_checkpoints"]]
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
            hi_ =n >=8 
            per =[cp_openings .connection_points (
            V ,F ,pb .argmax (-1 ),min_v =int (pp ["min_vertices"]),classes =(CE ,CT ),
            dedupe_mm =10.0 ,probs =pb ,vertex_conf =float (pp ["vertex_confidence_mask"]),
            ct_depth_min_mm =1.0 ,cluster_mm =float (pp ["cluster_mm"]),
            conn_promote =(0.25 if hi_ else 0.0 ))for pb in pbs ]
            cps =robot_cp ._vote2 (per ,min_votes =1 )
            j =json .load (open (jf ,encoding ="utf-8-sig"))
            Vj =np .array ([[q [c ]for c in "XYZ"]for q in j ["Graphic3d"]["Points"]],float )
            G =np .array ([[c ["Point"][q ]for q in "XYZ"]for c in j ["ConnectionPoints"]],float )
            Gd =np .array ([[c ["InsertDirection"][q ]for q in "XYZ"]for c in j ["ConnectionPoints"]],float )
            Gd /=np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 
            R ,t ,_ =align_frames (Vr ,Vj )
            cache .append (dict (V =V ,F =F ,probs =sum (pbs )/len (pbs ),cps =cps ,
            G =(G -t )@R ,Gd =Gd @R ,regime ="cok"if hi_ else "dusuk",
            tol =max (3.0 ,0.06 *float (np .linalg .norm (V .max (0 )-V .min (0 ))))))
        except Exception :
            continue 
    print (f"cache: {len (cache )} part\n",flush =True )

    def score (model_path ,thr_low ,thr_hi ):
        agg ={"dusuk":[0 ,0 ,0 ],"cok":[0 ,0 ,0 ]}
        for r in cache :
            thr =thr_hi if r ["regime"]=="cok"else thr_low 
            s_ =wire_gate .apply (r ["V"],r ["F"],r ["probs"],copy .deepcopy (r ["cps"]),CE ,CT ,
            threshold =thr ,model_path =model_path ,top_n =None )if r ["cps"]else []
            Q =np .array ([c ["point"]for c in s_ ],float )if s_ else np .zeros ((0 ,3 ))
            G ,Gd =r ["G"],r ["Gd"];hit =np .zeros (len (G ),bool );used =set ()
            if len (Q )and len (G ):
                diff =Q [:,None ,:]-G [None ,:,:];al =(diff *Gd [None ,:,:]).sum (-1 )
                pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
                pe =np .where (np .abs (al )<=40.0 ,pe ,np .inf )
                for d_ ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )for a in range (len (Q ))for b in range (len (G ))):
                    if d_ >r ["tol"]or a_ in used or hit [b_ ]:continue 
                    hit [b_ ]=True ;used .add (a_ )
            tp =int (hit .sum ());k =r ["regime"]
            agg [k ][0 ]+=tp ;agg [k ][1 ]+=len (Q )-tp ;agg [k ][2 ]+=len (G )-tp 
        out ={}
        for k ,(T ,Fp ,Fn )in agg .items ():
            p =T /max (T +Fp ,1 );rc =T /max (T +Fn ,1 )
            out [k ]=2 *p *rc /max (p +rc ,1e-9 )
        out ["weighted"]=sum (W [k ]*out [k ]for k in W )
        return out 

    print (f"{'gate':<34}{'threshold':>10}{'dusuk':>9}{'cok':>9}{'agirlikli':>11}")
    res ={}
    dep =score ("results/wire_gate.pkl",0.40 ,0.25 )
    res ["deployed"]=dep 
    print (f"{'A) DAGITILAN':<34}{'0.40/0.25':>10}{dep ['dusuk']:>9.4f}{dep ['cok']:>9.4f}"
    f"{dep ['weighted']:>11.4f}",flush =True )
    # DURUST arm: threshold BUYUK veride (1903 part, aile-disi OOF) secildi, test kumesine BAKILMADI.
    # K2d olcumu: low-CP 0.45 / very-CP 0.25.
    r =score (HOLD_GATE ,0.45 ,0.25 )
    res ["refit"]=r ;res ["refit_thr"]=[0.45 ,0.25 ]
    print (f"{'B) YENI (threshold buyuk veriden)':<34}{'0.45/0.25':>10}{r ['dusuk']:>9.4f}"
    f"{r ['cok']:>9.4f}{r ['weighted']:>11.4f}")
    # REFERANS: test kumesinde secilseydi ne olurdu (IYIMSER, rapor edilmez)
    best =None 
    for tl in (0.35 ,0.40 ,0.45 ,0.50 ):
        for th in (0.20 ,0.25 ,0.30 ):
            rr =score (HOLD_GATE ,tl ,th )
            if best is None or rr ["weighted"]>best [0 ]["weighted"]:
                best =(rr ,tl ,th )
    res ["refit_testselected"]={"weighted":best [0 ]["weighted"],"thr":[best [1 ],best [2 ]]}
    print (f"{'   (test-secimli, IYIMSER)':<34}{f'{best [1 ]:.2f}/{best [2 ]:.2f}':>10}"
    f"{best [0 ]['dusuk']:>9.4f}{best [0 ]['cok']:>9.4f}{best [0 ]['weighted']:>11.4f}")
    dlt =r ["weighted"]-dep ["weighted"]
    print (f"\nB - A = {dlt :+.4f}")
    print (f"KAPI (>= +0.02): {'GECTI'if dlt >=0.02 else 'OLU'}")
    res ["delta"]=dlt 
    json .dump (res ,open ("results/d1_gate_gercek.json","w"),indent =1 )
    print ("receipt -> results/d1_gate_gercek.json")


if __name__ =="__main__":
    main ()
