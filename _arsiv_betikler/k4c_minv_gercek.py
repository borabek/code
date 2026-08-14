# -*- coding: utf-8 -*-
"""K4c: min_v 10 -> 4 degisikligini GERCEK BORU HATTINDA dogrula (4-model union).

WHY ZORUNLU: K1'de gate-data seviyesinde olculen +0.0096, calisan hatta -0.0010 output.
Onbellek only ORTALAMA olasiligi tutuyor; min_v whereas HER MODELIN own turetmesinde
uygulaniyor, i.e. union yolu onbellekten dogrulanamaz. Tam inference sart.
"""
import os ,sys ,json ,copy 
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
    cfg =json .load (open ("cp_config.json"));pp =cfg ["prediction_postproc"]
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    models =[load_any (c ,dev =dev )[:2 ]for c in cfg ["robot_vote2_checkpoints"]]
    LOCK =set (json .load (open ("results/split_lock.json"))["locked_parts"])
    parts =[]
    for m ,p ,jf ,s in eligible ():
        if p in LOCK :continue 
        try :n =len (json .load (open (jf ,encoding ="utf-8-sig"))["ConnectionPoints"])
        except Exception :continue 
        if n >0 :parts .append ((m ,p ,jf ,s ,n ))
    rng =np .random .RandomState (0 )
    lo =[x for x in parts if x [4 ]<8 ];hi =[x for x in parts if x [4 ]>=8 ]
    sel =([lo [i ]for i in rng .choice (len (lo ),24 ,replace =False )]+
    [hi [i ]for i in rng .choice (len (hi ),20 ,replace =False )])
    print (f"{len (sel )} part (24 dusuk / 20 cok)",flush =True )

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
            j =json .load (open (jf ,encoding ="utf-8-sig"))
            Vj =np .array ([[q [c ]for c in "XYZ"]for q in j ["Graphic3d"]["Points"]],float )
            G =np .array ([[c ["Point"][q ]for q in "XYZ"]for c in j ["ConnectionPoints"]],float )
            Gd =np .array ([[c ["InsertDirection"][q ]for q in "XYZ"]for c in j ["ConnectionPoints"]],float )
            Gd /=np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 
            R ,t ,_ =align_frames (Vr ,Vj )
            cache .append (dict (V =V ,F =F ,pbs =pbs ,G =(G -t )@R ,Gd =Gd @R ,n =n ,
            tol =max (3.0 ,0.06 *float (np .linalg .norm (V .max (0 )-V .min (0 )))),
            regime ="very"if n >=8 else "low"))
        except Exception :
            continue 
    print (f"cache: {len (cache )} part\n",flush =True )

    def score (minv ):
        agg ={"low":[0 ,0 ,0 ],"very":[0 ,0 ,0 ]}
        for r in cache :
            hi_ =r ["regime"]=="very"
            per =[cp_openings .connection_points (
            r ["V"],r ["F"],pb .argmax (-1 ),min_v =minv ,classes =(CE ,CT ),dedupe_mm =10.0 ,
            probs =pb ,vertex_conf =float (pp ["vertex_confidence_mask"]),
            ct_depth_min_mm =1.0 ,cluster_mm =float (pp ["cluster_mm"]),
            conn_promote =(0.25 if hi_ else 0.0 ))for pb in r ["pbs"]]
            cps =robot_cp ._vote2 (per ,min_votes =1 )
            probs =sum (r ["pbs"])/len (r ["pbs"])
            thr =0.25 if hi_ else float (cfg .get ("robot_wire_gate_threshold",0.40 ))
            s_ =wire_gate .apply (r ["V"],r ["F"],probs ,copy .deepcopy (cps ),CE ,CT ,
            threshold =thr ,top_n =None )if cps else []
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

    print (f"{'min_v':>7}{'low-F1':>10}{'very-F1':>9}{'agirlikli':>11}")
    res ={}
    for mv in (4 ,6 ,10 ):
        r =score (mv );res [mv ]=r 
        mk ="  <- urunde"if mv ==10 else ""
        print (f"{mv :>7}{r ['low']:>10.4f}{r ['very']:>9.4f}{r ['weighted']:>11.4f}{mk }",flush =True )
    d =res [4 ]["weighted"]-res [10 ]["weighted"]
    print (f"\nmin_v 10 -> 4 GERCEK HATTA: {d :+.4f}")
    json .dump ({str (k ):v for k ,v in res .items ()},open ("results/k4c_minv.json","w"),indent =1 )
    print ("receipt -> results/k4c_minv.json")


if __name__ =="__main__":
    main ()
