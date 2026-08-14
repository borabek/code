# -*- coding: utf-8 -*-
"""M2 (conn_promote) + M4 (ince threshold izgarasi) -- YENI gate under, GERCEK boru hatti."""
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
    MINV =int (pp ["min_vertices"]);VC =float (pp ["vertex_confidence_mask"])
    CL =float (pp ["cluster_mm"])
    TL =float (cfg ["robot_wire_gate_threshold"]);TH =float (cfg ["robot_wire_gate_threshold_highcp"])
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    LOCK =set (json .load (open ("results/split_lock.json"))["locked_parts"])
    models =[load_any (c ,dev =dev )[:2 ]for c in cfg ["robot_vote2_checkpoints"]]

    parts =[]
    for m ,p ,jf ,s in eligible ():
        if p in LOCK :continue 
        try :n =len (json .load (open (jf ,encoding ="utf-8-sig"))["ConnectionPoints"])
        except Exception :continue 
        if n >0 :parts .append ((m ,p ,jf ,s ,n ))
    rng =np .random .RandomState (9 )
    lo =[x for x in parts if x [4 ]<8 ];hi =[x for x in parts if x [4 ]>=8 ]
    sel =([lo [i ]for i in rng .choice (len (lo ),34 ,replace =False )]+
    [hi [i ]for i in rng .choice (len (hi ),20 ,replace =False )])
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
            tol =max (3.0 ,0.06 *float (np .linalg .norm (V .max (0 )-V .min (0 ))))))
        except Exception :
            continue 
    print (f"cache {len (cache )} part\n",flush =True )

    def score (promote ,tl ,th ):
        agg ={"low":[0 ,0 ,0 ],"very":[0 ,0 ,0 ]}
        for r in cache :
            hi_ =r ["n"]>=8 
            per =[cp_openings .connection_points (
            r ["V"],r ["F"],pb .argmax (-1 ),min_v =MINV ,classes =(CE ,CT ),dedupe_mm =10.0 ,
            probs =pb ,vertex_conf =VC ,ct_depth_min_mm =1.0 ,cluster_mm =CL ,
            conn_promote =(promote if hi_ else 0.0 ))for pb in r ["pbs"]]
            cps =robot_cp ._vote2 (per ,min_votes =1 )
            probs =sum (r ["pbs"])/len (r ["pbs"])
            s_ =wire_gate .apply (r ["V"],r ["F"],probs ,copy .deepcopy (cps ),CE ,CT ,
            threshold =(th if hi_ else tl ),top_n =None )if cps else []
            Q =np .array ([c ["point"]for c in s_ ],float )if s_ else np .zeros ((0 ,3 ))
            G ,Gd =r ["G"],r ["Gd"];hit =np .zeros (len (G ),bool );used =set ()
            if len (Q )and len (G ):
                diff =Q [:,None ,:]-G [None ,:,:];al =(diff *Gd [None ,:,:]).sum (-1 )
                pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
                pe =np .where (np .abs (al )<=40.0 ,pe ,np .inf )
                for d_ ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )for a in range (len (Q ))for b in range (len (G ))):
                    if d_ >r ["tol"]or a_ in used or hit [b_ ]:continue 
                    hit [b_ ]=True ;used .add (a_ )
            tp =int (hit .sum ());k ="very"if hi_ else "low"
            agg [k ][0 ]+=tp ;agg [k ][1 ]+=len (Q )-tp ;agg [k ][2 ]+=len (G )-tp 
        out ={}
        for k ,(T ,Fp ,Fn )in agg .items ():
            p =T /max (T +Fp ,1 );rc =T /max (T +Fn ,1 )
            out [k ]=2 *p *rc /max (p +rc ,1e-9 )
        out ["w"]=sum (W [k ]*out [k ]for k in W )
        return out 

    res ={"base":score (0.25 ,TL ,TH )}
    print (f"TABAN (promote 0.25, threshold {TL }/{TH }): {res ['base']['w']:.4f}"
    f"  (dusuk {res ['base']['low']:.4f} very {res ['base']['very']:.4f})\n",flush =True )
    print ("M2 -- conn_promote:",flush =True )
    res ["m2"]={}
    for pr in (0.10 ,0.15 ,0.20 ,0.25 ,0.35 ):
        r =score (pr ,TL ,TH );res ["m2"][str (pr )]=r 
        mk ="  <- urunde"if pr ==0.25 else ""
        print (f"  {pr :.2f}  very {r ['very']:.4f}  agirlikli {r ['w']:.4f}{mk }",flush =True )
    print ("\nM4 -- ince threshold izgarasi (low-CP):",flush =True )
    res ["m4"]={}
    for tl in (0.375 ,0.40 ,0.425 ,0.45 ,0.475 ,0.50 ,0.55 ):
        r =score (0.25 ,tl ,TH );res ["m4"][str (tl )]=r 
        mk ="  <- urunde"if abs (tl -TL )<1e-9 else ""
        print (f"  {tl :.3f}  dusuk {r ['low']:.4f}  agirlikli {r ['w']:.4f}{mk }",flush =True )
    json .dump (res ,open ("results/m2m4.json","w"),indent =1 )
    print ("\nmakbuz -> results/m2m4.json",flush =True )


if __name__ =="__main__":
    main ()
