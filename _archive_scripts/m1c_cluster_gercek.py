# -*- coding: utf-8 -*-
"""M1c: regime-kosullu cluster_mm (low-CP 5.0 / very-CP 3.0) GERCEK boru hattinda.

M1b (single turetme): low-CP 5.0 with +0.019, very-CP 3.0'da kalmali (-0.062 zarar).
Rejim ayrimi this projede two times odedi (min_v, threshold). Ama single-turetme olcumu this night two times
yaniltti -> real 4-model union hattinda dogrulanmadan urune girmez.

Dagitilan mantigi taklit eder: before default turetme -> router -> rejime according to YENIDEN derive.
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
    MINV =int (pp ["min_vertices"]);VC =float (pp ["vertex_confidence_mask"])
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
    rng =np .random .RandomState (3 )
    lo =[x for x in parts if x [4 ]<8 ];hi =[x for x in parts if x [4 ]>=8 ]
    sel =([lo [i ]for i in rng .choice (len (lo ),34 ,replace =False )]+
    [hi [i ]for i in rng .choice (len (hi ),22 ,replace =False )])
    print (f"{len (sel )} part (34 dusuk / 22 very)",flush =True )

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

    def derive (r ,cl ,promote ):
        return [cp_openings .connection_points (
        r ["V"],r ["F"],pb .argmax (-1 ),min_v =MINV ,classes =(CE ,CT ),dedupe_mm =10.0 ,
        probs =pb ,vertex_conf =VC ,ct_depth_min_mm =1.0 ,cluster_mm =cl ,
        conn_promote =promote )for pb in r ["pbs"]]

    def score (cl_low ,cl_high ):
        agg ={"low":[0 ,0 ,0 ],"very":[0 ,0 ,0 ]}
        for r in cache :
            probs =sum (r ["pbs"])/len (r ["pbs"])
            # 1) default turetme -> router (dagitilan mantik)
            base =robot_cp ._vote2 (derive (r ,cl_low ,0.0 ),min_votes =1 )
            is_hi =robot_cp ._highcp_router (None ,r ["V"],len (base ))
            cl =cl_high if is_hi else cl_low 
            cps =robot_cp ._vote2 (derive (r ,cl ,0.25 if is_hi else 0.0 ),min_votes =1 )
            s_ =wire_gate .apply (r ["V"],r ["F"],probs ,copy .deepcopy (cps ),CE ,CT ,
            threshold =(TH if is_hi else TL ),top_n =None )if cps else []
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
        out ["weighted"]=sum (W [k ]*out [k ]for k in W )
        return out 

    print (f"{'cluster (low/very)':<22}{'low':>9}{'very':>9}{'agirlikli':>11}")
    res ={}
    for cl_lo ,cl_hi ,tag in ((3.0 ,3.0 ,"3.0 / 3.0  (URUNDE)"),(5.0 ,3.0 ,"5.0 / 3.0  (candidate)"),
    (5.0 ,5.0 ,"5.0 / 5.0")):
        r =score (cl_lo ,cl_hi );res [tag ]=r 
        print (f"{tag :<22}{r ['low']:>9.4f}{r ['very']:>9.4f}{r ['weighted']:>11.4f}",flush =True )
    b =res ["3.0 / 3.0  (URUNDE)"]["weighted"]
    c =res ["5.0 / 3.0  (candidate)"]["weighted"]
    print (f"\nrejim-kosullu katki: {c -b :+.4f}")
    print (f"KAPI (>= +0.01): {'GECTI'if c -b >=0.01 else 'OLU'}")
    json .dump ({k :v for k ,v in res .items ()},open ("results/m1c_cluster.json","w"),indent =1 )
    print ("receipt -> results/m1c_cluster.json")


if __name__ =="__main__":
    main ()
