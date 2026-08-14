# -*- coding: utf-8 -*-
"""GECE OLCUMU: bugunku urunun GENIS ornekte CP-F1'i (44 not, 120 part).

WHY: gunun butun kararlari 44-64 parcalik orneklerde alindi and same yapilandirma different
orneklerde 0.65-0.78 arasi degerler verdi (ornekleme gurultusu ~±0.04). "Urun rakami" demek
for more genis a measurement sart.

SIZINTI: gate 1903 parcayla egitildi, i.e. this parcalari gordu -> LEHINE leakage present.
Bu yuzden sonuc URUN RAKAMI DEGIL, a UST SINIR as raporlanir. Sizintisiz number
K2c'dir (0.7817, 44 part, test aileleri disarida).
"""
import os ,sys ,json ,copy 
import numpy as np 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
W ={"dusuk":0.895 ,"cok":0.105 }


def main ():
    import torch ,thesis_remesh ,cp_openings ,robot_cp ,wire_gate ,diffusionnet as D 
    from cad_eval import align_frames 
    from infer_step_cp import step_to_mesh ,load_any 
    from big_arbiter import eligible 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    cfg =json .load (open ("cp_config.json"));pp =cfg ["prediction_postproc"]
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    LOCK =set (json .load (open ("results/split_lock.json"))["locked_parts"])
    models =[load_any (c ,dev =dev )[:2 ]for c in cfg ["robot_vote2_checkpoints"]]
    thr_lo =float (cfg ["robot_wire_gate_threshold"]);thr_hi =float (cfg ["robot_wire_gate_threshold_highcp"])

    parts =[]
    for m ,p ,jf ,s in eligible ():
        if p in LOCK :continue 
        try :n =len (json .load (open (jf ,encoding ="utf-8-sig"))["ConnectionPoints"])
        except Exception :continue 
        if n >0 :parts .append ((m ,p ,jf ,s ,n ))
    rng =np .random .RandomState (7 )
    lo =[x for x in parts if x [4 ]<8 ];hi =[x for x in parts if x [4 ]>=8 ]
    sel =([lo [i ]for i in rng .choice (len (lo ),80 ,replace =False )]+
    [hi [i ]for i in rng .choice (len (hi ),40 ,replace =False )])
    print (f"{len (sel )} part (80 dusuk / 40 cok) | threshold {thr_lo }/{thr_hi } | min_v {pp ['min_vertices']}",
    flush =True )

    agg ={"dusuk":[0 ,0 ,0 ],"cok":[0 ,0 ,0 ]}
    per_part =[]
    for i ,(mfg ,pid ,jf ,stp ,n )in enumerate (sel ,1 ):
        try :
            Vr ,Fr =step_to_mesh (stp )
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
            hi_ =n >=8 
            pbs =[]
            for model ,meta in models :
                _ ,pb =D .predict (model ,meta ,V ,F ,device =dev ,
                op_cache_dir =f"results/step_infer/ops_k{int (meta .get ('k_eig',64 ))}",
                return_probs =True )
                pbs .append (np .asarray (pb ,float ))
            per =[cp_openings .connection_points (
            V ,F ,pb .argmax (-1 ),min_v =int (pp ["min_vertices"]),classes =(CE ,CT ),
            dedupe_mm =10.0 ,probs =pb ,vertex_conf =float (pp ["vertex_confidence_mask"]),
            ct_depth_min_mm =1.0 ,cluster_mm =float (pp ["cluster_mm"]),
            conn_promote =(0.25 if hi_ else 0.0 ))for pb in pbs ]
            cps =robot_cp ._vote2 (per ,min_votes =1 )
            probs =sum (pbs )/len (pbs )
            s_ =wire_gate .apply (V ,F ,probs ,copy .deepcopy (cps ),CE ,CT ,
            threshold =(thr_hi if hi_ else thr_lo ),top_n =None )if cps else []
            Q =np .array ([c ["point"]for c in s_ ],float )if s_ else np .zeros ((0 ,3 ))
            j =json .load (open (jf ,encoding ="utf-8-sig"))
            Vj =np .array ([[q [c ]for c in "XYZ"]for q in j ["Graphic3d"]["Points"]],float )
            G =np .array ([[c ["Point"][q ]for q in "XYZ"]for c in j ["ConnectionPoints"]],float )
            Gd =np .array ([[c ["InsertDirection"][q ]for q in "XYZ"]for c in j ["ConnectionPoints"]],float )
            Gd /=np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 
            R ,t ,_ =align_frames (Vr ,Vj )
            Gm =(G -t )@R ;Gdm =Gd @R 
            tol =max (3.0 ,0.06 *float (np .linalg .norm (V .max (0 )-V .min (0 ))))
            hit =np .zeros (len (Gm ),bool );used =set ()
            if len (Q )and len (Gm ):
                diff =Q [:,None ,:]-Gm [None ,:,:];al =(diff *Gdm [None ,:,:]).sum (-1 )
                pe =np .linalg .norm (diff -al [...,None ]*Gdm [None ,:,:],axis =-1 )
                pe =np .where (np .abs (al )<=40.0 ,pe ,np .inf )
                for d_ ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )for a in range (len (Q ))for b in range (len (Gm ))):
                    if d_ >tol or a_ in used or hit [b_ ]:continue 
                    hit [b_ ]=True ;used .add (a_ )
            tp =int (hit .sum ());k ="cok"if hi_ else "dusuk"
            agg [k ][0 ]+=tp ;agg [k ][1 ]+=len (Q )-tp ;agg [k ][2 ]+=len (Gm )-tp 
            pr =tp /max (len (Q ),1 );rc =tp /max (len (Gm ),1 )
            per_part .append ({"pid":pid ,"regime":k ,"gt":len (Gm ),"pred":len (Q ),
            "f1":2 *pr *rc /max (pr +rc ,1e-9 )})
        except Exception :
            continue 
        if i %20 ==0 :
            print (f"  {i }/{len (sel )}",flush =True )

    out ={}
    for k ,(T ,Fp ,Fn )in agg .items ():
        p =T /max (T +Fp ,1 );r =T /max (T +Fn ,1 )
        out [k ]={"P":p ,"R":r ,"F1":2 *p *r /max (p +r ,1e-9 ),"TP":T ,"FP":Fp ,"FN":Fn }
    wf1 =sum (W [k ]*out [k ]["F1"]for k in W )
    print (f"\n{'regime':<8}{'part':>7}{'P':>8}{'R':>8}{'F1':>9}")
    for k in ("dusuk","cok"):
        n =sum (1 for r in per_part if r ["regime"]==k )
        print (f"{k :<8}{n :>7}{out [k ]['P']:>8.3f}{out [k ]['R']:>8.3f}{out [k ]['F1']:>9.4f}")
    print (f"\nKORPUS-AGIRLIKLI CP-F1: {wf1 :.4f}   ({len (per_part )} part)")
    print ("UYARI: gate bu parcalari egitimde gordu -> UST SINIR. Sizintisiz sayi K2c = 0.7817")
    json .dump ({"weighted":wf1 ,"regimes":out ,"n_parts":len (per_part ),
    "per_part":per_part ,"leakage":"gate saw these parts; upper bound"},
    open ("results/gece_urun_olcum.json","w"),indent =1 )
    print ("receipt -> results/gece_urun_olcum.json")


if __name__ =="__main__":
    main ()
