# -*- coding: utf-8 -*-
"""UNION ensemble plato-kirici (ROBOT_STRATEJI backup plani #2).

Softmax-ORTALAMA (big_arbiter'in coklu-ckpt yolu) recall'i kesiyordu -> became.
TERSI never denenmedi: each model KENDI CP'lerini turetir, after CP'lerin BIRLESIMI alinir
(a CP'yi HERHANGI a model bulduysa tut), yakinlik with dedup. Recall'i maksimize eder.
Ayni kosuda hem single modelleri hem birlesimi same part setinde skorlar -> adil comparison.

Kullanim: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe union_arbiter.py \
    --ckpts A.pt B.pt C.pt --only-mfg WEI --only-parts $(cat _hw_r3.txt) --axis-aware \
    --cluster-mm 5 --min-v 30 --vertex-conf 0.5 --tag union_wei
"""
import os ,sys ,json ,argparse ,time 
import numpy as np ,torch 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import diffusionnet as D ,cp_openings 
import thesis_remesh 
from cad_eval import align_frames 
from infer_step_cp import step_to_mesh ,load_any 
from big_arbiter import greedy ,eligible ,CE ,CT ,OP 


def derive (V ,F ,probs ,min_v ,vc ,cluster_mm ):
    lab =probs .argmax (-1 )
    return cp_openings .connection_points (V ,F ,lab ,min_v =min_v ,classes =(CE ,CT ),dedupe_mm =10.0 ,
    probs =probs ,vertex_conf =vc ,ct_depth_min_mm =1.0 ,cluster_mm =cluster_mm )


def union_cps (cp_lists ,cluster_mm ,min_votes =1 ):
    """Pool CPs from all models; greedy dedup by 3D proximity, keep highest-confidence per cluster.
    votes = how many models found a CP within cluster_mm of this one. min_votes filters agreement."""
    allc =[c for lst in cp_lists for c in lst ]
    allc .sort (key =lambda c :-float (c .get ("confidence",0.0 )))
    kept =[]
    for c in allc :
        p =np .asarray (c ["point"],float )
        hit =None 
        for k in kept :
            if np .linalg .norm (p -np .asarray (k ["point"],float ))<=cluster_mm :
                hit =k ;break 
        if hit is None :
            c =dict (c );c ["_votes"]=1 ;kept .append (c )
        else :
            hit ["_votes"]+=1 
    return [c for c in kept if c ["_votes"]>=min_votes ]


def score (parts ,models ,a ,mode ,min_votes =1 ):
    """mode: index into models (single) or 'union' (with min_votes agreement filter)."""
    T =Fp =Fn =0 ;rows =[]
    for mfg ,pid ,jf ,stp ,V ,F ,Vr ,Vj ,G ,Gd ,per_model_cps in parts :
        if mode =="union":
            cps =union_cps (per_model_cps ,a .cluster_mm ,min_votes )
        else :
            cps =per_model_cps [mode ]
        R ,t ,_ =align_frames (Vr ,Vj )
        P =(np .array ([np .asarray (c ["point"])for c in cps ],float )@R .T +t )if cps else np .zeros ((0 ,3 ))
        tol =max (3.0 ,0.06 *float (np .linalg .norm (Vj .max (0 )-Vj .min (0 ))))
        tp ,fp ,fn =greedy (P ,G ,tol ,Gd ,a .axis_tol )if a .axis_aware else greedy (P ,G ,tol )
        T +=tp ;Fp +=fp ;Fn +=fn 
        rows .append ({"part_id":pid ,"tp":tp ,"fp":fp ,"fn":fn })
    pr =T /max (T +Fp ,1 );rc =T /max (T +Fn ,1 );f1 =2 *pr *rc /max (pr +rc ,1e-9 )
    return f1 ,pr ,rc ,T ,Fp ,Fn ,rows 


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--ckpts",nargs ="+",required =True )
    ap .add_argument ("--only-mfg",default ="")
    ap .add_argument ("--only-parts",nargs ="+",default =[])
    ap .add_argument ("--axis-aware",action ="store_true")
    ap .add_argument ("--axis-tol",type =float ,default =40.0 )
    ap .add_argument ("--cluster-mm",type =float ,default =5.0 )
    ap .add_argument ("--min-v",type =int ,default =30 )
    ap .add_argument ("--vertex-conf",type =float ,default =0.5 )
    ap .add_argument ("--remesh-target",type =int ,default =6000 )
    ap .add_argument ("--tag",default ="union")
    ap .add_argument ("--device",default ="cuda"if torch .cuda .is_available ()else "cpu")
    a =ap .parse_args ()

    models =[load_any (c ,dev =a .device )for c in a .ckpts ]
    os .environ ["BA_ALLOW_SEEN"]="1"# --only-parts is already a leakage-free explicit list
    raw =eligible ()
    if a .only_mfg :raw =[p for p in raw if p [0 ]==a .only_mfg ]
    if a .only_parts :
        keep =set (a .only_parts );raw =[p for p in raw if p [1 ]in keep ]
    print (f"{len (raw )} eligible part | {len (models )} model UNION | cluster {a .cluster_mm } min_v {a .min_v }",flush =True )

    # each parcayi a times remesh + each modelden CP derive (cache)
    parts =[];t0 =time .time ()
    for k ,(mfg ,pid ,jf ,stp )in enumerate (raw ,1 ):
        try :
            j =json .load (open (jf ,encoding ="utf-8-sig"))
            Vj =np .array ([[p ["X"],p ["Y"],p ["Z"]]for p in j ["Graphic3d"]["Points"]],float )
            G =np .array ([[c ["Point"]["X"],c ["Point"]["Y"],c ["Point"]["Z"]]for c in j .get ("ConnectionPoints",[])],float )
            Gd =np .array ([[c ["InsertDirection"]["X"],c ["InsertDirection"]["Y"],c ["InsertDirection"]["Z"]]for c in j .get ("ConnectionPoints",[])],float )
            if not len (G ):continue 
            Gd =Gd /(np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 )
            Vr ,Fr =step_to_mesh (stp )
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =a .remesh_target )
            V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
            per_model =[]
            for model ,meta ,_ in models :
                _ ,pb =D .predict (model ,meta ,V ,F ,device =a .device ,op_cache_dir =OP ,return_probs =True )
                per_model .append (derive (V ,F ,np .asarray (pb ,float ),a .min_v ,a .vertex_conf ,a .cluster_mm ))
            parts .append ((mfg ,pid ,jf ,stp ,V ,F ,Vr ,Vj ,G ,Gd ,per_model ))
        except Exception :
            continue 
        if k %25 ==0 :
            print (f"  {k }/{len (raw )} turetildi  {time .time ()-t0 :.0f}s",flush =True )

    print (f"\n=== SONUC ({len (parts )} part) — axis_aware={a .axis_aware } ===")
    for i ,c in enumerate (a .ckpts ):
        f1 ,pr ,rc ,T ,Fp ,Fn ,_ =score (parts ,models ,a ,i )
        print (f"  TEK  {os .path .basename (c ):28s}  F1={f1 :.3f}  P={pr :.3f}  R={rc :.3f}  (TP{T } FP{Fp } FN{Fn })")
    n =len (models );best =None ;sweep ={}
    for mv in range (1 ,n +1 ):
        f1 ,pr ,rc ,T ,Fp ,Fn ,rows =score (parts ,models ,a ,"union",mv )
        tag =f"vote>={mv }"+("  (full union)"if mv ==1 else "  (all of them anlasir)"if mv ==n else "")
        print (f"  UNION {tag :22s}  F1={f1 :.3f}  P={pr :.3f}  R={rc :.3f}  (TP{T } FP{Fp } FN{Fn })")
        sweep [f"vote{mv }"]={"f1":f1 ,"p":pr ,"r":rc }
        if best is None or f1 >best [0 ]:best =(f1 ,mv )
    print (f"  --> en iyi union F1={best [0 ]:.3f} @ vote>={best [1 ]}  (tek urun recall_hard_s2 = ilk satir)")
    json .dump ({"tag":a .tag ,"ckpts":a .ckpts ,"sweep":sweep ,"best_vote":best [1 ]},
    open (f"results/union_{a .tag }.json","w"),indent =1 )
    print (f"  -> results/union_{a .tag }.json")


if __name__ =="__main__":
    main ()
