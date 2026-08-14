# -*- coding: utf-8 -*-
"""Is min_v 10 recovering real small openings, or matching NOISE by luck?

CONCERN (user): min_v 10 accepts fragments of just 10 vertices out of 6000 (~0.17% of the mesh). A
noise speck that happens to fall near a manufacturer CP scores as TP under axis-aware matching, which
would look like a gain ten the arbiter but give jittery/unreliable CPs in the product.

TEST: for the product, derive CPs at min_v 10, then bucket each predicted CP by the vertex count of
its source fragment (10-20, 20-30, 30-45, 45+). Report, per bucket, how many are TP vs FP. If the
small buckets are mostly TP -> min_v 10 legitimately recovers small openings. If mostly FP -> it is
adding noise and min_v should go back up.

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe diag_minv.py [--ckpt ...] [--per-mfg 60]
"""
import os ,sys ,glob ,json ,argparse 
import numpy as np ,torch 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import diffusionnet as D ,connector3d ,cp_openings ,thesis_remesh 
from cad_eval import align_frames 
from infer_step_cp import step_to_mesh ,load_any 

CE =int (connector3d .CABLE_ENTRY );CT =int (connector3d .CONTACT )
OP ="results/step_infer/ops";DS ="_ds1/DataSet"
BUCKETS =[(10 ,20 ),(20 ,30 ),(30 ,45 ),(45 ,10 **9 )]


def eligible (mfg ):
    step ={os .path .basename (s ).split ("_")[1 ]:s for s in glob .glob ("all_wscad_stp/*.stp")}
    seen =set ()
    for d in ("_label_targets","_label_targets_2","_label_targets_3","_label_targets_4","_label_targets_recall"):
        seen |={os .path .basename (os .path .normpath (p ))for p in glob .glob (d +"/*/")}
    import scheffler_dataset as ds 
    for sp in ("train","val"):
        seen |={s ["part_id"]for s in ds .load_split ("wscad_corpus_scheffler_exact",sp ,verify_hashes =False )}
    out =[]
    for f in sorted (glob .glob (os .path .join (DS ,"*ElectricalTerminal*.json"))):
        head =os .path .basename (f ).split ("_")[0 ]
        m ,pid =(head .split (".",1 )+[""])[:2 ]
        if pid in seen or pid not in step or m !=mfg :continue 
        out .append ((pid ,f ,step [pid ]))
    return out 


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--ckpt",default ="results/seg_extra/human77c_s2.pt")
    ap .add_argument ("--per-mfg",type =int ,default =60 )
    ap .add_argument ("--device",default ="cuda"if torch .cuda .is_available ()else "cpu")
    a =ap .parse_args ()
    model ,meta ,_ =load_any (a .ckpt ,dev =a .device )
    counts ={b :[0 ,0 ]for b in BUCKETS }# [TP, FP] per size bucket

    for mfg in ("PXC","WEI"):
        for pid ,jf ,stp in eligible (mfg )[:a .per_mfg ]:
            try :
                j =json .load (open (jf ,encoding ="utf-8-sig"))
                G =np .array ([[c ["Point"]["X"],c ["Point"]["Y"],c ["Point"]["Z"]]for c in j .get ("ConnectionPoints",[])],float )
                Gd =np .array ([[c ["InsertDirection"]["X"],c ["InsertDirection"]["Y"],c ["InsertDirection"]["Z"]]for c in j .get ("ConnectionPoints",[])],float )
                if not len (G ):continue 
                Gd =Gd /(np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 )
                Vj =np .array ([[p ["X"],p ["Y"],p ["Z"]]for p in j ["Graphic3d"]["Points"]],float )
                Vr ,Fr =step_to_mesh (stp )
                V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
                V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
                _ ,probs =D .predict (model ,meta ,V ,F ,device =a .device ,op_cache_dir =OP ,return_probs =True )
                probs =np .asarray (probs ,float );lab =probs .argmax (-1 )
                cps =cp_openings .connection_points (V ,F ,lab ,min_v =10 ,classes =(CE ,CT ),dedupe_mm =10.0 ,
                probs =probs ,vertex_conf =0.5 ,ct_depth_min_mm =1.0 ,cluster_mm =5.0 )
                if not cps :continue 
                R ,t ,_ =align_frames (Vr ,Vj )
                Gs =(G -t )@R ;Gds =Gd @R 
                P =np .array ([np .asarray (c ["point"])for c in cps ],float )
                nv =np .array ([int (c .get ("n_verts",0 ))for c in cps ])
                tol =max (3.0 ,0.06 *float (np .linalg .norm (V .max (0 )-V .min (0 ))))
                # axis-aware greedy match, remembering which prediction won
                diff =P [:,None ,:]-Gs [None ,:,:]
                along =(diff *Gds [None ,:,:]).sum (-1 )
                dm =np .linalg .norm (diff -along [...,None ]*Gds [None ,:,:],axis =-1 )
                dm =np .where (np .abs (along )<=40.0 ,dm ,np .inf )
                order =sorted ((dm [i ,k ],i ,k )for i in range (len (P ))for k in range (len (Gs ))if dm [i ,k ]<=tol )
                up ,ug ,matched =set (),set (),set ()
                for d ,i ,k in order :
                    if i in up or k in ug :continue 
                    up .add (i );ug .add (k );matched .add (i )
                for i in range (len (P )):
                    for b in BUCKETS :
                        if b [0 ]<=nv [i ]<b [1 ]:
                            counts [b ][0 if i in matched else 1 ]+=1 
                            break 
            except Exception :
                continue 

    print (f"\nmin_v 10 -- tahmin CP'lerinin source fragment boyutuna per TP/FP dagilimi:")
    print (f"  {'boyut':>10s}  {'TP':>4s} {'FP':>4s}  {'precision':>9s}")
    for b in BUCKETS :
        tp ,fp =counts [b ]
        pr =tp /max (tp +fp ,1 )
        tag ="  <- min_v 30 bunlari ATIYORDU"if b [1 ]<=30 else ""
        print (f"  {b [0 ]:4d}-{b [1 ]if b [1 ]<10 **8 else '+':<4}  {tp :4d} {fp :4d}  {pr :9.2f}{tag }")
    small =[counts [b ]for b in BUCKETS if b [1 ]<=30 ]
    st ,sf =sum (x [0 ]for x in small ),sum (x [1 ]for x in small )
    print (f"\n  KUCUK fragmentler (10-30v, min_v 10'un ekledigi): TP {st } / FP {sf } -> precision {st /max (st +sf ,1 ):.2f}")
    print (f"  -> yuksekse gercek kucuk opening, dusukse noise")


if __name__ =="__main__":
    main ()
