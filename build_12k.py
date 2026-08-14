# -*- coding: utf-8 -*-
"""Build a higher-resolution copy of everything we train/measure ten, with the labels carried over.

WHY (evidence, 2026-07-22): the CPs the product still misses are the SMALL openings. Measured ten the
82-CP held-out: openings the model FINDS have a median 212 vertices, the ones it MISSES only 84 --
2.5x fewer -- and it gives them connection-probability 0.19 vs 0.87. So they are plausibly
UNDER-RESOLVED at the 6000-vertex remesh. Raising the target puts a missed opening nearer the range the
model handles well. MEASURED CAP: training memory scales as n * k_eig * width, and 11254 verts
(target 12000) blows the 4GB T1200 (3964 MiB, 0 epochs in 3 min, classic WDDM sysmem spill).
Inference is cheap at any of these (10k = 97 MB); it is BACKPROP that does not fit. 9000 -> ~9067
verts fits.

Labels are per-vertex, so they cannot be reused directly: each 12k vertex takes the label of its
NEAREST 6k vertex (the meshes describe the same surface, so this is a faithful transfer).

Outputs load_extra-style dirs:  <out>/<pid>/{<pid>.obj, <pid>.labels.txt}
Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe build_12k.py
"""
import os ,sys ,glob ,json ,time 
import numpy as np 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import thesis_remesh 
from infer_step_cp import step_to_mesh 
from region_label_helper import load_obj 

import argparse 
_ap =argparse .ArgumentParser ();_ap .add_argument ("--target",type =int ,default =9000 );_ap .add_argument ("--tag",default ="9k")
_A ,_ =_ap .parse_known_args ()
TARGET =_A .target ;TAG =_A .tag 
POOL ={os .path .basename (s ).split ("_")[1 ]:s for s in glob .glob ("all_wscad_stp/*.stp")}


def save_obj (path ,V ,F ):
    with open (path ,"w")as f :
        for v in V :f .write (f"v {v [0 ]:.6f} {v [1 ]:.6f} {v [2 ]:.6f}\n")
        for t in F :f .write (f"f {t [0 ]+1 } {t [1 ]+1 } {t [2 ]+1 }\n")


def transfer (V6 ,L6 ,V12 ,chunk =4096 ):
    """Nearest-neighbour label transfer 6k -> 12k, chunked so a 12k x 6k distance matrix never
    materialises in one go."""
    out =np .empty (len (V12 ),dtype =np .int64 )
    for i in range (0 ,len (V12 ),chunk ):
        blk =V12 [i :i +chunk ]
        d =np .linalg .norm (blk [:,None ,:]-V6 [None ,:,:],axis =2 )
        out [i :i +chunk ]=L6 [d .argmin (1 )]
    return out 


def build_one (pid ,step ,V6 ,L6 ,outdir ):
    Vr ,Fr =step_to_mesh (step )
    V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =TARGET )
    V =np .ascontiguousarray (V ,float );F =np .ascontiguousarray (F ,np .int64 )
    L =transfer (np .asarray (V6 ,float ),np .asarray (L6 ,np .int64 ),V )
    d =os .path .join (outdir ,pid );os .makedirs (d ,exist_ok =True )
    save_obj (os .path .join (d ,f"{pid }.obj"),V ,F )
    open (os .path .join (d ,f"{pid }.labels.txt"),"w").write ("\n".join (map (str ,L .tolist ())))
    return len (V ),int ((L ==3 ).sum ())


def main ():
    t0 =time .time ();done =fail =0 
    # 1) corpus train + val (FULL 5-class labels)
    import scheffler_dataset as ds 
    for split ,out in (("train",f"_corpus{TAG }_train"),("val",f"_corpus{TAG }_val")):
        os .makedirs (out ,exist_ok =True )
        for s in ds .load_split ("wscad_corpus_scheffler_exact",split ,verify_hashes =False ):
            pid =s ["part_id"];step =s .get ("step_path")or POOL .get (pid )
            if not step or not os .path .exists (step ):
                print (f"  {pid }: STEP yok, atlandi");fail +=1 ;continue 
            try :
                nv ,nce =build_one (pid ,step ,s ["verts"],s ["labels"],out )
                done +=1 
                if done %20 ==0 :print (f"  ... {done } part ({time .time ()-t0 :.0f}s)",flush =True )
            except Exception as e :
                print (f"  {pid }: HATA {str (e )[:45 ]}");fail +=1 
                # 2) the human partial-label batches
    for src in ("_label_targets","_label_targets_2","_label_targets_3","_label_targets_4"):
        out =src +"_"+TAG ;os .makedirs (out ,exist_ok =True )
        for d in sorted (glob .glob (os .path .join (src ,"*"))):
            if not os .path .isdir (d ):continue 
            pid =os .path .basename (os .path .normpath (d ))
            of ,lf =os .path .join (d ,f"{pid }.obj"),os .path .join (d ,f"{pid }.labels.txt")
            if not (os .path .exists (of )and os .path .exists (lf )):continue 
            step =POOL .get (pid )
            if not step :# batch-1 parts may come from another candidate set
                pv =os .path .join (d ,f"{pid }.provenance.json")
                if os .path .exists (pv ):
                    sp =json .load (open (pv )).get ("source_step")or ""
                    step =sp if os .path .exists (sp )else POOL .get (os .path .basename (sp ).split ("_")[1 ]if "_"in sp else "")
            if not step or not os .path .exists (step ):
                print (f"  {pid }: STEP yok, atlandi");fail +=1 ;continue 
            V6 ,_ =load_obj (of )
            L6 =np .array ([int (x )for x in open (lf ).read ().split ()],np .int64 )
            if len (L6 )!=len (V6 ):print (f"  {pid }: vertex uyusmazligi");fail +=1 ;continue 
            try :
                nv ,nce =build_one (pid ,step ,V6 ,L6 ,out )
                done +=1 
                if done %20 ==0 :print (f"  ... {done } part ({time .time ()-t0 :.0f}s)",flush =True )
            except Exception as e :
                print (f"  {pid }: HATA {str (e )[:45 ]}");fail +=1 
    print (f"\n12k veri seti hazir: {done } part, {fail } atlandi, {time .time ()-t0 :.0f}s")


if __name__ =="__main__":
    main ()
