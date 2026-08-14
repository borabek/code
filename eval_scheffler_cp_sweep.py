# -*- coding: utf-8 -*-
"""Precision cleanup sweep for the CP layer. The preview CP-F1 had recall 0.96 but
precision 0.66 (22 FP from over-split / tiny spurious predicted CableEntry components).
This runs refit91.pt ONCE per val part (caches predicted labels), then sweeps two cleanup
knobs ten the PREDICTED CPs only -- min predicted-component size, and a greedy NMS merge
radius -- matches vs the human-CableEntry-derived CPs at 5 mm, and reports the F1 grid.
Dev preview ten val (not the sealed test); GT is human-region-derived.

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe eval_scheffler_cp_sweep.py
"""
import numpy as np 
import scheffler_dataset as dataset 
import diffusionnet ,connector3d ,metrics 

CORPUS ="wscad_corpus_scheffler_exact";CKPT ="results/scheffler_semantic/refit91.pt"
OPCACHE ="results/scheffler_semantic/operators";MATCH_MM =5.0 
CABLE =int (connector3d .CABLE_ENTRY )


def cps (verts ,faces ,labels ,min_v ):
    frags =connector3d .build_fragments (verts ,faces ,labels ,min_vertices =min_v )
    bc =np .asarray (verts ,float ).mean (0 );pts =[]
    for f in frags :
        if int (f .label )!=CABLE :
            continue 
        cp =connector3d .ConnectionPoint ([f ])
        try :
            cp .compute_direction (verts ,faces ,smooth_subdiv =0 ,body_center =bc )
        except Exception :
            pass 
        pts .append (np .asarray (cp .entry_point ,float ))
    return np .array (pts ,float )if pts else np .zeros ((0 ,3 ))


def nms (pts ,radius ):
    if len (pts )==0 or radius <=0 :
        return pts 
    keep =[]
    for p in pts :
        if all (np .linalg .norm (p -k )>radius for k in keep ):
            keep .append (p )
    return np .array (keep ,float )if keep else np .zeros ((0 ,3 ))


model ,meta ,_ =diffusionnet .load_checkpoint (CKPT ,device ="cuda")
samples =dataset .load_split (CORPUS ,"val",verify_hashes =True )
cache =[]# (V, F, gt_cps, pred_labels)
for s in samples :
    V ,F =np .asarray (s ["verts"],float ),np .asarray (s ["faces"],int )
    gt =cps (V ,F ,s ["labels"],1 )
    pl =np .asarray (diffusionnet .predict (model ,meta ,V ,F ,device ="cuda",op_cache_dir =OPCACHE ))
    cache .append ((V ,F ,gt ,pl ))
print (f"val {len (cache )} parts inferred once; sweeping min-size x NMS\n")

print (f"  {'min_v':>5}{'nms':>5}{'F1':>8}{'prec':>7}{'recall':>8}   TP/FP/FN")
best =None 
for min_v in (1 ,5 ,10 ,20 ,40 ):
    for nms_mm in (0.0 ,3.0 ,5.0 ,8.0 ):
        TP =FP =FN =0 
        for V ,F ,gt ,pl in cache :
            pr =nms (cps (V ,F ,pl ,min_v ),nms_mm )
            if len (gt )and len (pr ):
                m ,up ,ug =metrics .match_predictions (pr ,gt ,MATCH_MM )
                TP +=len (m );FP +=len (up );FN +=len (ug )
            else :
                FP +=len (pr );FN +=len (gt )
        f1 =2 *TP /max (2 *TP +FP +FN ,1 );pr_ =TP /max (TP +FP ,1 );rc =TP /max (TP +FN ,1 )
        mark =""
        if best is None or f1 >best [2 ]:
            best =(min_v ,nms_mm ,f1 );mark ="  <=="
        print (f"  {min_v :>5}{nms_mm :>5.0f}{f1 :>8.3f}{pr_ :>7.3f}{rc :>8.3f}   {TP }/{FP }/{FN }{mark }")

print (f"\n>>> BEST: min_v={best [0 ]} nms={best [1 ]:.0f}mm  F1={best [2 ]:.3f}  (baseline min_v=1/nms=0: F1 0.782)")
print ("DONE")
