# -*- coding: utf-8 -*-
"""PREVIEW CP-F1 ten the dev VAL split (NOT the burned locked-11). Runs refit91.pt, turns
its predicted CableEntry vertices into CP points via the SAME clustering as
derive_scheffler_cp_candidates (connected components -> ConnectionPoint.entry_point), and
matches them against the human-CableEntry-derived CP points at a 5 mm radius. Honest caveats:
(1) the GT here is human-region-derived, not human-clicked/confirmed CPs; (2) VAL was used
for epoch selection, so this is a dev preview, not a sealed CP benchmark.

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe eval_scheffler_cp_f1.py
"""
import sys 
import numpy as np 
import scheffler_dataset as dataset 
import diffusionnet 
import connector3d 
import metrics 

CORPUS ="wscad_corpus_scheffler_exact"
CKPT ="results/scheffler_semantic/refit91.pt"
OPCACHE ="results/scheffler_semantic/operators"
SPLIT =sys .argv [1 ]if len (sys .argv )>1 else "val"
MIN_V =int (sys .argv [2 ])if len (sys .argv )>2 else 1 # predicted-component size filter (val-selected: 10)
MATCH_MM =5.0 
CABLE =int (connector3d .CABLE_ENTRY )


def cps_from_labels (verts ,faces ,labels ,min_v =1 ):
    frags =connector3d .build_fragments (verts ,faces ,labels ,min_vertices =min_v )
    bc =np .asarray (verts ,float ).mean (0 )
    pts =[]
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


model ,meta ,cfg =diffusionnet .load_checkpoint (CKPT ,device ="cuda")
samples =dataset .load_split (CORPUS ,SPLIT ,allow_locked =(SPLIT =="test_locked"),verify_hashes =True )
print (f"{SPLIT }: {len (samples )} parts\n")

TP =FP =FN =0 
per =[]
for s in samples :
    V ,Fc =np .asarray (s ["verts"],float ),np .asarray (s ["faces"],int )
    gt =cps_from_labels (V ,Fc ,s ["labels"],min_v =1 )# human GT unchanged
    pred_lbl =diffusionnet .predict (model ,meta ,V ,Fc ,device ="cuda",op_cache_dir =OPCACHE )
    pr =cps_from_labels (V ,Fc ,np .asarray (pred_lbl ),min_v =MIN_V )# cleanup ten predictions
    if len (gt )and len (pr ):
        m ,up ,ug =metrics .match_predictions (pr ,gt ,MATCH_MM )
        tp ,fp ,fn =len (m ),len (up ),len (ug )
    else :
        tp ,fp ,fn =0 ,len (pr ),len (gt )
    TP +=tp ;FP +=fp ;FN +=fn 
    per .append ((s ["part_id"],len (gt ),len (pr ),tp ,fp ,fn ))

f1 =2 *TP /max (2 *TP +FP +FN ,1 )
jac =TP /max (TP +FP +FN ,1 )
prec =TP /max (TP +FP ,1 );rec =TP /max (TP +FN ,1 )
print (f"{'part':<10}{'GT':>3}{'pred':>5}{'TP':>4}{'FP':>4}{'FN':>4}")
for pid ,ng ,npd ,tp ,fp ,fn in per :
    print (f"  {pid :<8}{ng :>3}{npd :>5}{tp :>4}{fp :>4}{fn :>4}")
print (f"\n>>> PREVIEW CP-F1 ({SPLIT }, human-region-derived GT, {MATCH_MM }mm):")
print (f"    F1={f1 :.3f}  Jaccard={jac :.3f}  precision={prec :.3f}  recall={rec :.3f}  TP={TP } FP={FP } FN={FN }")
print ("    caveat: GT is human-REGION-derived (not human-clicked); VAL is dev (epoch-selected).")
print ("DONE")
