# -*- coding: utf-8 -*-
"""Failure taxonomy for the CP model (audit P0). For every part (all splits) it matches model CPs
to GT CPs (cp-v2 CableEntry, 5mm) and CLASSIFIES each error:
  FP: duplicate_cp   (within 8mm of a TP -> over-split of one opening)
      side_rail_fp   (predicted ten a SnapPoint/rail region)
      wrong_surface  (predicted ten Housing/LabelSurface -> not a connection)
      spurious_fp    (none of the above)
  FN: missed_cp      (a GT opening with no model prediction near it)
  TP: bad_direction  (position matched but approach angle vs GT > 30 deg)
Outputs results/cp_failure_set.json (worst parts first) + a printed summary. This turns "one bad
picture" into a measurable pattern.

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe cp_failure_taxonomy.py
"""
import json ,os 
import numpy as np 
from scipy .spatial import cKDTree 
import scheffler_dataset as dataset 
import diffusionnet ,metrics ,cp_openings ,connector3d 

CE =int (connector3d .CABLE_ENTRY );S =int (connector3d .SNAP_POINT )
H =int (connector3d .HOUSING );LS =int (connector3d .LABEL_SURFACE )
OP ="results/scheffler_semantic/operators";CKPT ="results/scheffler_semantic/refit91.pt"
MATCH =5.0 


def label_at (V ,L ,tree ,p ):
    return int (L [tree .query (p )[0 if False else 1 ]])


def main ():
    model ,meta ,_ =diffusionnet .load_checkpoint (CKPT ,device ="cuda")
    counts ={};parts =[]
    for sp in ("train","val","test_locked"):
        for s in dataset .load_split ("wscad_corpus_scheffler_exact",sp ,allow_locked =True ,verify_hashes =False ):
            pid =s ["part_id"];V =np .asarray (s ["verts"],float );F =np .asarray (s ["faces"],int );L =np .asarray (s ["labels"])
            tree =cKDTree (V )
            gcps =cp_openings .connection_points (V ,F ,L ,min_v =1 ,classes =(CE ,))
            gt =np .array ([c ["point"]for c in gcps ]or []).reshape (-1 ,3 )
            gdir =np .array ([c ["direction"]for c in gcps ]or []).reshape (-1 ,3 )
            plab ,probs =diffusionnet .predict (model ,meta ,V ,F ,device ="cuda",op_cache_dir =OP ,return_probs =True )
            pcps =cp_openings .connection_points (V ,F ,np .asarray (plab ),min_v =20 ,probs =probs ,vertex_conf =0.9 ,classes =(CE ,))
            pr =np .array ([c ["point"]for c in pcps ]or []).reshape (-1 ,3 )
            pdir =np .array ([c ["direction"]for c in pcps ]or []).reshape (-1 ,3 )
            errs =[]
            if len (gt )and len (pr ):
                m ,up ,ug =metrics .match_predictions (pr ,gt ,MATCH )
                mp ={i for i ,_ ,_ in m };mg ={j for _ ,j ,_ in m }
                for i ,j ,_ in m :
                    ang =np .degrees (np .arccos (np .clip (np .dot (pdir [i ],gdir [j ]),-1 ,1 )))
                    if ang >30 :
                        errs .append ({"kind":"bad_direction","angle_deg":round (float (ang ),1 )})
                up =[i for i in range (len (pr ))if i not in mp ];ug =[j for j in range (len (gt ))if j not in mg ]
            else :
                up =list (range (len (pr )));ug =list (range (len (gt )));m =[]
            tp_pts =pr [[i for i ,_ ,_ in m ]]if len (m )else np .zeros ((0 ,3 ))
            for i in up :# FP
                p =pr [i ]
                dtp =float (np .linalg .norm (tp_pts -p ,axis =1 ).min ())if len (tp_pts )else 1e9 
                lab =int (L [tree .query (p )[1 ]])
                kind =("duplicate_cp"if dtp <8 else "side_rail_fp"if lab ==S 
                else "wrong_surface"if lab in (H ,LS )else "spurious_fp")
                errs .append ({"kind":kind ,"nearest_tp_mm":round (dtp ,1 ),"label_here":int (lab )})
            for j in ug :# FN
                errs .append ({"kind":"missed_cp","point":[round (float (x ),1 )for x in gt [j ]]})
            for e in errs :
                counts [e ["kind"]]=counts .get (e ["kind"],0 )+1 
            if errs :
                parts .append ({"part_id":pid ,"split":sp ,"gt":len (gt ),"model":len (pr ),
                "n_err":len (errs ),"errors":errs })
    parts .sort (key =lambda x :-x ["n_err"])
    out ={"frozen":"2026-07-18","match_mm":MATCH ,"cp_def":"cp-v2 CableEntry",
    "error_counts":counts ,"n_parts_with_errors":len (parts ),
    "failure_set":parts [:30 ]}
    os .makedirs ("results",exist_ok =True )
    json .dump (out ,open ("results/cp_failure_set.json","w"),indent =1 )
    print ("ERROR COUNTS:",counts )
    print (f"{len (parts )} parts have >=1 error. failure_set (worst 30) -> results/cp_failure_set.json")
    for p in parts [:12 ]:
        kinds =",".join (sorted (set (e ["kind"]for e in p ["errors"])))
        print (f"  {p ['part_id']:<11}{p ['split']:<12} GT{p ['gt']} model{p ['model']}  errs={p ['n_err']} [{kinds }]")


if __name__ =="__main__":
    main ()
