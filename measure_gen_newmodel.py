# -*- coding: utf-8 -*-
"""Re-measure generalisation ten the 40 unseen candidates with the RETRAINED (234-part) model, and
compare to the 102-part baseline (18 clean / 6 noisy / 16 no_cableentry). Same verdict logic as
measure_generalization.py.

CORRECTION (2026-07-20, cp-v3): this "clean/no_cableentry" verdict is a HEURISTIC ten UNLABELLED
parts, still deliberately CableEntry-only (classes=(CE,) below) for comparability with the earlier
receipts above -- it is NOT the CP definition (see cp_openings.py for cp-v3: CPs can be CableEntry
OR depth-gated Contact). This heuristic was also shown to be ANTI-CORRELATED with real human-GT
segmentation quality (rec_augcew_s0 scored best here but worst ten measure_val_cableentry.py's
honest Connection IoU) -- do NOT use this script's verdict to select a product model. Use
measure_val_cableentry.py (human-GT val, Connection IoU) for that; this script is kept only for
the historical clean/noisy/no_cableentry trend line.
Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe measure_gen_newmodel.py [--ckpt ...]
"""
import argparse ,os ,json 
import numpy as np 
import torch 
import diffusionnet as D 
import connector3d ,cp_openings ,thesis_remesh 
from infer_step_cp import step_to_mesh 

CE =int (connector3d .CABLE_ENTRY );OP ="results/step_infer/ops"
NAMES ={int (getattr (connector3d ,n )):n for n in ("HOUSING","CONTACT","SNAP_POINT","CABLE_ENTRY","LABEL_SURFACE")}


def predict (model ,meta ,V ,F ,dev ):
    V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
    ops =D .precompute_operators (V ,F ,meta .get ("n_eig",48 ),op_cache_dir =OP )
    ops ={k :(v .to (dev )if hasattr (v ,"to")else v )for k ,v in ops .items ()}
    with torch .no_grad ():
        return D ._forward (model ,ops ,D ._model_input (ops ,meta )).argmax (-1 ).cpu ().numpy ()


def main ():
    ap =argparse .ArgumentParser ();ap .add_argument ("--ckpt",default ="results/seg_extra/best.pt");a =ap .parse_args ()
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    ck =torch .load (a .ckpt ,map_location =dev ,weights_only =False )
    meta =ck ["meta"];meta .setdefault ("n_eig",ck ["cfg"].get ("n_eig",48 ))
    model ,_ =D .build_diffusionnet (ck ["cfg"],n_classes =5 );model .load_state_dict (ck ["state"]);model .to (dev ).eval ()
    import hashlib 
    setname ="gen_dev_40.json"if os .path .exists ("gen_dev_40.json")else "benchmark_candidates.json"
    cand =json .load (open (setname ))["candidates"]
    def sha (p ):return hashlib .sha256 (open (p ,"rb").read ()).hexdigest ()[:16 ]if os .path .exists (p )else None 
    clean =noisy =none =err =0 ;rows =[]
    for c in cand :
        pid =c ["part_id"]
        try :
            Vr ,Fr =step_to_mesh (c ["step_file"]);V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            plab =predict (model ,meta ,V ,F ,dev )
        except Exception as e :
            err +=1 ;rows .append ({"part_id":pid ,"error":str (e )[:60 ]});continue 
        cef =[f for f in connector3d .build_fragments (V ,F ,plab ,min_vertices =5 )if int (f .label )==CE ]
        ce_tot =int ((plab ==CE ).sum ());n_comp =len (cef )
        big =max ((len (f .idx )for f in cef ),default =0 );frac =big /max (ce_tot ,1 )
        cps =cp_openings .connection_points (V ,F ,plab ,min_v =20 ,classes =(CE ,))
        v ="no_cableentry"if ce_tot ==0 else ("clean"if (n_comp <=8 and frac >=0.25 and 1 <=len (cps )<=12 )else "noisy")
        clean +=v =="clean";noisy +=v =="noisy";none +=v =="no_cableentry"
        rows .append ({"part_id":pid ,"verdict":v ,"cp_count":len (cps ),
        "class_counts":{NAMES [k ]:int ((plab ==k ).sum ())for k in NAMES }})
        print (f"  {pid }: cp={len (cps )} -> {v }",flush =True )
    rep ={"metric":"generalisation_heuristic (NOT a scored benchmark; verdict is a heuristic, needs visual/CP-count confirm)",
    "candidate_set":setname ,"candidate_set_role":"gen_dev (used for model selection -> NOT final benchmark)",
    "ckpt":a .ckpt ,"ckpt_sha16":sha (a .ckpt ),"candidate_set_sha16":sha (setname ),
    "clean":clean ,"noisy":noisy ,"no_cableentry":none ,"err":err ,"n":len (cand ),
    "baseline_102part":{"clean":18 ,"noisy":6 ,"no_cableentry":16 },"per_part":rows }
    os .makedirs ("results/gen_eval",exist_ok =True )
    out =os .path .join ("results/gen_eval",os .path .basename (a .ckpt ).replace (".pt","")+f"_{os .path .basename (setname ).replace ('.json','')}.json")
    json .dump (rep ,open (out ,"w"),indent =1 )
    print (f"\nNEW MODEL: clean {clean } | noisy {noisy } | no_cableentry {none } | err {err }  -> {out }")
    print ("BASELINE (102-part): clean 18 | noisy 6 | no_cableentry 16")


if __name__ =="__main__":
    main ()
