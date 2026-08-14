# -*- coding: utf-8 -*-
"""Mine the model's MISSES: manufacturer ConnectionPoints that the product finds NOTHING near.

WHY THIS AND NOT MORE ADJUDICATION: two adjudication rounds judged what the model FOUND and 183 of
189 were real openings (96%, 98%) -- so the model's precision is, in reality, ~98% and the measured
"false positives" are mostly our own incomplete labels. That means the remaining error is almost
entirely RECALL (0.63-0.67): openings the model never proposes. Adjudication cannot reach them,
because there is nothing to adjudicate.

But the manufacturer data can: for every ConnectionPoint the product fails to match, we KNOW an
opening exists there. That turns an impossible search task into a trivial painting task --
    manufacturer gives the LOCATION, the human gives the SHAPE.
which is exactly the split that defeated four attempts at generating regions automatically
(IoU vs human labels never passed 0.18).

Bonus: WEI parts have NO human labels at all today, and WEI is where the model collapses
(F1 0.233 vs PXC 0.780), so painting a few of these also puts the first human WEI supervision
into training.

Output: results/misses.json -- one entry per missed manufacturer CP, in MESH coordinates
(align_frames applied), ready for the painting tool.

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe mine_misses.py \
         [--ckpts ...] [--mfg WEI] [--limit 60]
"""
import os ,sys ,glob ,json ,argparse 
import numpy as np ,torch 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import diffusionnet as D ,connector3d ,cp_openings ,thesis_remesh 
from cad_eval import align_frames 
from infer_step_cp import step_to_mesh ,load_any 

CE =int (connector3d .CABLE_ENTRY );CT =int (connector3d .CONTACT )
OP ="results/step_infer/ops";DS ="_ds1/DataSet"


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--ckpts",nargs ="+",
    default =[f"results/seg_extra/adj_s{i }.pt"for i in (0 ,1 ,2 )])
    ap .add_argument ("--mfg",default ="",help ="restrict to one manufacturer (WEI = the weak one)")
    ap .add_argument ("--limit",type =int ,default =0 ,help ="max PARTS to scan")
    ap .add_argument ("--hard",action ="store_true",
    help ="prioritise parts where the model MISSES THE MOST (lowest recall), e.g. the "
    "32-CP WEI part where it found only 2. Sorts eligible parts by miss-count "
    "descending. These complex/rare parts are the recall bottleneck and yield "
    "many recall labels each. Overrides --diverse ordering.")
    ap .add_argument ("--diverse",action ="store_true",
    help ="sample parts across the WHOLE family spectrum instead of the first N "
    "alphabetically. LEARNED: the first WEI recall round trained ten 27 parts "
    "that were ALL one sub-group / almost all 2-CP / part numbers 101x-103x -- a "
    "single narrow family, so the model learns that family, not the general "
    "concept. Diverse sampling spreads across sub-group, CP-count, and size, and "
    "deliberately includes rare configurations.")
    ap .add_argument ("--max-per-part",type =int ,default =4 ,help ="do not flood one part")
    ap .add_argument ("--out",default ="results/misses.json")
    ap .add_argument ("--device",default ="cuda"if torch .cuda .is_available ()else "cpu")
    a =ap .parse_args ()

    step ={os .path .basename (s ).split ("_")[1 ]:s for s in glob .glob ("all_wscad_stp/*.stp")}
    seen =set ()
    for d in ("_label_targets","_label_targets_2","_label_targets_3","_label_targets_4",
    "_label_targets_recall","_label_targets_recall_pxc",
    "_QUARANTINE_labels/_label_targets_recall_pxc",
    "_QUARANTINE_labels/_label_targets_recall_v2","_label_targets_recall_hard"):
        seen |={os .path .basename (os .path .normpath (p ))for p in glob .glob (d +"/*/")}
    import scheffler_dataset as ds 
    for sp in ("train","val"):
        seen |={s ["part_id"]for s in ds .load_split ("wscad_corpus_scheffler_exact",sp ,verify_hashes =False )}

    def diversify (parts ):
        """Order parts to MAXIMISE variety: bucket by (sub-group, CP-count-band, size-band) from the
        JSON, then round-robin across buckets so the first N span the whole spectrum and rare buckets
        are represented, instead of N near-identical siblings."""
        import numpy as _np 
        buckets ={}
        for pid ,jf ,stp in parts :
            try :
                j =json .load (open (jf ,encoding ="utf-8-sig"))
                sub =j .get ("ProductSubGroup",0 )
                ncp =len (j .get ("ConnectionPoints",[]))
                bb =j .get ("BoundingBox",{})
                sz =0 
                if isinstance (bb ,dict ):
                    ext =[abs (bb .get (k ,0 ))for k in ("MaxX","MaxY","MaxZ")]
                    sz =int (max (ext )//20 )if any (ext )else 0 
                key =(sub ,min (ncp ,5 ),sz )
            except Exception :
                key =("?",0 ,0 )
            buckets .setdefault (key ,[]).append ((pid ,jf ,stp ))
            # RARE-FIRST round robin: order buckets by size (rarest first) so under-represented types
            # (high-CP multi-level/power/knife terminals -- only ~100 of 728 parts) get sampled BEFORE
            # the 416 simple 1-2 CP feed-throughs that dominate the catalogue. Geometric diversity is the
            # goal (robot must handle every terminal type), and rare types are the weakest coverage.
        bl =sorted (buckets .values (),key =len )# rarest bucket first
        order =[]
        i =0 
        while any (bl ):
            b =bl [i %len (bl )]
            if b :order .append (b .pop (0 ))
            i +=1 
            if i >100000 :break 
        return order 

    models =[load_any (c ,dev =a .device )for c in a .ckpts ]
    cand =[]
    for f in sorted (glob .glob (os .path .join (DS ,"*ElectricalTerminal*.json"))):
        head =os .path .basename (f ).split ("_")[0 ]
        mfg0 ,pid0 =(head .split (".",1 )+[""])[:2 ]
        if pid0 not in step or pid0 in seen :continue 
        if a .mfg and mfg0 !=a .mfg :continue 
        cand .append ((pid0 ,f ,step [pid0 ]))
    if a .hard :
    # sort by manufacturer-CP count DESC as a proxy for miss-count (high-CP parts are where recall
    # collapses -- the 32-CP WEI part found only 2). Complex parts first.
        def ncp (item ):
            try :return len (json .load (open (item [1 ],encoding ="utf-8-sig")).get ("ConnectionPoints",[]))
            except Exception :return 0 
        cand =sorted (cand ,key =ncp ,reverse =True )
        print (f"  [hard] {len (cand )} candidate, en cok-CP (en zor) parts once",flush =True )
    elif a .diverse :
        cand =diversify (cand )
        print (f"  [diverse] {len (cand )} candidate part cesitlilik sirasina dizildi",flush =True )
    rows =[];n_parts =0 ;n_cp =0 
    for _f_item in cand :
        f =_f_item [1 ]
        head =os .path .basename (f ).split ("_")[0 ]
        mfg ,pid =(head .split (".",1 )+[""])[:2 ]
        if pid not in step or pid in seen :continue 
        if a .mfg and mfg !=a .mfg :continue 
        try :
            j =json .load (open (f ,encoding ="utf-8-sig"))
            G =np .array ([[c ["Point"]["X"],c ["Point"]["Y"],c ["Point"]["Z"]]
            for c in j .get ("ConnectionPoints",[])],float )
            Gd =np .array ([[c ["InsertDirection"]["X"],c ["InsertDirection"]["Y"],c ["InsertDirection"]["Z"]]
            for c in j .get ("ConnectionPoints",[])],float )
            if not len (G ):continue 
            Vj =np .array ([[p ["X"],p ["Y"],p ["Z"]]for p in j ["Graphic3d"]["Points"]],float )
            Vr ,Fr =step_to_mesh (step [pid ])
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
            acc =None 
            for model ,meta ,_ in models :
                _ ,pb =D .predict (model ,meta ,V ,F ,device =a .device ,op_cache_dir =OP ,return_probs =True )
                pb =np .asarray (pb ,float );acc =pb if acc is None else acc +pb 
            probs =acc /len (models );lab =probs .argmax (-1 )
            cps =cp_openings .connection_points (V ,F ,lab ,min_v =10 ,classes =(CE ,CT ),dedupe_mm =10.0 ,
            probs =probs ,vertex_conf =0.5 ,ct_depth_min_mm =1.0 ,
            cluster_mm =5.0 )# CURRENT product post-proc, so
            # "misses" are what the product ACTUALLY misses (min_v 45/vc 0.7 would over-count)
            R ,t ,_ =align_frames (Vr ,Vj )
            Gs =(G -t )@R # manufacturer CPs INTO the mesh frame
            Gds =(Gd /(np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 ))@R 
            P =np .array ([np .asarray (c ["point"])for c in cps ],float )if cps else np .zeros ((0 ,3 ))
            tol =max (3.0 ,0.06 *float (np .linalg .norm (V .max (0 )-V .min (0 ))))
            miss =[]
            for i in range (len (Gs )):
                if len (P ):
                # axis-aware: same opening? perpendicular distance to this CP's own axis
                    diff =P -Gs [i ]
                    along =diff @Gds [i ]
                    perp =np .linalg .norm (diff -along [:,None ]*Gds [i ],axis =1 )
                    if perp .min ()<=tol :
                        continue # found -> not a miss
                miss .append (i )
            if not miss :continue 
            n_parts +=1 
            for i in miss [:a .max_per_part ]:
                rows .append ({"part_id":pid ,"mfg":mfg ,"step":step [pid ],
                "cp":Gs [i ].tolist (),"dir":Gds [i ].tolist (),
                "n_mfg_cps":len (G ),"n_pred":len (P )})
                n_cp +=1 
            if a .limit and n_parts >=a .limit :break 
        except Exception :
            continue 

    os .makedirs ("results",exist_ok =True )
    json .dump ({"n_parts":n_parts ,"n_misses":n_cp ,"ckpts":a .ckpts ,"items":rows },
    open (a .out ,"w"),indent =1 )
    from collections import Counter 
    c =Counter (r ["mfg"]for r in rows )
    print (f"\n{n_parts } parcada {n_cp } KACIRILAN manufacturer CP-si")
    print ("  manufacturer dagilimi: "+", ".join (f"{k }:{v }"for k ,v in c .most_common ()))
    print (f"  -> {a .out }")


if __name__ =="__main__":
    main ()
