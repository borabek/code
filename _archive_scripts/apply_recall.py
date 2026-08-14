# -*- coding: utf-8 -*-
"""Turn the recall-painting JSON into partial training labels for the WEI parts the model missed.

WHY: two adjudication rounds proved precision is ~98%; the remaining error is RECALL, worst ten
Weidmueller (0.387). The user painted the 60 manufacturer-located missed openings (27 WEI parts,
6925 vertices) with the recall tool. Each painted vertex set becomes a POSITIVE (CableEntry) region;
everything else ten the part stays 0 and is handled by the masked BCE (only supervises the connection
channel), exactly like _label_targets.

LEAKAGE: these 27 parts live in the WEI arbiter TEST set. Training ten them means they must be excluded
from WEI scoring afterwards -- apply writes their ids to _label_targets_recall/trained_parts.json and
big_arbiter's leakage guard already unions any such file. The remaining ~155 WEI parts stay as an
honest held-out.

Rebuilds each part's 6000-vertex mesh from its STEP (the recall tool packed vertices in that exact
indexing), writes _label_targets_recall/<pid>/{<pid>.obj, .labels.txt}.

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe apply_recall.py \
         --paint "C:/Users/.../recall_paint.json"
"""
import os ,sys ,json ,glob ,argparse 
from collections import defaultdict 
import numpy as np 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import thesis_remesh ,connector3d 
from infer_step_cp import step_to_mesh 

CE =int (connector3d .CABLE_ENTRY )
OUT ="_label_targets_recall"# WEI turu; PXC for --out with ayrilir


def save_obj (path ,V ,F ):
    with open (path ,"w")as f :
        for v in V :f .write (f"v {v [0 ]:.6f} {v [1 ]:.6f} {v [2 ]:.6f}\n")
        for t in F :f .write (f"f {t [0 ]+1 } {t [1 ]+1 } {t [2 ]+1 }\n")


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--paint",required =True )
    ap .add_argument ("--out",default ="_label_targets_recall")
    ap .add_argument ("--mfg",default ="WEI")
    ap .add_argument ("--min-region",type =int ,default =25 ,help ="skip a part if its total painted < this")
    a =ap .parse_args ()
    paint =json .load (open (a .paint ))
    step ={os .path .basename (s ).split ("_")[1 ]:s for s in glob .glob ("all_wscad_stp/*.stp")}

    by_part =defaultdict (list )
    for key ,idx in paint .items ():
        pid =key .split ("__")[0 ]
        by_part [pid ].extend (int (i )for i in idx )

    global OUT ;OUT =a .out ;os .makedirs (OUT ,exist_ok =True )
    done =skip =0 ;trained =[]
    for pid ,verts in sorted (by_part .items ()):
        verts =sorted (set (verts ))
        if len (verts )<a .min_region :
            print (f"  {pid }: {len (verts )} vertex < threshold, atlandi");skip +=1 ;continue 
        if pid not in step :
            print (f"  {pid }: STEP none, atlandi");skip +=1 ;continue 
        Vr ,Fr =step_to_mesh (step [pid ])
        V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
        V =np .ascontiguousarray (V ,float );F =np .ascontiguousarray (F ,np .int64 )
        verts =[v for v in verts if 0 <=v <len (V )]# guard against index drift
        L =np .zeros (len (V ),np .int64 );L [verts ]=CE 
        d =os .path .join (OUT ,pid );os .makedirs (d ,exist_ok =True )
        save_obj (os .path .join (d ,f"{pid }.obj"),V ,F )
        open (os .path .join (d ,f"{pid }.labels.txt"),"w").write ("\n".join (map (str ,L .tolist ())))
        json .dump ({"part_id":pid ,"mfg":a .mfg ,"ce_vertices":int ((L ==CE ).sum ()),
        "source":"recall painting (manufacturer-located, human-shaped)"},
        open (os .path .join (d ,f"{pid }.provenance.json"),"w"),indent =1 )
        trained .append (pid );done +=1 
    json .dump ({"parts":trained ,"note":"WEI arbiter parts trained via recall painting -- EXCLUDE from WEI scoring"},
    open (os .path .join (OUT ,"trained_parts.json"),"w"),indent =1 )
    print (f"\n{done } part etiketlendi, {skip } atlandi -> {OUT }/")
    print (f"  this {done } WEI parcasi artik EGITIMDE -> hakemde WEI test seti {done } part kuculecek")


if __name__ =="__main__":
    main ()
