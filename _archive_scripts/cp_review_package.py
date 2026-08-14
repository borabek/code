# -*- coding: utf-8 -*-
"""Build a HUMAN-CONFIRMATION package to SEAL the CP benchmark. For each part in a split it
writes (into _cp_confirm/):
  <pid>_review.png : contact sheet (front + iso) with the GT-derived CP balls NUMBERED 1..N.
  <pid>.json       : pre-filled {cps:[{id, point, source_class, confirmed:true}, ...]}.
You review the PNG and edit the JSON: set "confirmed": false to REJECT a wrong CP, or append
{"point":[x,y,z], "confirmed":true, "added":true} for a MISSED opening (read x,y,z off the STEP
in your CAD tool, same frame as the OBJ). Then `cp_seal_score.py` scores the model against your
confirmed points -> a fully-human sealed CP number.

The candidates are the human-REGION-derived CPs (cp_openings ten the human labels) -- your job is
just to confirm/correct the exact points, which is what turns a region-derived GT into a
human-clicked one.

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe cp_review_package.py --split test_locked
"""
import argparse ,os ,json 
import numpy as np 
import matplotlib ;matplotlib .use ("Agg");import matplotlib .pyplot as plt 
from mpl_toolkits .mplot3d .art3d import Poly3DCollection 
import scheffler_dataset as dataset 
import connector3d ,cp_openings 

NAME ={int (connector3d .CONTACT ):"Contact",int (connector3d .CABLE_ENTRY ):"CableEntry"}


def _angles (V ):
    """Face-ten + oblique views down the THIN bbox axis (the face the openings sit ten), so all
    CPs show instead of being edge-ten."""
    thin =int (np .argmin (V .max (0 )-V .min (0 )))
    face ={0 :(0 ,0 ),1 :(0 ,-90 ),2 :(90 ,-90 )}[thin ]
    obl ={0 :(22 ,-38 ),1 :(22 ,-62 ),2 :(58 ,-62 )}[thin ]
    return [("face-ten",*face ),("oblique",*obl )]


def render (V ,F ,cps ,out ):
    fig =plt .figure (figsize =(13 ,6.5 ))
    ctr =V .mean (0 );rng =(V .max (0 )-V .min (0 )).max ()
    for i ,(name ,elev ,azim )in enumerate (_angles (V ),1 ):
        ax =fig .add_subplot (1 ,2 ,i ,projection ="3d")
        # TRANSLUCENT mesh so CP balls BEHIND the slab still show (matplotlib has no z-buffer;
        # translucency sidesteps the occlusion that hid balls in earlier renders).
        pc =Poly3DCollection (V [F ],facecolors =(0.62 ,0.62 ,0.62 ,0.22 ),edgecolors ="none")
        ax .add_collection3d (pc )
        for j ,c in enumerate (cps ,1 ):
            p =c ["point"]
            col ="lime"if int (c ["source_label"])==int (connector3d .CABLE_ENTRY )else "dodgerblue"
            ax .scatter ([p [0 ]],[p [1 ]],[p [2 ]],c =col ,s =300 ,edgecolors ="k",linewidths =1.4 ,
            depthshade =False ,zorder =10 )
            ax .text (p [0 ],p [1 ],p [2 ],f"  {j }",color ="red",fontsize =14 ,fontweight ="bold",zorder =11 )
        ax .set_xlim (ctr [0 ]-rng /2 ,ctr [0 ]+rng /2 );ax .set_ylim (ctr [1 ]-rng /2 ,ctr [1 ]+rng /2 );ax .set_zlim (ctr [2 ]-rng /2 ,ctr [2 ]+rng /2 )
        nce =sum (1 for c in cps if int (c ["source_label"])==int (connector3d .CABLE_ENTRY ))
        ax .view_init (elev =elev ,azim =azim );ax .set_axis_off ()
        ax .set_title (f"{name }  (green CableEntry={nce }, blue Contact={len (cps )-nce })")
    plt .tight_layout ();plt .savefig (out ,dpi =90 ,bbox_inches ="tight");plt .close ()


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--split",default ="test_locked");ap .add_argument ("--out",default ="_cp_confirm")
    ap .add_argument ("--min-v",type =int ,default =20 )
    a =ap .parse_args ()
    os .makedirs (a .out ,exist_ok =True )
    samples =dataset .load_split ("wscad_corpus_scheffler_exact",a .split ,
    allow_locked =(a .split =="test_locked"),verify_hashes =False )
    import hashlib 
    corpus_root =os .path .join ("wscad_corpus_scheffler_exact",a .split )

    def sha (p ):
        return hashlib .sha256 (open (p ,"rb").read ()).hexdigest ()[:16 ]if os .path .exists (p )else None 
    total =0 
    for s in samples :
        pid =s ["part_id"];V =np .asarray (s ["verts"],float );F =np .asarray (s ["faces"],int )
        cps =cp_openings .connection_points (V ,F ,np .asarray (s ["labels"]),min_v =a .min_v ,
        classes =(int (connector3d .CABLE_ENTRY ),int (connector3d .CONTACT )),
        dedupe_mm =0.0 )
        render (V ,F ,cps ,os .path .join (a .out ,f"{pid }_review.png"))
        pdir =os .path .join (corpus_root ,pid )
        # Candidates are MACHINE proposals -> status PENDING. A human sets decision to accept/
        # reject/move/add/flip_direction and fills reviewer + reviewed_utc. cp_seal_score refuses
        # to score until every CP is decided AND the part carries a signed seal. No auto-confirm.
        rec ={"part_id":pid ,"split":a .split ,"cp_def_version":"cp-v2-cableentry-primary",
        "frame":"STEP/OBJ (same frame)","review_status":"PENDING",
        "source_sha":{"obj":sha (os .path .join (pdir ,f"{pid }.obj")),
        "labels":sha (os .path .join (pdir ,f"{pid }.labels.txt")),
        "step":sha (os .path .join (pdir ,f"{pid }.stp"))},
        "instructions":("For each cp set decision to one of accept/reject/move/flip_direction; "
        "for a MISSED opening append {point:[x,y,z], direction:[x,y,z], "
        "decision:'add', added:true}. Fill reviewer + reviewed_utc. Then sign: "
        "set seal.reviewer, seal.reviewed_utc, seal.signature (any non-empty "
        "token you control)."),
        "cps":[{"id":j ,"point":[round (float (x ),2 )for x in c ["point"]],
        "direction":[round (float (x ),3 )for x in c ["direction"]],
        "source_class":NAME .get (c ["source_label"],"?"),
        "decision":None ,"reviewer":None ,"reviewed_utc":None }
        for j ,c in enumerate (cps ,1 )],
        "seal":{"reviewer":None ,"reviewed_utc":None ,"signature":None }}
        json .dump (rec ,open (os .path .join (a .out ,f"{pid }.json"),"w"),indent =1 )
        total +=len (cps )
        print (f"  {pid }: {len (cps )} PENDING candidate CP -> {pid }_review.png + {pid }.json")
    print (f"\n{len (samples )} parts, {total } PENDING candidate CPs -> {a .out }/  (NOT human-confirmed; "
    f"review PNG, set decisions + sign seal, then cp_seal_score.py)")
    print ("DONE")


if __name__ =="__main__":
    main ()
