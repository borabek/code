# -*- coding: utf-8 -*-
"""D7 physical-opening proposal ceiling audit (isolated; no product changes).

The production pose dictionary only offers STEP primitives near an existing
segmentation candidate.  This audit asks a different question: how much does the
one-to-one ceiling rise when exact STEP opening positions are allowed to become
independent candidate *nodes*?

Node sources
------------
* ``base``: G7BIRLESIK segmentation candidates.
* ``cylinder``: both finite endpoints of every exact OCP cylinder with radius in
  [0.5, 4.5] mm.  Both axis signs are pose choices of the same position node.
* ``planar``: centres of inner planar wires from ``brep_aciklik``.  Both normal
  signs are pose choices of the same position node.

Positions within ``DEDUPE_MM`` are single-linkage clustered.  A cluster remains
ONE candidate row, regardless of how many sources/directions it carries, so it can
claim at most one GT in the bipartite matching.  The oracle removes false-positive
rows (FP=0); it is a representation ceiling, not a deployable score.

The public helper functions are intentionally import-safe so a separate audit can
cross these position nodes with local direction dictionaries without duplicating
the source/deduplication contract.
"""
from __future__ import annotations 

import argparse 
import hashlib 
import json 
import os 
import pickle 
from pathlib import Path 
from typing import Iterable ,Mapping ,Sequence 

import numpy as np 
from scipy .sparse import csr_matrix 
from scipy .sparse .csgraph import maximum_bipartite_matching 
from scipy .spatial import cKDTree 

from sina_cluster import f1_rejim ,f1w 


ROOT =Path (__file__ ).resolve ().parent 
D7_MANIFEST =ROOT /"results"/"d7_sinav_kumesi.json"
G7_RECORDS =ROOT /"results"/"_der_yeni_G7BIRLESIK.pkl"
CYLINDER_CACHE =ROOT /"results"/"_d7_silindirler.pkl"
PLANAR_CACHE =ROOT /"results"/"_d7_acikliklar.pkl"
D7_PROB_CACHE =ROOT /"results"/"_p1_olasilik_d7"
DEFAULT_RECEIPT =ROOT /"results"/"audit_d7_physical_node_oracle.json"

RADIUS_MIN_MM =0.5 
RADIUS_MAX_MM =4.5 
DEDUPE_MM =0.5 
DETECTION_AXIAL_MM =40.0 
ROBOT_LATERAL_MM =2.0 
ROBOT_ANGLE_DEG =10.0 
LOCAL_DIRECTION_MM =8.0 


def _unit (v :Sequence [float ])->np .ndarray |None :
    a =np .asarray (v ,dtype =float )
    n =float (np .linalg .norm (a ))
    if a .shape !=(3 ,)or not np .isfinite (a ).all ()or n <=1e-12 :
        return None 
    return a /n 


def _signed_pair (v :Sequence [float ])->list [np .ndarray ]:
    a =_unit (v )
    return []if a is None else [a ,-a ]


def make_node (
point :Sequence [float ],
directions :Iterable [Sequence [float ]],
source :str ,
member_id :str ,
)->dict :
    p =np .asarray (point ,dtype =float )
    dirs =[d for raw in directions if (d :=_unit (raw ))is not None ]
    return {
    "point":p ,
    "directions":dirs ,
    "sources":{str (source )},
    "member_ids":[str (member_id )],
    "cluster_id":None ,
    }


def _dedupe_directions (directions :Iterable [Sequence [float ]])->list [np .ndarray ]:
    out :list [np .ndarray ]=[]
    seen :set [tuple [float ,float ,float ]]=set ()
    for raw in directions :
        d =_unit (raw )
        if d is None :
            continue 
        key =tuple (np .round (d ,8 ).tolist ())
        if key not in seen :
            seen .add (key )
            out .append (d )
    return out 


def cluster_nodes (nodes :Sequence [Mapping [str ,object ]],tol :float =DEDUPE_MM )->list [dict ]:
    """Spatially deduplicate resources while preserving all signed directions.

    The output cluster, not each source face, is the one-to-one candidate row.
    ``member_ids`` therefore exposes exactly which resources were collapsed.
    """
    valid =[]
    for node in nodes :
        p =np .asarray (node ["point"],dtype =float )
        if p .shape ==(3 ,)and np .isfinite (p ).all ():
            valid .append (node )
    if not valid :
        return []

    points =np .asarray ([n ["point"]for n in valid ],dtype =float )
    parent =np .arange (len (valid ),dtype =np .int64 )

    def find (i :int )->int :
        while parent [i ]!=i :
            parent [i ]=parent [parent [i ]]
            i =int (parent [i ])
        return i 

    for i ,j in sorted (cKDTree (points ).query_pairs (float (tol ))):
        a ,b =find (int (i )),find (int (j ))
        if a !=b :
            parent [b ]=a 

    groups :dict [int ,list [int ]]={}
    for i in range (len (valid )):
        groups .setdefault (find (i ),[]).append (i )

    merged =[]
    for indices in groups .values ():
        dirs =_dedupe_directions (
        d for i in indices for d in valid [i ].get ("directions",[])
        )
        merged .append (
        {
        "point":points [indices ].mean (axis =0 ),
        "directions":dirs ,
        "sources":set ().union (
        *(set (valid [i ].get ("sources",set ()))for i in indices )
        ),
        "member_ids":sorted (
        str (x )
        for i in indices 
        for x in valid [i ].get ("member_ids",[])
        ),
        "cluster_id":None ,
        }
        )

    merged .sort (key =lambda n :tuple (np .round (n ["point"],9 ).tolist ()))
    for cluster_id ,node in enumerate (merged ):
        node ["cluster_id"]=int (cluster_id )
    return merged 


def build_source_nodes (
pid :str ,
record :Mapping [str ,object ],
cylinders :Mapping [str ,Sequence [Mapping [str ,object ]]],
planars :Mapping [str ,Sequence [Mapping [str ,object ]]],
)->dict [str ,list [dict ]]:
    """Return unclustered nodes by source; callers choose the union/dedupe branch."""
    base =[]
    for i ,(p ,d )in enumerate (zip (np .asarray (record ["P"]),np .asarray (record ["Pd"]))):
        base .append (make_node (p ,[d ],"base",f"base:{i }"))

    cylinder =[]
    for i ,item in enumerate (cylinders .get (pid ,())):
        try :
            radius =float (item ["radius"])
        except (KeyError ,TypeError ,ValueError ):
            continue 
        if not np .isfinite (radius )or not (RADIUS_MIN_MM <=radius <=RADIUS_MAX_MM ):
            continue 
        dirs =_signed_pair (item .get ("axis",()))
        if not dirs :
            continue 
        for side in ("mouth_a","mouth_b"):
            if side in item :
                cylinder .append (
                make_node (item [side ],dirs ,"cylinder",f"cylinder:{i }:{side }")
                )

    planar =[]
    for i ,item in enumerate (planars .get (pid ,())):
        if "center"not in item :
            continue 
        dirs =_signed_pair (item .get ("normal",()))
        if dirs :
            planar .append (make_node (item ["center"],dirs ,"planar",f"planar:{i }"))

    return {"base":base ,"cylinder":cylinder ,"planar":planar }


def load_inputs ()->tuple [dict [str ,dict ],dict ,dict ,dict ]:
    with D7_MANIFEST .open (encoding ="utf-8")as handle :
        manifest =json .load (handle )
    wanted =set (map (str ,manifest ["pidler"]))
    with G7_RECORDS .open ("rb")as handle :
        records ={r ["pid"]:r for r in pickle .load (handle )if r ["pid"]in wanted }
    with CYLINDER_CACHE .open ("rb")as handle :
        cylinders =pickle .load (handle )
    with PLANAR_CACHE .open ("rb")as handle :
        planars =pickle .load (handle )
    return records ,cylinders ,planars ,manifest 


def _acceptance (
nodes :Sequence [Mapping [str ,object ]],
gt_points :np .ndarray ,
gt_directions :np .ndarray ,
diag :float ,
robot :bool ,
)->np .ndarray :
    if not nodes or not len (gt_points ):
        return np .zeros ((len (nodes ),len (gt_points )),dtype =bool )
    points =np .asarray ([n ["point"]for n in nodes ],dtype =float )
    diff =points [:,None ,:]-gt_points [None ,:,:]
    axial =(diff *gt_directions [None ,:,:]).sum (axis =-1 )
    lateral =np .linalg .norm (
    diff -axial [...,None ]*gt_directions [None ,:,:],axis =-1 
    )
    lateral_limit =ROBOT_LATERAL_MM if robot else max (3.0 ,0.06 *float (diag ))
    accepted =(np .abs (axial )<=DETECTION_AXIAL_MM )&(lateral <=lateral_limit )
    if robot :
        cos_min =float (np .cos (np .deg2rad (ROBOT_ANGLE_DEG )))
        for i ,node in enumerate (nodes ):
            directions =np .asarray (node .get ("directions",[]),dtype =float ).reshape (-1 ,3 )
            if not len (directions ):
                accepted [i ,:]=False 
                continue 
                # Signed metric.  Physical nodes explicitly contain +axis and -axis;
                # a base node contains only its actual predicted signed direction.
            accepted [i ,:]&=np .max (directions @gt_directions .T ,axis =0 )>=cos_min 
    return accepted 


def match_nodes (
nodes :Sequence [Mapping [str ,object ]],
gt_points :Sequence [Sequence [float ]],
gt_directions :Sequence [Sequence [float ]],
diag :float ,
*,
robot :bool =False ,
)->int :
    """Maximum-cardinality one-to-one TP count for candidate-resource nodes."""
    gt_points =np .asarray (gt_points ,dtype =float ).reshape (-1 ,3 )
    gt_directions =np .asarray (gt_directions ,dtype =float ).reshape (-1 ,3 )
    accepted =_acceptance (nodes ,gt_points ,gt_directions ,diag ,robot )
    if not accepted .any ():
        return 0 
    matched =maximum_bipartite_matching (csr_matrix (accepted ),perm_type ="column")
    return int ((matched >=0 ).sum ())


def obb_axes (vertices :Sequence [Sequence [float ]])->list [np .ndarray ]:
    """PCA approximation used by the project's existing OBB direction dictionary."""
    points =np .asarray (vertices ,dtype =float ).reshape (-1 ,3 )
    if len (points )<3 :
        return []
    try :
        _ ,_ ,axes =np .linalg .svd (points -points .mean (axis =0 ),full_matrices =False )
    except np .linalg .LinAlgError :
        return []
    return _dedupe_directions (axes [:3 ])


def _unsigned_axis_arrays (nodes :Sequence [Mapping [str ,object ]])->tuple [np .ndarray ,np .ndarray ]:
    """Direction-factor resources as (position, unsigned axis) rows.

    Every current/cylinder/planar direction is factorised into +/- choices.  Keeping
    one unsigned representative avoids materialising both signs or every Cartesian
    (position, direction) pose; the signed feasibility test uses ``abs(dot)``.
    """
    positions ,axes =[],[]
    for node in nodes :
        for raw in node .get ("directions",[]):
            axis =_unit (raw )
            if axis is None :
                continue 
                # Canonical sign only reduces duplicate work; abs(dot) restores +/-.
            first =next ((x for x in axis if abs (float (x ))>1e-10 ),1.0 )
            if first <0 :
                axis =-axis 
            positions .append (np .asarray (node ["point"],dtype =float ))
            axes .append (axis )
    if not axes :
        return np .zeros ((0 ,3 )),np .zeros ((0 ,3 ))
        # Same node often carries +a and -a, and coincident STEP faces repeat axes.
    seen =set ()
    keep =[]
    for i ,(point ,axis )in enumerate (zip (positions ,axes )):
        key =tuple (np .round (np .r_ [point ,axis ],7 ).tolist ())
        if key not in seen :
            seen .add (key )
            keep .append (i )
    return np .asarray (positions )[keep ],np .asarray (axes )[keep ]


def match_nodes_direction_factor (
nodes :Sequence [Mapping [str ,object ]],
gt_points :Sequence [Sequence [float ]],
gt_directions :Sequence [Sequence [float ]],
diag :float ,
*,
mode :str ,
local_mm :float =LOCAL_DIRECTION_MM ,
part_obb_axes :Sequence [Sequence [float ]]=(),
)->int :
    """Strict robot oracle with lazily factorised position x direction choices.

    ``mode='local'`` lets a position node use +/- current/cylinder/planar axes whose
    source node is within ``local_mm``.  ``mode='partwide'`` uses every such axis in
    the part.  ``mode='obb'`` uses only +/- PCA/OBB axes.  Appending ``'+obb'`` adds
    those global axes to local/part-wide feasibility.

    Crucially, the Cartesian choices do not create extra candidate rows: matching
    still runs over the original physical position clusters, one row per resource.
    """
    gt_points =np .asarray (gt_points ,dtype =float ).reshape (-1 ,3 )
    gt_directions =np .asarray (gt_directions ,dtype =float ).reshape (-1 ,3 )
    if not nodes or not len (gt_points ):
        return 0 
        # First apply the strict robot position box, deliberately without an angle.
    accepted =_acceptance (nodes ,gt_points ,gt_directions ,diag ,robot =False )
    points =np .asarray ([node ["point"]for node in nodes ],dtype =float )
    diff =points [:,None ,:]-gt_points [None ,:,:]
    axial =(diff *gt_directions [None ,:,:]).sum (axis =-1 )
    lateral =np .linalg .norm (
    diff -axial [...,None ]*gt_directions [None ,:,:],axis =-1 
    )
    accepted &=(np .abs (axial )<=DETECTION_AXIAL_MM )&(lateral <=ROBOT_LATERAL_MM )

    source_points ,source_axes =_unsigned_axis_arrays (nodes )
    cos_min =float (np .cos (np .deg2rad (ROBOT_ANGLE_DEG )))
    direction_ok =np .zeros ((len (nodes ),len (gt_points )),dtype =bool )
    base_mode =mode .replace ("+obb","")

    if base_mode in ("local","partwide")and len (source_axes ):
        aligned =np .abs (source_axes @gt_directions .T )>=cos_min 
        if base_mode =="partwide":
            direction_ok [:]=aligned .any (axis =0 )[None ,:]
        else :
        # Lazy local Cartesian feasibility: for each GT direction, find the
        # nearest source carrying an aligned axis.  No pose product is built.
            for gi in range (len (gt_points )):
                usable =source_points [aligned [:,gi ]]
                if len (usable ):
                    distance ,_ =cKDTree (usable ).query (points ,k =1 )
                    direction_ok [:,gi ]=distance <=float (local_mm )

    if base_mode =="obb"or "+obb"in mode :
        axes =np .asarray (
        [a for raw in part_obb_axes if (a :=_unit (raw ))is not None ],dtype =float 
        ).reshape (-1 ,3 )
        if len (axes ):
            obb_ok =(np .abs (axes @gt_directions .T )>=cos_min ).any (axis =0 )
            direction_ok |=obb_ok [None ,:]

    accepted &=direction_ok 
    if not accepted .any ():
        return 0 
    matched =maximum_bipartite_matching (csr_matrix (accepted ),perm_type ="column")
    return int ((matched >=0 ).sum ())


def _file_sha256 (path :Path )->str :
    h =hashlib .sha256 ()
    with path .open ("rb")as handle :
        for block in iter (lambda :handle .read (1024 *1024 ),b""):
            h .update (block )
    return h .hexdigest ()


def _summary (rows :list [tuple ])->dict :
    detail =f1_rejim (rows )
    return {
    "f1":float (f1w (rows )),
    "low_cp_f1":detail ["F1"]["dusuk"],
    "high_cp_f1":detail ["F1"]["very"],
    "low_cp_parts":int (detail ["n"]["dusuk"]),
    "high_cp_parts":int (detail ["n"]["very"]),
    }


def run_audit (dedupe_mm :float =DEDUPE_MM )->dict :
    records ,cylinders ,planars ,manifest =load_inputs ()
    branch_names =(
    "base",
    "cylinder",
    "planar",
    "physical",
    "base+cylinder",
    "base+planar",
    "base+physical",
    )
    detection ={name :[]for name in branch_names }
    robot ={name :[]for name in branch_names }
    per_mfg_detection ={name :{}for name in branch_names }
    per_mfg_robot ={name :{}for name in branch_names }
    clustered_counts ={name :0 for name in branch_names }
    raw_counts ={"base":0 ,"cylinder":0 ,"planar":0 }
    factor_names =(
    "own_signed",
    "factor_local_8mm",
    "factor_partwide",
    "factor_obb_only",
    "factor_local_8mm+obb",
    "factor_partwide+obb",
    )
    factor_rows ={name :[]for name in factor_names }
    factor_per_mfg ={name :{}for name in factor_names }
    factor_axis_rows =[]
    mesh_cache_found =0 

    for pid in sorted (records ):
        record =records [pid ]
        source =build_source_nodes (pid ,record ,cylinders ,planars )
        for name in raw_counts :
            raw_counts [name ]+=len (source [name ])
        branches ={
        "base":cluster_nodes (source ["base"],dedupe_mm ),
        "cylinder":cluster_nodes (source ["cylinder"],dedupe_mm ),
        "planar":cluster_nodes (source ["planar"],dedupe_mm ),
        "physical":cluster_nodes (source ["cylinder"]+source ["planar"],dedupe_mm ),
        "base+cylinder":cluster_nodes (source ["base"]+source ["cylinder"],dedupe_mm ),
        "base+planar":cluster_nodes (source ["base"]+source ["planar"],dedupe_mm ),
        "base+physical":cluster_nodes (
        source ["base"]+source ["cylinder"]+source ["planar"],dedupe_mm 
        ),
        }
        gt_points =np .asarray (record ["G"],dtype =float )
        gt_directions =np .asarray (record ["Gd"],dtype =float )
        regime ="very"if len (gt_points )>=8 else "dusuk"
        mfg =str (record ["mfg"])
        for name ,nodes in branches .items ():
            clustered_counts [name ]+=len (nodes )
            det_tp =match_nodes (
            nodes ,gt_points ,gt_directions ,float (record ["diag"]),robot =False 
            )
            rob_tp =match_nodes (
            nodes ,gt_points ,gt_directions ,float (record ["diag"]),robot =True 
            )
            det_row =(regime ,det_tp ,0 ,len (gt_points )-det_tp )
            rob_row =(regime ,rob_tp ,0 ,len (gt_points )-rob_tp )
            detection [name ].append (det_row )
            robot [name ].append (rob_row )
            per_mfg_detection [name ].setdefault (mfg ,[]).append (det_row )
            per_mfg_robot [name ].setdefault (mfg ,[]).append (rob_row )

            # Position ceiling is fixed to the strongest branch.  The following
            # ablation changes only which signed direction factors each candidate row
            # may choose; candidate/resource count and one-to-one assignment stay fixed.
        union_nodes =branches ["base+physical"]
        mesh_file =D7_PROB_CACHE /f"{pid }.npz"
        axes =[]
        if mesh_file .exists ():
            try :
                with np .load (mesh_file )as mesh_data :
                    axes =obb_axes (mesh_data ["V"])
                mesh_cache_found +=1 
            except (OSError ,ValueError ,KeyError ):
                axes =[]
        _source_points ,source_axes =_unsigned_axis_arrays (union_nodes )
        factor_axis_rows .append ((len (source_axes ),len (axes )))
        factor_tp ={
        "own_signed":match_nodes (
        union_nodes ,gt_points ,gt_directions ,float (record ["diag"]),robot =True 
        ),
        "factor_local_8mm":match_nodes_direction_factor (
        union_nodes ,gt_points ,gt_directions ,float (record ["diag"]),
        mode ="local",local_mm =LOCAL_DIRECTION_MM ,
        ),
        "factor_partwide":match_nodes_direction_factor (
        union_nodes ,gt_points ,gt_directions ,float (record ["diag"]),
        mode ="partwide",
        ),
        "factor_obb_only":match_nodes_direction_factor (
        union_nodes ,gt_points ,gt_directions ,float (record ["diag"]),
        mode ="obb",part_obb_axes =axes ,
        ),
        "factor_local_8mm+obb":match_nodes_direction_factor (
        union_nodes ,gt_points ,gt_directions ,float (record ["diag"]),
        mode ="local+obb",local_mm =LOCAL_DIRECTION_MM ,part_obb_axes =axes ,
        ),
        "factor_partwide+obb":match_nodes_direction_factor (
        union_nodes ,gt_points ,gt_directions ,float (record ["diag"]),
        mode ="partwide+obb",part_obb_axes =axes ,
        ),
        }
        for name ,tp in factor_tp .items ():
            row =(regime ,tp ,0 ,len (gt_points )-tp )
            factor_rows [name ].append (row )
            factor_per_mfg [name ].setdefault (mfg ,[]).append (row )

    table ={}
    for name in branch_names :
        table [name ]={
        "raw_nodes":(
        raw_counts ["base"]if name =="base"else 
        raw_counts ["cylinder"]if name =="cylinder"else 
        raw_counts ["planar"]if name =="planar"else None 
        ),
        "deduped_candidate_rows":int (clustered_counts [name ]),
        "detection_oracle":_summary (detection [name ]),
        "strict_signed_robot_oracle":_summary (robot [name ]),
        "per_manufacturer":{
        mfg :{
        "detection_f1":float (f1w (per_mfg_detection [name ][mfg ])),
        "strict_signed_robot_f1":float (f1w (per_mfg_robot [name ][mfg ])),
        "parts":len (per_mfg_detection [name ][mfg ]),
        }
        for mfg in sorted (per_mfg_detection [name ])
        },
        }

    direction_table ={}
    for name in factor_names :
        direction_table [name ]={
        "candidate_rows":int (clustered_counts ["base+physical"]),
        "strict_signed_robot_oracle":_summary (factor_rows [name ]),
        "per_manufacturer":{
        mfg :{
        "strict_signed_robot_f1":float (f1w (factor_per_mfg [name ][mfg ])),
        "parts":len (factor_per_mfg [name ][mfg ]),
        }
        for mfg in sorted (factor_per_mfg [name ])
        },
        }

    paths =(D7_MANIFEST ,G7_RECORDS ,CYLINDER_CACHE ,PLANAR_CACHE ,Path (__file__ ))
    return {
    "audit":"D7 independent exact-STEP physical candidate-node oracle",
    "n_manifest_parts":len (set (map (str ,manifest ["pidler"]))),
    "n_scored_parts":len (records ),
    "missing_record_parts":sorted (set (map (str ,manifest ["pidler"]))-set (records )),
    "contract":{
    "cylinder_position":"OCP exact finite cylinder mouth_a/mouth_b",
    "cylinder_radius_mm":[RADIUS_MIN_MM ,RADIUS_MAX_MM ],
    "planar_position":"brep_aciklik inner planar-wire centroid",
    "spatial_dedupe_mm":float (dedupe_mm ),
    "resource_rule":"one deduped spatial cluster = one candidate row = at most one GT",
    "oracle_rule":"maximum-cardinality bipartite match; FP forced to zero",
    "detection":"lateral <= max(3mm, 0.06*diag), |axial| <= 40mm, angle free",
    "robot":"lateral <= 2mm, |axial| <= 40mm, signed angle <= 10deg",
    "directions":"base=current signed direction; physical=explicit +/- own axis/normal",
    "direction_factor":(
    "same base+physical position rows; +/- current/cylinder/planar axes; "
    "local uses nearest aligned source within 8mm; partwide uses any part axis; "
    "OBB=PCA axes; feasibility evaluated lazily without pose-row expansion"
    ),
    },
    "raw_source_nodes":{k :int (v )for k ,v in raw_counts .items ()},
    "branches":table ,
    "direction_factor_branches":direction_table ,
    "direction_factor_stats":{
    "mesh_cache_parts":int (mesh_cache_found ),
    "local_mm":LOCAL_DIRECTION_MM ,
    "unsigned_source_axis_rows_median":float (np .median ([x [0 ]for x in factor_axis_rows ])),
    "unsigned_source_axis_rows_p95":float (np .percentile ([x [0 ]for x in factor_axis_rows ],95 )),
    "obb_axes_median":float (np .median ([x [1 ]for x in factor_axis_rows ])),
    },
    "sha256":{str (path .relative_to (ROOT )):_file_sha256 (path )for path in paths },
    }


def main ()->None :
    parser =argparse .ArgumentParser ()
    parser .add_argument ("--dedupe-mm",type =float ,default =DEDUPE_MM )
    parser .add_argument ("--out",default =str (DEFAULT_RECEIPT ))
    args =parser .parse_args ()
    receipt =run_audit (args .dedupe_mm )
    output =Path (args .out )
    output .parent .mkdir (parents =True ,exist_ok =True )
    with output .open ("w",encoding ="utf-8")as handle :
        json .dump (receipt ,handle ,indent =1 ,ensure_ascii =False )
    print (f"scored {receipt ['n_scored_parts']} D7 parts -> {output }")
    for name ,result in receipt ["branches"].items ():
        det =result ["detection_oracle"]
        rob =result ["strict_signed_robot_oracle"]
        print (
        f"{name :<14} rows={result ['deduped_candidate_rows']:>7}  "
        f"det={det ['f1']:.6f} (high={det ['high_cp_f1']:.6f})  "
        f"robot={rob ['f1']:.6f} (high={rob ['high_cp_f1']:.6f})"
        )
    print ("direction factors (base+physical positions; row count unchanged)")
    for name ,result in receipt ["direction_factor_branches"].items ():
        rob =result ["strict_signed_robot_oracle"]
        print (f"{name :<27} robot={rob ['f1']:.6f} (high={rob ['high_cp_f1']:.6f})")


if __name__ =="__main__":
    main ()
