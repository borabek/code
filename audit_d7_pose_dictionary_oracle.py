# -*- coding: utf-8 -*-
"""Clean D7 one-to-one pose-dictionary oracle with branch ablations.

This is an audit tool, not a production selector.  For every part and branch it builds
an ``existing candidate -> GT`` feasibility graph under the frozen robot metric, then
finds a maximum-cardinality one-to-one matching.  Two ceilings are reported:

* ``oracle_null``: unmatched candidates may be suppressed (ideal final selector);
* ``keep_all``: all input candidates survive (isolates pose selection from precision).

The metric matches ``results/metrik_dondurulmus.json`` / ``sina_cluster.match_hungarian``:
lateral distance is measured around the GT axis, signed angle <= 10 degrees, absolute
axial offset <= 40 mm.  It intentionally does *not* reuse p5-v2's label helper, whose
lateral/axial decomposition currently uses the predicted direction.

Inputs are frozen D7 derived records plus read-only STEP dictionaries.  No production
file is modified.
"""
from __future__ import annotations 

import argparse 
import collections 
import hashlib 
import json 
import os 
import pickle 
from pathlib import Path 
from typing import Iterable ,Sequence 

import numpy as np 
from scipy .optimize import linear_sum_assignment 

import audit_d7_pose_options as PO 
import d6_record 


DEFAULT_BRANCHES =tuple (PO .BRANCH_BUILDERS )
YANAL_MM =2.0 
ANGLE_DEG =10.0 
AXIAL_MM =40.0 


def _sha16 (path :str |Path )->str |None :
    path =Path (path )
    if not path .exists ():
        return None 
    h =hashlib .sha256 ()
    with path .open ("rb")as stream :
        for block in iter (lambda :stream .read (1024 *1024 ),b""):
            h .update (block )
    return h .hexdigest ()[:16 ]


def _unit_rows (values :Sequence [Sequence [float ]])->np .ndarray :
    a =np .asarray (values ,dtype =float )
    if not len (a ):
        return np .zeros ((0 ,3 ),dtype =float )
    n =np .linalg .norm (a ,axis =1 ,keepdims =True )
    return a /np .maximum (n ,1e-12 )


def feasibility (
candidates :Sequence [Sequence [PO .PoseOption ]],
gt_points :Sequence [Sequence [float ]],
gt_directions :Sequence [Sequence [float ]],
)->tuple [np .ndarray ,np .ndarray ,np .ndarray ]:
    """Return candidate/GT feasibility, best option index, and best tie-break cost."""
    G =np .asarray (gt_points ,dtype =float )
    Gd =_unit_rows (gt_directions )
    edge =np .zeros ((len (candidates ),len (G )),dtype =bool )
    best_index =np .full ((len (candidates ),len (G )),-1 ,dtype =int )
    best_cost =np .full ((len (candidates ),len (G )),np .inf ,dtype =float )
    for i ,options in enumerate (candidates ):
        if not options or not len (G ):
            continue 
        P =np .asarray ([option .point for option in options ],dtype =float )
        D =_unit_rows ([option .direction for option in options ])
        diff =P [:,None ,:]-G [None ,:,:]
        axial =np .sum (diff *Gd [None ,:,:],axis =2 )
        lateral =np .linalg .norm (
        diff -axial [:,:,None ]*Gd [None ,:,:],axis =2 
        )
        angle =np .degrees (
        np .arccos (np .clip (D @Gd .T ,-1.0 ,1.0 ))
        )
        accepted =(
        (lateral <=YANAL_MM )
        &(np .abs (axial )<=AXIAL_MM )
        &(angle <=ANGLE_DEG )
        )
        # Only breaks ties between valid options; cardinality remains the objective.
        quality =lateral /YANAL_MM +angle /ANGLE_DEG +np .abs (axial )/AXIAL_MM 
        quality =np .where (accepted ,quality ,np .inf )
        idx =np .argmin (quality ,axis =0 )
        values =quality [idx ,np .arange (len (G ))]
        ok =np .isfinite (values )
        edge [i ,ok ]=True 
        best_index [i ,ok ]=idx [ok ]
        best_cost [i ,ok ]=values [ok ]
    return edge ,best_index ,best_cost 


def maximum_one_to_one (
candidates :Sequence [Sequence [PO .PoseOption ]],
gt_points :Sequence [Sequence [float ]],
gt_directions :Sequence [Sequence [float ]],
)->tuple [int ,list [tuple [int ,int ,int ]],collections .Counter [str ]]:
    """Maximum-cardinality candidate/GT match under a branch's pose dictionary."""
    edge ,best_index ,best_cost =feasibility (candidates ,gt_points ,gt_directions )
    if not edge .size :
        return 0 ,[],collections .Counter ()
        # Every full assignment has min(n_candidate, n_gt) entries.  Minimizing invalid
        # (large) edges therefore maximizes the number of valid one-to-one edges.  The tiny
        # quality term only selects a deterministic witness among equal-cardinality optima.
    cost =np .where (edge ,best_cost *1e-6 ,1.0 )
    rows ,cols =linear_sum_assignment (cost )
    matches :list [tuple [int ,int ,int ]]=[]
    kinds :collections .Counter [str ]=collections .Counter ()
    for i ,j in zip (rows .tolist (),cols .tolist ()):
        if not edge [i ,j ]:
            continue 
        option_index =int (best_index [i ,j ])
        matches .append ((i ,j ,option_index ))
        kinds [candidates [i ][option_index ].kind ]+=1 
    return len (matches ),matches ,kinds 


def lazy_cartesian_options (
catalogs :Sequence [PO .PoseCatalog ],
gt_points :Sequence [Sequence [float ]],
gt_directions :Sequence [Sequence [float ]],
)->tuple [list [tuple [PO .PoseOption ,...]],int ]:
    """Materialise only GT-relevant witnesses from the Cartesian pose product.

    Existence factorises exactly: a Cartesian pose passes a GT iff at least one
    position component passes its lateral/axial bounds and at least one direction
    component passes its angle bound.  Keeping one best pair per GT therefore preserves
    every candidate/GT edge while avoiding the full positions x directions expansion.
    """
    G =np .asarray (gt_points ,dtype =float )
    Gd =_unit_rows (gt_directions )
    out :list [tuple [PO .PoseOption ,...]]=[]
    theoretical_count =0 
    for catalog in catalogs :
        positions ,directions =PO .cartesian_components (catalog )
        theoretical_count +=len (positions )*len (directions )
        if not positions or not directions or not len (G ):
            out .append (())
            continue 
        P =np .asarray ([item [0 ]for item in positions ],dtype =float )
        D =_unit_rows ([item [0 ]for item in directions ])
        diff =P [:,None ,:]-G [None ,:,:]
        axial =np .sum (diff *Gd [None ,:,:],axis =2 )
        lateral =np .linalg .norm (
        diff -axial [:,:,None ]*Gd [None ,:,:],axis =2 
        )
        pos_quality =lateral /YANAL_MM +np .abs (axial )/AXIAL_MM 
        pos_quality =np .where (
        (lateral <=YANAL_MM )&(np .abs (axial )<=AXIAL_MM ),
        pos_quality ,
        np .inf ,
        )
        angle =np .degrees (np .arccos (np .clip (D @Gd .T ,-1.0 ,1.0 )))
        dir_quality =np .where (angle <=ANGLE_DEG ,angle /ANGLE_DEG ,np .inf )
        pi =np .argmin (pos_quality ,axis =0 )
        di =np .argmin (dir_quality ,axis =0 )
        witnesses :list [PO .PoseOption ]=[]
        for j in range (len (G )):
            if not np .isfinite (pos_quality [pi [j ],j ])or not np .isfinite (dir_quality [di [j ],j ]):
                continue 
            point ,point_kind ,point_resource =positions [int (pi [j ])]
            direction ,direction_kind ,direction_resource =directions [int (di [j ])]
            witnesses .append (
            PO .PoseOption (
            point ,
            direction ,
            f"cartesian:{point_kind }x{direction_kind }",
            f"cart:{point_resource }|{direction_resource }",
            )
            )
            # Different GTs can choose the same witness; branch_options' numerical dedupe is
            # not needed here because maximum_one_to_one handles repeated poses correctly.
        out .append (tuple (witnesses ))
    return out ,theoretical_count 


def _aggregate (rows :Iterable [tuple [str ,int ,int ,int ]])->dict [str ,object ]:
    rows =list (rows )
    counts =collections .defaultdict (lambda :[0 ,0 ,0 ,0 ])
    for regime ,tp ,fp ,fn in rows :
        counts [regime ][0 ]+=int (tp )
        counts [regime ][1 ]+=int (fp )
        counts [regime ][2 ]+=int (fn )
        counts [regime ][3 ]+=1 

    per_regime :dict [str ,object ]={}
    weighted_f1 =0.0 
    available_weight =0.0 
    weights ={"dusuk":0.895 ,"very":0.105 }
    for regime in ("dusuk","very"):
        tp ,fp ,fn ,n =counts [regime ]
        if not n :
            continue 
        precision =tp /max (tp +fp ,1 )
        recall =tp /max (tp +fn ,1 )
        f1 =2.0 *precision *recall /max (precision +recall ,1e-12 )
        per_regime [regime ]={
        "n_part":n ,
        "tp":tp ,
        "fp":fp ,
        "fn":fn ,
        "precision":precision ,
        "recall":recall ,
        "f1":f1 ,
        }
        weighted_f1 +=weights [regime ]*f1 
        available_weight +=weights [regime ]

    tp =sum (v [0 ]for v in counts .values ())
    fp =sum (v [1 ]for v in counts .values ())
    fn =sum (v [2 ]for v in counts .values ())
    precision =tp /max (tp +fp ,1 )
    recall =tp /max (tp +fn ,1 )
    return {
    "n_part":len (rows ),
    "tp":tp ,
    "fp":fp ,
    "fn":fn ,
    "raw_precision":precision ,
    "raw_recall":recall ,
    "raw_f1":2.0 *precision *recall /max (precision +recall ,1e-12 ),
    "weighted_f1":weighted_f1 /max (available_weight ,1e-12 ),
    "regime":per_regime ,
    }


def _gate_mask (record :dict [str ,object ],gate :object )->np .ndarray :
    import wire_gate 
    from p1c_threshold import maske 

    X =d6_record .x58 (record )
    n =len (record .get ("P",()))
    if X is None or len (X )!=n :
        return np .zeros (n ,dtype =bool )
    expected =int (gate .get ("n_feat",X .shape [1 ]*2 ))if isinstance (gate ,dict )else X .shape [1 ]*2 
    if X .shape [1 ]*2 !=expected :
        return np .zeros (n ,dtype =bool )
    scores =np .asarray (wire_gate .decision_score (gate ,X ),dtype =float )
    return np .asarray (maske (scores ,0.40 ,0.30 ),dtype =bool )


def _load_records (manifest_path :str ,records_path :str )->tuple [dict ,dict [str ,dict ]]:
    with open (manifest_path ,encoding ="utf-8")as stream :
        manifest =json .load (stream )
    wanted =set (map (str ,manifest ["pidler"]))
    with open (records_path ,"rb")as stream :
        all_records =pickle .load (stream )
    records ={
    str (record ["pid"]):record 
    for record in all_records 
    if str (record .get ("pid"))in wanted 
    }
    return manifest ,records 


def run (args :argparse .Namespace )->dict [str ,object ]:
    manifest ,records =_load_records (args .manifest ,args .records )
    with open (args .cylinders ,"rb")as stream :
        cylinders =pickle .load (stream )
    with open (args .planars ,"rb")as stream :
        planars =pickle .load (stream )
    with open (args .gate ,"rb")as stream :
        gate =pickle .load (stream )

    pids =[str (pid )for pid in manifest ["pidler"]if str (pid )in records ]
    if args .limit :
        pids =pids [:args .limit ]
    missing =sorted (set (map (str ,manifest ["pidler"]))-set (records ))
    branches =args .branches 
    pools =("raw","gate_before")
    rows :dict [str ,dict [str ,dict [str ,list [tuple [str ,int ,int ,int ]]]]]={
    pool :{
    branch :{"oracle_null":[],"keep_all":[]}for branch in branches 
    }
    for pool in pools 
    }
    manufacturer_rows :dict [
    str ,dict [str ,dict [str ,dict [str ,list [tuple [str ,int ,int ,int ]]]]]
    ]={
    pool :{
    branch :collections .defaultdict (
    lambda :{"oracle_null":[],"keep_all":[]}
    )
    for branch in branches 
    }
    for pool in pools 
    }
    chosen :dict [str ,dict [str ,collections .Counter [str ]]]={
    pool :{branch :collections .Counter ()for branch in branches }for pool in pools 
    }
    option_counts :dict [str ,dict [str ,int ]]={
    pool :collections .Counter ()for pool in pools 
    }

    for index ,pid in enumerate (pids ,1 ):
        record =records [pid ]
        G =np .asarray (record .get ("G",()),dtype =float )
        Gd =np .asarray (record .get ("Gd",()),dtype =float )
        P_all =np .asarray (record .get ("P",()),dtype =float )
        D_all =np .asarray (record .get ("Pd",()),dtype =float )
        if not len (G ):
            continue 
        gate_keep =_gate_mask (record ,gate )
        for pool ,keep in (
        ("raw",np .ones (len (P_all ),dtype =bool )),
        ("gate_before",gate_keep ),
        ):
            P ,D =P_all [keep ],D_all [keep ]
            catalogs =[
            PO .build_catalog (
            P [i ],
            D [i ],
            cylinders .get (pid ),
            planars .get (pid ),
            mm_max =args .mm_max ,
            radius_min =args .radius_min ,
            radius_max =args .radius_max ,
            )
            for i in range (len (P ))
            ]
            regime ="very"if int (record .get ("n",len (G )))>=8 else "dusuk"
            manufacturer =str (record .get ("mfg","UNKNOWN"))
            for branch in branches :
                if branch =="cartesian_all":
                    candidates ,theoretical_count =lazy_cartesian_options (catalogs ,G ,Gd )
                    option_counts [pool ][branch ]+=theoretical_count 
                else :
                    candidates =[PO .branch_options (branch ,catalog )for catalog in catalogs ]
                    option_counts [pool ][branch ]+=sum (map (len ,candidates ))
                tp ,_matches ,kind_counts =maximum_one_to_one (candidates ,G ,Gd )
                chosen [pool ][branch ].update (kind_counts )
                null_row =(regime ,tp ,0 ,len (G )-tp )
                keep_row =(regime ,tp ,len (P )-tp ,len (G )-tp )
                rows [pool ][branch ]["oracle_null"].append (null_row )
                rows [pool ][branch ]["keep_all"].append (keep_row )
                manufacturer_rows [pool ][branch ][manufacturer ]["oracle_null"].append (null_row )
                manufacturer_rows [pool ][branch ][manufacturer ]["keep_all"].append (keep_row )
        if index %100 ==0 :
            print (f"  {index }/{len (pids )}",flush =True )

    result :dict [str ,object ]={}
    for pool in pools :
        result [pool ]={}
        for branch in branches :
            branch_result ={
            mode :_aggregate (rows [pool ][branch ][mode ])
            for mode in ("oracle_null","keep_all")
            }
            branch_result ["chosen_witness_kind"]=dict (chosen [pool ][branch ])
            branch_result ["n_options_total"]=int (option_counts [pool ][branch ])
            branch_result ["manufacturer"]={
            manufacturer :{
            mode :_aggregate (mode_rows )
            for mode ,mode_rows in grouped .items ()
            }
            for manufacturer ,grouped in sorted (
            manufacturer_rows [pool ][branch ].items ()
            )
            }
            result [pool ][branch ]=branch_result 

    receipt ={
    "audit_only":True ,
    "dataset":"D7 (spent DEV; not final)",
    "metric":{
    "lateral_axis":"GT direction",
    "lateral_mm":YANAL_MM ,
    "signed_angle_deg":ANGLE_DEG ,
    "absolute_axial_mm":AXIAL_MM ,
    "matching":"maximum-cardinality one-to-one candidate<->GT (Hungarian)",
    "oracle_null":"unmatched candidates may be suppressed",
    "keep_all":"all input candidates survive",
    },
    "inputs":{
    "manifest":args .manifest ,
    "manifest_sha16":_sha16 (args .manifest ),
    "records":args .records ,
    "records_sha16":_sha16 (args .records ),
    "cylinders":args .cylinders ,
    "cylinders_sha16":_sha16 (args .cylinders ),
    "planars":args .planars ,
    "planars_sha16":_sha16 (args .planars ),
    "gate":args .gate ,
    "gate_sha16":_sha16 (args .gate ),
    },
    "coverage":{
    "manifest_parts":len (manifest ["pidler"]),
    "record_parts":len (records ),
    "evaluated_parts":len (pids ),
    "missing_record_ids":missing ,
    },
    "parameters":{
    "mm_max":args .mm_max ,
    "radius_min":args .radius_min ,
    "radius_max":args .radius_max ,
    "branches":branches ,
    },
    "result":result ,
    }
    return receipt 


def _print_summary (receipt :dict [str ,object ])->None :
    coverage =receipt ["coverage"]
    print (
    f"D7 audit: {coverage ['evaluated_parts']}/{coverage ['manifest_parts']} parts "
    f"(missing records={len (coverage ['missing_record_ids'])})"
    )
    print (
    f"{'pool':<13}{'branch':<24}{'null F1':>10}{'keep F1':>10}"
    f"{'high null':>11}{'recall':>9}"
    )
    for pool ,branches in receipt ["result"].items ():
        for branch ,metrics in branches .items ():
            null =metrics ["oracle_null"]
            keep =metrics ["keep_all"]
            high =null ["regime"].get ("very",{}).get ("f1",0.0 )
            print (
            f"{pool :<13}{branch :<24}{null ['weighted_f1']:>10.4f}"
            f"{keep ['weighted_f1']:>10.4f}{high :>11.4f}"
            f"{null ['raw_recall']:>9.4f}"
            )


def main ()->None :
    parser =argparse .ArgumentParser ()
    parser .add_argument ("--manifest",default ="results/d7_sinav_kumesi.json")
    parser .add_argument ("--records",default ="results/_der_yeni_G7BIRLESIK.pkl")
    parser .add_argument ("--cylinders",default ="results/_d7_silindirler.pkl")
    parser .add_argument ("--planars",default ="results/_d7_acikliklar.pkl")
    parser .add_argument ("--gate",default ="results/wire_gate_v6.pkl")
    parser .add_argument ("--output",default ="results/audit_d7_pose_dictionary_oracle.json")
    parser .add_argument ("--branches",nargs ="+",choices =sorted (PO .BRANCH_BUILDERS ),default =list (DEFAULT_BRANCHES ))
    parser .add_argument ("--mm-max",type =float ,default =8.0 )
    parser .add_argument ("--radius-min",type =float ,default =0.5 )
    parser .add_argument ("--radius-max",type =float ,default =4.5 )
    parser .add_argument ("--limit",type =int ,default =0 ,help ="deterministic smoke-test prefix")
    args =parser .parse_args ()
    receipt =run (args )
    _print_summary (receipt )
    output =Path (args .output )
    output .parent .mkdir (parents =True ,exist_ok =True )
    with output .open ("w",encoding ="utf-8")as stream :
        json .dump (receipt ,stream ,indent =1 ,ensure_ascii =False )
    print (f"receipt -> {output }")


if __name__ =="__main__":
    main ()
