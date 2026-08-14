# -*- coding: utf-8 -*-
"""Isolated P5-v3 physical-resource selector experiment.

This experiment is deliberately outside the live product path.  It trains ten
D6 only, chooses all hyper-parameters from manufacturer-out D6 predictions,
then writes one locked D7 receipt.  The candidate representation is the same
one used by :mod:`audit_d7_physical_node_oracle`:

* one spatially deduplicated resource row for base + cylinder + planar mouths;
* direction choices from axes within 8 mm plus the three part OBB/PCA axes;
* one selected signed direction, or explicit NULL, per resource row.

The important label correction versus P5-v2 is that positional error is
decomposed along the *GT axis*.  Using a proposed direction for that
decomposition can make a bad proposal appear laterally correct.

Typical full run (does not alter production models/configuration)::

    .venv/Scripts/python.exe p5v3_selector_exp.py --rebuild

Fast plumbing check::

    .venv/Scripts/python.exe p5v3_selector_exp.py --smoke 24 --cv-folds 4 --rebuild
"""
from __future__ import annotations 

import argparse 
import collections 
import dataclasses 
import hashlib 
import json 
import math 
import pickle 
import sys 
from pathlib import Path 
from typing import Iterable ,Mapping ,Sequence 

import numpy as np 
from scipy .optimize import linear_sum_assignment 
from scipy .spatial import cKDTree 

import audit_d7_physical_node_oracle as NODE_ORACLE 
import d6_record 
from sina_cluster import match_hungarian ,f1w 


# A script-mode dataclass would otherwise pickle as ``__main__.Part`` and the
# cache could only be reopened by another script-mode invocation.  Give the
# module a stable import identity before any Part object is serialized.
if __name__ =="__main__":
    sys .modules .setdefault ("p5v3_selector_exp",sys .modules [__name__ ])


ROOT =Path (__file__ ).resolve ().parent 
RESULTS =ROOT /"results"
CACHE_VERSION ="p5v3-resource-selector-v1"

# Frozen robot metric.  Do not tune these values.
ROBOT_LATERAL_MM =2.0 
ROBOT_AXIAL_MM =40.0 
ROBOT_ANGLE_DEG =10.0 
LOCAL_AXIS_MM =8.0 
DEDUPE_MM =0.5 

# connector3d.CONTACT == 1, connector3d.CABLE_ENTRY == 3.  Keeping the two
# integer constants here avoids importing the full inference stack to build a
# read-only feature cache.
CONTACT_CLASS =1 
CABLE_ENTRY_CLASS =3 

SPLITS ={
"d6":{
"manifest":RESULTS /"d6_exam_set.json",
"records":RESULTS /"_der_yeni_g10_n22.pkl",
"cylinders":RESULTS /"_d6_silindirler.pkl",
"planars":RESULTS /"_d6_acikliklar.pkl",
"probabilities":RESULTS /"_p1_olasilik_g10",
},
"d7":{
"manifest":RESULTS /"d7_exam_set.json",
"records":RESULTS /"_der_yeni_G7BIRLESIK.pkl",
"cylinders":RESULTS /"_d7_silindirler.pkl",
"planars":RESULTS /"_d7_acikliklar.pkl",
"probabilities":RESULTS /"_p1_olasilik_d7",
},
}


@dataclasses .dataclass 
class Part :
    pid :str 
    mfg :str 
    diag :float 
    gt_points :np .ndarray 
    gt_directions :np .ndarray 
    points :np .ndarray 
    node_x :np .ndarray 
    directions :list [np .ndarray ]
    direction_x :list [np .ndarray ]
    accept_y :np .ndarray 
    direction_y :list [np .ndarray ]
    source_counts :np .ndarray 


Part .__module__ ="p5v3_selector_exp"


def _unit (v :Sequence [float ])->np .ndarray |None :
    a =np .asarray (v ,float )
    n =float (np .linalg .norm (a ))
    if a .shape !=(3 ,)or not np .isfinite (a ).all ()or n <=1e-12 :
        return None 
    return a /n 


def _canonical_unsigned (v :Sequence [float ])->np .ndarray |None :
    a =_unit (v )
    if a is None :
        return None 
    k =int (np .argmax (np .abs (a )))
    return -a if a [k ]<0 else a 


def _finite (x :np .ndarray ,limit :float =1e6 )->np .ndarray :
    return np .nan_to_num (np .asarray (x ,float ),nan =0.0 ,posinf =limit ,
    neginf =-limit ).clip (-limit ,limit )


def _sha16 (path :Path )->str :
    h =hashlib .sha256 ()
    with path .open ("rb")as handle :
        for block in iter (lambda :handle .read (1024 *1024 ),b""):
            h .update (block )
    return h .hexdigest ()[:16 ]


def _probability_signature (directory :Path ,pids :Iterable [str ])->str :
    """Content signature of the exact per-part probability inputs."""
    h =hashlib .sha256 ()
    for pid in sorted (map (str ,pids )):
        path =directory /f"{pid }.npz"
        h .update (pid .encode ("utf-8"))
        h .update (b"\0")
        if not path .exists ():
            h .update (b"MISSING\0")
            continue 
        with path .open ("rb")as handle :
            for block in iter (lambda :handle .read (1024 *1024 ),b""):
                h .update (block )
    return h .hexdigest ()[:16 ]


def _source_hashes (cfg :Mapping [str ,Path ],pids :Iterable [str ])->dict [str ,str ]:
    """Every source that can change a cached row, including probability bytes."""
    return {
    "manifest_sha16":_sha16 (cfg ["manifest"]),
    "records_sha16":_sha16 (cfg ["records"]),
    "cylinders_sha16":_sha16 (cfg ["cylinders"]),
    "planars_sha16":_sha16 (cfg ["planars"]),
    "probabilities_sha16":_probability_signature (cfg ["probabilities"],pids ),
    }


def _balanced_subset (records :Mapping [str ,dict ],n :int )->dict [str ,dict ]:
    """Round-robin manufacturers so a smoke run remains manufacturer-out."""
    if not n or n >=len (records ):
        return dict (records )
    by_mfg :dict [str ,list [str ]]=collections .defaultdict (list )
    for pid ,row in records .items ():
        by_mfg [str (row ["mfg"])].append (pid )
    for pids in by_mfg .values ():
        pids .sort ()
    chosen =[]
    while len (chosen )<n and any (by_mfg .values ()):
        for mfg in sorted (by_mfg ):
            if by_mfg [mfg ]and len (chosen )<n :
                chosen .append (by_mfg [mfg ].pop (0 ))
    return {pid :records [pid ]for pid in chosen }


def _load_inputs (split :str ,smoke :int =0 ):
    cfg =SPLITS [split ]
    manifest =json .loads (cfg ["manifest"].read_text (encoding ="utf-8"))
    wanted =set (map (str ,manifest ["pidler"]))
    with cfg ["records"].open ("rb")as handle :
        records ={r ["pid"]:r for r in pickle .load (handle )
        if str (r ["pid"])in wanted }
    records =_balanced_subset (records ,smoke )
    with cfg ["cylinders"].open ("rb")as handle :
        cylinders =pickle .load (handle )
    with cfg ["planars"].open ("rb")as handle :
        planars =pickle .load (handle )
    return records ,cylinders ,planars ,cfg ,manifest 


def _member_metadata (pid :str ,node :Mapping [str ,object ],record :Mapping [str ,object ],
cylinders :Mapping [str ,Sequence [Mapping [str ,object ]]],
planars :Mapping [str ,Sequence [Mapping [str ,object ]]]):
    base_ids ,cyl_ids ,planar_ids =[],[],[]
    for member in node .get ("member_ids",()):
        bits =str (member ).split (":")
        try :
            if bits [0 ]=="base":
                base_ids .append (int (bits [1 ]))
            elif bits [0 ]=="cylinder":
                cyl_ids .append (int (bits [1 ]))
            elif bits [0 ]=="planar":
                planar_ids .append (int (bits [1 ]))
        except (IndexError ,ValueError ):
            continue 
    cyl_rows =[cylinders .get (pid ,())[i ]for i in sorted (set (cyl_ids ))
    if i <len (cylinders .get (pid ,()))]
    pla_rows =[planars .get (pid ,())[i ]for i in sorted (set (planar_ids ))
    if i <len (planars .get (pid ,()))]
    radii =[float (c .get ("radius",0.0 ))for c in cyl_rows ]
    lengths =[float (np .linalg .norm (np .asarray (c .get ("mouth_b",[0 ,0 ,0 ]),float )-
    np .asarray (c .get ("mouth_a",[0 ,0 ,0 ]),float )))
    for c in cyl_rows ]
    esd =[float (o .get ("esd_r",0.0 ))for o in pla_rows ]
    area =[float (o .get ("alan",0.0 ))for o in pla_rows ]
    perimeter =[float (o .get ("cevre",0.0 ))for o in pla_rows ]
    roundness =[float (o .get ("yuvarlaklik",0.0 ))for o in pla_rows ]
    diag =max (float (record ["diag"]),1e-6 )
    analytic =np .array ([
    len (base_ids ),len (cyl_ids ),len (planar_ids ),
    min (radii ,default =0.0 )/diag ,max (radii ,default =0.0 )/diag ,
    np .mean (radii )/diag if radii else 0.0 ,
    min (lengths ,default =0.0 )/diag ,max (lengths ,default =0.0 )/diag ,
    min (esd ,default =0.0 )/diag ,max (esd ,default =0.0 )/diag ,
    math .log1p (max (area ,default =0.0 )),
    math .log1p (max (perimeter ,default =0.0 )),
    max (roundness ,default =0.0 ),
    ],float )
    return sorted (set (base_ids )),analytic 


def _part_frame (vertices :np .ndarray ):
    vertices =np .asarray (vertices ,float )
    center =vertices .mean (0 )
    try :
        _u ,_s ,axes =np .linalg .svd (vertices -center ,full_matrices =False )
    except np .linalg .LinAlgError :
        axes =np .eye (3 )
    q =(vertices -center )@axes .T 
    lo ,hi =q .min (0 ),q .max (0 )
    span =np .maximum (hi -lo ,1e-6 )
    return center ,axes ,lo ,hi ,span 


def _axis_rows (nodes :Sequence [Mapping [str ,object ]]):
    """Unsigned local axis resources with their spatial/source provenance."""
    rows =[]
    seen ={}
    for ni ,node in enumerate (nodes ):
        src =set (node .get ("sources",()))
        flags =np .array ([float ("base"in src ),float ("cylinder"in src ),
        float ("planar"in src )],float )
        for raw in node .get ("directions",()):
            a =_canonical_unsigned (raw )
            if a is None :
                continue 
            key =tuple (np .round (np .r_ [node ["point"],a ],7 ))
            if key in seen :
                rows [seen [key ]][2 ]=np .maximum (rows [seen [key ]][2 ],flags )
            else :
                seen [key ]=len (rows )
                rows .append ([np .asarray (node ["point"],float ),a ,flags ,int (ni )])
    return rows 


def _direction_options (node_index :int ,nodes :Sequence [Mapping [str ,object ]],
axis_rows ,axis_tree :cKDTree |None ,
obb_axes :Sequence [np .ndarray ],local_mm :float =LOCAL_AXIS_MM ):
    """Signed local+OBB choices, cosine-deduped with merged provenance."""
    point =np .asarray (nodes [node_index ]["point"],float )
    own_axes =[_canonical_unsigned (a )for a in nodes [node_index ].get ("directions",())]
    own_axes =[a for a in own_axes if a is not None ]
    candidates =[]

    def add_axis (axis ,meta ):
        a =_canonical_unsigned (axis )
        if a is None :
            return 
        for sign in (1.0 ,-1.0 ):
            d =sign *a 
            hit =next ((k for k ,(old ,_m )in enumerate (candidates )
            if float (old @d )>=0.9999 ),None )
            if hit is None :
                candidates .append ([d ,np .asarray (meta ,float )])
            else :
            # provenance flags are OR/max; source distance is min.
                old =candidates [hit ][1 ]
                old [:5 ]=np .maximum (old [:5 ],meta [:5 ])
                old [5 ]=min (float (old [5 ]),float (meta [5 ]))

    for a in own_axes :
        add_axis (a ,[1 ,0 ,0 ,0 ,0 ,0 ])
    if axis_tree is not None :
        for ai in axis_tree .query_ball_point (point ,float (local_mm )):
            origin ,axis ,flags ,_source_node =axis_rows [int (ai )]
            dist =float (np .linalg .norm (np .asarray (origin )-point ))
            add_axis (axis ,[0 ,flags [0 ],flags [1 ],flags [2 ],0 ,
            dist /max (local_mm ,1e-6 )])
    for a in obb_axes :
        add_axis (a ,[0 ,0 ,0 ,0 ,1 ,1 ])
    if not candidates :
        add_axis ([0 ,0 ,1 ],[0 ,0 ,0 ,0 ,1 ,1 ])
    return candidates 


def assign_strict_labels (points :np .ndarray ,directions :Sequence [np .ndarray ],
gt_points :np .ndarray ,gt_directions :np .ndarray ):
    """Maximum-cardinality one-resource/one-GT labels under the frozen metric.

    Position is decomposed along ``gt_directions`` (not a proposed direction),
    and the direction test is signed.  Every acceptable direction for the
    assigned GT is positive so equivalent axes do not receive arbitrary noise.
    """
    points =np .asarray (points ,float ).reshape (-1 ,3 )
    gt_points =np .asarray (gt_points ,float ).reshape (-1 ,3 )
    gd =np .asarray (gt_directions ,float ).reshape (-1 ,3 )
    gd /=np .linalg .norm (gd ,axis =1 ,keepdims =True )+1e-12 
    accept =np .zeros (len (points ),np .int8 )
    dy =[np .zeros (len (d ),np .int8 )for d in directions ]
    if not len (points )or not len (gt_points ):
        return accept ,dy 

    diff =points [:,None ,:]-gt_points [None ,:,:]
    axial =np .sum (diff *gd [None ,:,:],axis =-1 )
    lateral =np .linalg .norm (diff -axial [...,None ]*gd [None ,:,:],axis =-1 )
    position_ok =((np .abs (axial )<=ROBOT_AXIAL_MM )&
    (lateral <=ROBOT_LATERAL_MM ))
    cos_min =float (np .cos (np .deg2rad (ROBOT_ANGLE_DEG )))
    feasible =np .zeros_like (position_ok )
    angle_cost =np .full_like (lateral ,180.0 )
    angle_ok_by_node =[]
    for i ,dirs in enumerate (directions ):
        dirs =np .asarray (dirs ,float ).reshape (-1 ,3 )
        dots =dirs @gd .T if len (dirs )else np .zeros ((0 ,len (gd )))
        aok =dots >=cos_min 
        angle_ok_by_node .append (aok )
        feasible [i ]=position_ok [i ]&aok .any (axis =0 )
        if len (dirs ):
            angle_cost [i ]=np .degrees (np .arccos (np .clip (dots .max (0 ),-1 ,1 )))

            # BIG dominates the sum of every feasible quality term, hence Hungarian
            # first maximises cardinality and only then breaks feasible ties by quality.
    quality =(lateral /ROBOT_LATERAL_MM +
    0.05 *np .abs (axial )/ROBOT_AXIAL_MM +
    0.20 *angle_cost /ROBOT_ANGLE_DEG )
    cost =np .where (feasible ,quality ,1e6 )
    rows ,cols =linear_sum_assignment (cost )
    for i ,j in zip (rows ,cols ):
        if not feasible [i ,j ]:
            continue 
        accept [i ]=1 
        dy [i ][angle_ok_by_node [i ][:,j ]]=1 
    return accept ,dy 


def build_part (pid :str ,record :Mapping [str ,object ],cylinders ,planars ,
probability_dir :Path )->Part :
    npz_path =probability_dir /f"{pid }.npz"
    with np .load (npz_path )as data :
        vertices =np .asarray (data ["V"],float )
        probs =np .asarray (data ["pbs"],float ).mean (0 )
    conn =probs [:,CONTACT_CLASS ]+probs [:,CABLE_ENTRY_CLASS ]
    vtree =cKDTree (vertices )
    center ,obb ,obb_lo ,obb_hi ,obb_span =_part_frame (vertices )

    source =NODE_ORACLE .build_source_nodes (pid ,record ,cylinders ,planars )
    nodes =NODE_ORACLE .cluster_nodes (
    source ["base"]+source ["cylinder"]+source ["planar"],DEDUPE_MM )
    points =np .asarray ([n ["point"]for n in nodes ],float ).reshape (-1 ,3 )
    ntree =cKDTree (points )if len (points )else None 
    axis_rows =_axis_rows (nodes )
    atree =cKDTree (np .asarray ([a [0 ]for a in axis_rows ],float ))if axis_rows else None 
    obb_axes =[np .asarray (a ,float )for a in obb ]

    base_p =np .asarray (record .get ("P",[]),float ).reshape (-1 ,3 )
    base_d =np .asarray (record .get ("Pd",[]),float ).reshape (-1 ,3 )
    base_tree =cKDTree (base_p )if len (base_p )else None 
    raw_x =d6_record .x58 (record )
    if raw_x is None or len (raw_x )!=len (base_p ):
        raw_x =np .zeros ((len (base_p ),58 ),float )
    raw_x =_finite (raw_x )

    node_features ,all_dirs ,dir_features ,source_counts =[],[],[],[]
    diag =max (float (record ["diag"]),1e-6 )
    for ni ,(node ,point )in enumerate (zip (nodes ,points )):
        base_ids ,analytic =_member_metadata (pid ,node ,record ,cylinders ,planars )
        src =set (node .get ("sources",()))
        src_flags =np .array ([float ("base"in src ),float ("cylinder"in src ),
        float ("planar"in src )],float )
        source_counts .append (analytic [:3 ])

        if base_tree is not None :
            nearest_base_dist ,nearest_base =base_tree .query (point ,k =1 )
            nearest_base =int (nearest_base )
            nearest_base_dir =_unit (base_d [nearest_base ])
            nearest_base_x =raw_x [nearest_base ]
        else :
            nearest_base_dist ,nearest_base =diag ,-1 
            nearest_base_dir =None 
            nearest_base_x =np .zeros (58 )

        q =(point -center )@obb .T 
        obb_abs =np .abs ((q -0.5 *(obb_lo +obb_hi ))/obb_span )
        boundary =np .minimum (np .abs (q -obb_lo ),np .abs (obb_hi -q ))/diag 
        local_ids =np .asarray (vtree .query_ball_point (point ,8.0 ),dtype =int )
        local_v =vertices [local_ids ]if len (local_ids )else np .zeros ((0 ,3 ))
        local_c =conn [local_ids ]if len (local_ids )else np .zeros (0 )
        vd =np .linalg .norm (local_v -point ,axis =1 )if len (local_v )else np .zeros (0 )
        prob_stats =[]
        for radius in (1.0 ,2.0 ,4.0 ,8.0 ):
            z =local_c [vd <=radius ]
            prob_stats .extend ([
            math .log1p (len (z )),float (z .mean ())if len (z )else 0.0 ,
            float (z .max ())if len (z )else 0.0 ,
            float (np .quantile (z ,0.9 ))if len (z )else 0.0 ,
            ])
        density =[]
        if ntree is not None :
            for radius in (0.5 ,2.0 ,4.0 ,8.0 ):
                density .append (math .log1p (max (0 ,len (ntree .query_ball_point (point ,radius ))-1 )))
        else :
            density =[0.0 ]*4 

        nx =np .r_ [
        src_flags ,math .log1p (len (node .get ("member_ids",()))),
        math .log1p (len (node .get ("directions",()))),analytic ,
        np .linalg .norm (point -center )/diag ,obb_abs ,boundary ,
        density ,prob_stats ,
        float (bool (base_ids )),float (nearest_base_dist )/diag ,
        nearest_base_x ,
        ]
        nx =_finite (nx )
        node_features .append (nx )

        candidates =_direction_options (ni ,nodes ,axis_rows ,atree ,obb_axes )
        dirs =np .asarray ([c [0 ]for c in candidates ],float )
        dx =[]
        radial =point -center 
        radial /=np .linalg .norm (radial )+1e-12 
        for direction ,provenance in candidates :
            obb_dot =np .asarray ([float (direction @a )for a in obb_axes ],float )
            base_dot =float (direction @nearest_base_dir )if nearest_base_dir is not None else 0.0 
            rel =local_v -point 
            if len (rel ):
                along =rel @direction 
                perp =np .linalg .norm (rel -along [:,None ]*direction [None ,:],axis =1 )
                forward =local_c [(along >=0 )&(along <=8 )&(perp <=3 )]
                backward =local_c [(along <=0 )&(along >=-8 )&(perp <=3 )]
            else :
                forward =backward =np .zeros (0 )
            fmean =float (forward .mean ())if len (forward )else 0.0 
            bmean =float (backward .mean ())if len (backward )else 0.0 
            dx .append (np .r_ [
            provenance ,
            base_dot ,abs (base_dot ),float (direction @radial ),
            abs (float (direction @radial )),obb_dot ,np .abs (obb_dot ),
            fmean ,bmean ,fmean -bmean ,
            math .log1p (len (forward )),math .log1p (len (backward )),
            ])
        all_dirs .append (dirs )
        dir_features .append (_finite (np .asarray (dx ,float )))

    node_x =_finite (np .asarray (node_features ,float ))
    gt_points =np .asarray (record .get ("G",[]),float ).reshape (-1 ,3 )
    gt_dirs =np .asarray (record .get ("Gd",[]),float ).reshape (-1 ,3 )
    accept_y ,direction_y =assign_strict_labels (points ,all_dirs ,gt_points ,gt_dirs )
    return Part (
    pid =str (pid ),mfg =str (record ["mfg"]),diag =float (record ["diag"]),
    gt_points =gt_points ,gt_directions =gt_dirs ,points =points ,
    node_x =node_x ,directions =all_dirs ,direction_x =dir_features ,
    accept_y =accept_y ,direction_y =direction_y ,
    source_counts =np .asarray (source_counts ,int ),
    )


def cache_path (split :str ,smoke :int =0 )->Path :
    suffix =f"_smoke{smoke }"if smoke else ""
    return RESULTS /f"_p5v3_selector_{split }{suffix }.pkl"


def build_cache (split :str ,smoke :int =0 )->dict :
    records ,cylinders ,planars ,cfg ,manifest =_load_inputs (split ,smoke )
    parts =[]
    missing =[]
    for k ,pid in enumerate (sorted (records ),1 ):
        npz =cfg ["probabilities"]/f"{pid }.npz"
        if not npz .exists ():
            missing .append (pid )
            continue 
        parts .append (build_part (pid ,records [pid ],cylinders ,planars ,
        cfg ["probabilities"]))
        if k %25 ==0 or k ==len (records ):
            print (f"  {split }: {k }/{len (records )} part",flush =True )
    selected_pids =sorted (records )
    payload ={
    "version":CACHE_VERSION ,
    "split":split ,
    "smoke":int (smoke ),
    "selected_pids":selected_pids ,
    "source_hashes":_source_hashes (cfg ,selected_pids ),
    "missing":missing ,
    "parts":parts ,
    }
    out =cache_path (split ,smoke )
    with out .open ("wb")as handle :
        pickle .dump (payload ,handle ,protocol =pickle .HIGHEST_PROTOCOL )
    print (f"cache -> {out } ({len (parts )} parts; missing {len (missing )})")
    return payload 


def load_cache (split :str ,smoke :int =0 )->dict :
    with cache_path (split ,smoke ).open ("rb")as handle :
        payload =pickle .load (handle )
    if payload .get ("version")!=CACHE_VERSION :
        raise RuntimeError ("stale P5-v3 cache; rerun with --rebuild")
    expected =_source_hashes (SPLITS [split ],payload .get ("selected_pids",()))
    if payload .get ("source_hashes")!=expected :
        changed =sorted (k for k in expected 
        if payload .get ("source_hashes",{}).get (k )!=expected [k ])
        raise RuntimeError (f"stale P5-v3 cache sources {changed }; rerun with --rebuild")
    return payload 


def _manufacturer_class_weights (mfg :np .ndarray ,y :np .ndarray )->np .ndarray :
    """Each manufacturer and each available class gets equal total weight."""
    mfg =np .asarray (mfg ,object )
    y =np .asarray (y ,int )
    w =np .zeros (len (y ),float )
    brands =sorted (set (map (str ,mfg )))
    for brand in brands :
        bm =mfg ==brand 
        classes =np .unique (y [bm ])
        for cls in classes :
            mask =bm &(y ==cls )
            w [mask ]=1.0 /(len (brands )*len (classes )*max (mask .sum (),1 ))
    return w *len (w )/max (w .sum (),1e-12 )


def _flatten_nodes (parts :Sequence [Part ]):
    x =np .vstack ([p .node_x for p in parts ])
    y =np .concatenate ([p .accept_y for p in parts ])
    m =np .concatenate ([[p .mfg ]*len (p .node_x )for p in parts ])
    return x ,y ,m 


def _flatten_directions (parts :Sequence [Part ]):
    x ,y ,m =[],[],[]
    for part in parts :
        for i in np .where (part .accept_y ==1 )[0 ]:
            if not len (part .direction_x [i ]):
                continue 
            nx =np .repeat (part .node_x [i ][None ,:],len (part .direction_x [i ]),axis =0 )
            x .append (np .hstack ([nx ,part .direction_x [i ]]))
            y .append (part .direction_y [i ])
            m .extend ([part .mfg ]*len (part .direction_x [i ]))
    if not x :
        raise RuntimeError ("no matched direction training rows")
    return np .vstack (x ),np .concatenate (y ),np .asarray (m ,object )


def fit_models (parts :Sequence [Part ],max_iter :int =140 ,seed :int =0 ):
    from sklearn .ensemble import HistGradientBoostingClassifier 

    xn ,yn ,mn =_flatten_nodes (parts )
    xd ,yd ,md =_flatten_directions (parts )
    common =dict (max_iter =int (max_iter ),learning_rate =0.07 ,
    max_leaf_nodes =31 ,min_samples_leaf =20 ,
    l2_regularization =2.0 ,random_state =int (seed ))
    accept =HistGradientBoostingClassifier (**common )
    direction =HistGradientBoostingClassifier (**common )
    accept .fit (xn ,yn ,sample_weight =_manufacturer_class_weights (mn ,yn ))
    direction .fit (xd ,yd ,sample_weight =_manufacturer_class_weights (md ,yd ))
    return accept ,direction 


def _logit (p ):
    p =np .clip (np .asarray (p ,float ),1e-6 ,1 -1e-6 )
    return np .log (p /(1 -p ))


def score_parts (parts :Sequence [Part ],accept_model ,direction_model ):
    scored =[]
    for part in parts :
        pa =accept_model .predict_proba (part .node_x )[:,1 ]
        best_pd ,best_dir =np .zeros (len (part .points )),np .zeros ((len (part .points ),3 ))
        for i ,(dirs ,dx )in enumerate (zip (part .directions ,part .direction_x )):
            if not len (dx ):
                continue 
            nx =np .repeat (part .node_x [i ][None ,:],len (dx ),axis =0 )
            pd =direction_model .predict_proba (np .hstack ([nx ,dx ]))[:,1 ]
            k =int (np .argmax (pd ))
            best_pd [i ],best_dir [i ]=float (pd [k ]),dirs [k ]
        scored .append ({"part":part ,"accept_logit":_logit (pa ),
        "direction_logit":_logit (best_pd ),
        "best_direction":best_dir })
    return scored 


def _predictions (scored ,lam :float ,threshold :float ):
    out =[]
    for row in scored :
        score =row ["accept_logit"]+float (lam )*row ["direction_logit"]
        keep =score >=float (threshold )
        part =row ["part"]
        out .append ((part .points [keep ],row ["best_direction"][keep ]))
    return out 


def evaluate (scored ,lam :float ,threshold :float ):
    rows ,per_mfg =[],collections .defaultdict (list )
    predictions =_predictions (scored ,lam ,threshold )
    accepted =0 
    for row ,(points ,directions )in zip (scored ,predictions ):
        part =row ["part"]
        regime ="very"if len (part .gt_points )>=8 else "low"
        tp ,fp ,fn ,_ =match_hungarian (
        points ,directions ,part .gt_points ,part .gt_directions ,part .diag ,
        ROBOT_LATERAL_MM ,ROBOT_ANGLE_DEG ,False ,signed =True ,
        eksen_tol =ROBOT_AXIAL_MM ,
        )
        metric_row =(regime ,tp ,fp ,fn )
        rows .append (metric_row )
        per_mfg [part .mfg ].append (metric_row )
        accepted +=len (points )
    by_mfg ={m :float (f1w (r ))for m ,r in sorted (per_mfg .items ())}
    vals =list (by_mfg .values ())
    return {
    "robot_f1":float (f1w (rows )),
    "manufacturer_macro_f1":float (np .mean (vals ))if vals else 0.0 ,
    "manufacturer_worst_f1":float (np .min (vals ))if vals else 0.0 ,
    "per_manufacturer":by_mfg ,
    "accepted":int (accepted ),
    }


def manufacturer_folds (parts :Sequence [Part ],n_folds :int ):
    brands =sorted ({p .mfg for p in parts })
    n_folds =max (2 ,min (int (n_folds ),len (brands )))
    # Greedy balance by number of nodes while keeping whole manufacturers out.
    sizes =collections .Counter ()
    for p in parts :
        sizes [p .mfg ]+=len (p .points )
    buckets =[[]for _ in range (n_folds )]
    load =[0 ]*n_folds 
    for brand in sorted (brands ,key =lambda m :(-sizes [m ],m )):
        k =int (np .argmin (load ))
        buckets [k ].append (brand )
        load [k ]+=sizes [brand ]
    return [set (b )for b in buckets ]


def oof_scores (parts :Sequence [Part ],n_folds :int ,max_iter :int ):
    out =[]
    for fold ,held in enumerate (manufacturer_folds (parts ,n_folds ),1 ):
        train =[p for p in parts if p .mfg not in held ]
        test =[p for p in parts if p .mfg in held ]
        print (f"CV {fold }: held={sorted (held )} train={len (train )} test={len (test )}",
        flush =True )
        ma ,md =fit_models (train ,max_iter =max_iter ,seed =fold )
        out .extend (score_parts (test ,ma ,md ))
    return out 


def choose_null_threshold (scored ):
    """D6 manufacturer-out choice only; D7 is never consulted."""
    best =None 
    for lam in (0.0 ,0.25 ,0.5 ,0.75 ,1.0 ):
        all_score =np .concatenate ([r ["accept_logit"]+lam *r ["direction_logit"]
        for r in scored ])
        # Actual positives are a few percent.  Dense high-quantile sweep covers
        # useful precision/recall regimes without thousands of full matchings.
        qs =np .r_ [np .linspace (0.80 ,0.94 ,15 ),
        np .linspace (0.945 ,0.995 ,35 ),
        np .linspace (0.996 ,0.9995 ,8 )]
        thresholds =np .unique (np .quantile (all_score ,qs ))
        for threshold in thresholds :
            metric =evaluate (scored ,lam ,float (threshold ))
            key =(metric ["manufacturer_macro_f1"],metric ["robot_f1"],
            metric ["manufacturer_worst_f1"])
            if best is None or key >best [0 ]:
                best =(key ,float (lam ),float (threshold ),metric )
    assert best is not None 
    return {"lambda":best [1 ],"null_threshold":best [2 ],
    "oof":best [3 ]}


def run_experiment (d6 :Sequence [Part ],d7 :Sequence [Part ],n_folds :int ,
max_iter :int ,smoke :int =0 ,provenance :Mapping |None =None ):
    print ("manufacturer-out D6 scoring",flush =True )
    oof =oof_scores (d6 ,n_folds =n_folds ,max_iter =max_iter )
    choice =choose_null_threshold (oof )
    print ("D6 OOF choice:",json .dumps (choice ,indent =1 ),flush =True )

    print ("refit all D6; locked D7 application",flush =True )
    accept ,direction =fit_models (d6 ,max_iter =max_iter ,seed =101 )
    d7_scored =score_parts (d7 ,accept ,direction )
    d7_metric =evaluate (d7_scored ,choice ["lambda"],choice ["null_threshold"])
    print ("D7:",json .dumps (d7_metric ,indent =1 ),flush =True )

    suffix =f"_smoke{smoke }"if smoke else ""
    model_path =RESULTS /f"p5v3_selector_d6{suffix }.pkl"
    with model_path .open ("wb")as handle :
        pickle .dump ({"version":CACHE_VERSION ,"accept":accept ,
        "direction":direction ,"choice":choice },handle ,
        protocol =pickle .HIGHEST_PROTOCOL )
    receipt ={
    "experiment":CACHE_VERSION ,
    "production_changed":False ,
    "train":"D6 only",
    "validation":f"D6 manufacturer-out {n_folds }-fold",
    "test":"D7 locked application; D7 is DEV, not final",
    "contract":{
    "resource":"base+cylinder+planar, 0.5mm spatial cluster",
    "direction":"signed local axes within 8mm + OBB/PCA axes",
    "null":"joint accept/direction logit versus frozen OOF threshold",
    "capacity":"at most one signed direction per physical resource",
    "label_position_axis":"GT axis",
    "robot":"lateral<=2mm, |axial|<=40mm, signed angle<=10deg",
    },
    "selection":choice ,
    "d7":d7_metric ,
    "n_d6_parts":len (d6 ),
    "n_d7_parts":len (d7 ),
    "cache_version":CACHE_VERSION ,
    "sources":dict (provenance or {}),
    }
    receipt_path =RESULTS /f"p5v3_d6_to_d7{suffix }.json"
    receipt_path .write_text (json .dumps (receipt ,indent =1 ),encoding ="utf-8")
    print (f"model -> {model_path }\nreceipt -> {receipt_path }")
    return receipt 


def main ():
    parser =argparse .ArgumentParser ()
    parser .add_argument ("--rebuild",action ="store_true")
    parser .add_argument ("--build-only",action ="store_true")
    parser .add_argument ("--smoke",type =int ,default =0 ,
    help ="balanced number of parts per split; 0=full")
    parser .add_argument ("--cv-folds",type =int ,default =8 ,
    help ="whole-manufacturer folds; 8 is D6 LOMO")
    parser .add_argument ("--max-iter",type =int ,default =140 )
    args =parser .parse_args ()

    split_payload ={}
    for split in ("d6","d7"):
        path =cache_path (split ,args .smoke )
        if args .rebuild or not path .exists ():
            split_payload [split ]=build_cache (split ,args .smoke )
        else :
            split_payload [split ]=load_cache (split ,args .smoke )
            print (f"cache <- {path } ({len (split_payload [split ]['parts'])} parts)")
    if not args .build_only :
        provenance ={}
        for split ,payload in split_payload .items ():
            provenance [split ]={
            "cache_path":str (cache_path (split ,args .smoke )),
            "cache_sha16":_sha16 (cache_path (split ,args .smoke )),
            "source_hashes":payload ["source_hashes"],
            "selected_count":len (payload .get ("selected_pids",())),
            "scored_count":len (payload ["parts"]),
            "missing_count":len (payload .get ("missing",())),
            "missing_ids":payload .get ("missing",()),
            }
        run_experiment (split_payload ["d6"]["parts"],
        split_payload ["d7"]["parts"],
        n_folds =args .cv_folds ,max_iter =args .max_iter ,
        smoke =args .smoke ,provenance =provenance )


if __name__ =="__main__":
    main ()
