"""Build a five-part, STEP-native WSCAD connection-point review package.

The STEP geometry comes from ``_cad_eval_pxc``.  Connection-point labels come
from the exact-catalog Phoenix Contact JSON files ten the Desktop and are moved
from the JSON coordinate frame into the STEP coordinate frame with
``cad_eval.align_frames``.  No vertex cap or random face dropping is used.

The output is deliberately marked as AI-assisted/manufacturer-derived.  It is
ready for visual review and training experiments, but is not called human-GT
until a person signs it off.
"""

from __future__ import annotations 

import argparse 
import hashlib 
import json 
import math 
import os 
from pathlib import Path 
import shutil 
import struct 
from typing import Iterable 

import numpy as np 
from PIL import Image ,ImageDraw ,ImageFont 

import cad_eval 
import export_glb 
import json_dataset 
import step_to_json 


REPO_ROOT =Path (__file__ ).resolve ().parent 
DEFAULT_TEACHER_DIR =Path (r"C:\Users\DE00024082\Desktop\JSON")
DEFAULT_OUTPUT_DIR =Path (r"C:\Users\DE00024082\Desktop\check\wscad_labeled_5")
STEP_DIR =REPO_ROOT /"_cad_eval_pxc"
DEFL_MM =0.3 
MAX_ALIGNMENT_RESIDUAL_MM =2.0 
MAX_NEAREST_VERTEX_MM =5.0 


PARTS =[
{
"catalog":"3031238",
"part_nr":"PXC.3031238",
"product_name":"ST 2,5-PE",
"terminal_type":"spring-cage protective-conductor terminal",
"official_url":"https://www.phoenixcontact.com/de-de/produkte/schutzleiterklemme-st-25-pe-3031238",
"official_connection_count":2 ,
"step_file":"wscaduniverse_3031238_2026-06-21-22-27-09.stp",
},
{
"catalog":"3036550",
"part_nr":"PXC.3036550",
"product_name":"ST 4-HESILED 60 (5X20)",
"terminal_type":"spring-cage fuse terminal",
"official_url":"https://www.phoenixcontact.com/de-de/produkte/sicherungsklemme-st-4-hesiled-60-5x20-3036550",
"official_connection_count":2 ,
"step_file":"wscaduniverse_3036550_2026-07-10-16-05-43.stp",
},
{
"catalog":"3048357",
"part_nr":"PXC.3048357",
"product_name":"USEN 14 N",
"terminal_type":"screw fuse terminal",
"official_url":"https://www.phoenixcontact.com/de-de/produkte/sicherungsklemme-usen-14-n-3048357",
"official_connection_count":2 ,
"step_file":"wscaduniverse_3048357_2026-07-10-16-06-15.stp",
},
{
"catalog":"3211813",
"part_nr":"PXC.3211813",
"product_name":"PT 6",
"terminal_type":"push-in feed-through terminal",
"official_url":"https://www.phoenixcontact.com/de-de/produkte/durchgangsklemme-pt-6-3211813",
"official_connection_count":2 ,
"step_file":"wscaduniverse_3211813_2026-06-21-22-25-47.stp",
},
{
"catalog":"3212140",
"part_nr":"PXC.3212140",
"product_name":"PTME 4 WH",
"terminal_type":"push-in test-disconnect terminal",
"official_url":"https://www.phoenixcontact.com/de/produkte/3212140/pdf",
"official_connection_count":2 ,
"step_file":"wscaduniverse_3212140_2026-07-10-16-06-54.stp",
# This housing is close to 180-degree symmetric in its coarse JSON
# mesh.  Residual-only alignment can therefore put the two entry
# directions ten the underside.  In the WSCAD STEP frame the actual
# conductor entries face +Z, which disambiguates the two candidates.
"step_direction_constraint":{"axis":2 ,"sign":1.0 ,"min_component":0.5 },
},
]


def sha256 (path :Path )->str :
    h =hashlib .sha256 ()
    with path .open ("rb")as fh :
        for chunk in iter (lambda :fh .read (1024 *1024 ),b""):
            h .update (chunk )
    return h .hexdigest ()


def write_json (path :Path ,obj :object ,*,compact :bool =False )->None :
    tmp =path .with_suffix (path .suffix +".tmp")
    with tmp .open ("w",encoding ="utf-8")as fh :
        if compact :
            json .dump (obj ,fh ,ensure_ascii =False ,separators =(",",":"))
        else :
            json .dump (obj ,fh ,ensure_ascii =False ,indent =2 )
            fh .write ("\n")
    os .replace (tmp ,path )


def _step_signature_ok (path :Path )->bool :
    data =path .read_bytes ()
    upper =data .upper ()
    return b"ISO-10303-21"in upper [:4096 ]and b"END-ISO-10303-21"in upper [-4096 :]


def _remove_bad_faces (vertices :np .ndarray ,faces :np .ndarray )->tuple [np .ndarray ,int ]:
    faces =np .asarray (faces ,dtype =np .int64 ).reshape (-1 ,3 )
    distinct =(
    (faces [:,0 ]!=faces [:,1 ])
    &(faces [:,1 ]!=faces [:,2 ])
    &(faces [:,0 ]!=faces [:,2 ])
    )
    good_faces =faces [distinct ]
    a =vertices [good_faces [:,1 ]]-vertices [good_faces [:,0 ]]
    b =vertices [good_faces [:,2 ]]-vertices [good_faces [:,0 ]]
    area2 =np .linalg .norm (np .cross (a ,b ),axis =1 )
    nonzero =area2 >1e-10 
    cleaned =good_faces [nonzero ]
    return cleaned ,int (len (faces )-len (cleaned ))


def _align_with_direction_constraint (
vertices :np .ndarray ,
teacher_vertices :np .ndarray ,
teacher_directions :np .ndarray ,
constraint :dict |None ,
*,
sample :int ,
seed :int ,
)->tuple [np .ndarray ,np .ndarray ,float ,dict ]:
    """Align frames, optionally resolving a near-symmetry with CP directions."""
    if not constraint :
        rotation ,translation ,residual =cad_eval .align_frames (
        vertices ,teacher_vertices ,sample =sample ,seed =seed 
        )
        return rotation ,translation ,residual ,{"applied":False }

    from scipy .spatial import cKDTree 

    axis =int (constraint ["axis"])
    sign =float (constraint ["sign"])
    min_component =float (constraint ["min_component"])
    rng =np .random .default_rng (seed )
    idx =rng .choice (len (teacher_vertices ),min (sample ,len (teacher_vertices )),replace =False )
    sampled_teacher =teacher_vertices [idx ]
    teacher_center =0.5 *(teacher_vertices .min (axis =0 )+teacher_vertices .max (axis =0 ))
    step_center =0.5 *(vertices .min (axis =0 )+vertices .max (axis =0 ))
    tree =cKDTree (vertices )
    candidates =[]
    for candidate_index ,rotation in enumerate (cad_eval ._PERMS ):
        step_directions =teacher_directions @rotation 
        direction_components =sign *step_directions [:,axis ]
        if float (direction_components .min ())<min_component :
            continue 
        translation =teacher_center -rotation @step_center 
        query_step =(sampled_teacher -translation )@rotation 
        distances ,_ =tree .query (query_step ,k =1 )
        candidates .append (
        (
        float (distances .mean ()),
        candidate_index ,
        rotation ,
        translation ,
        direction_components ,
        )
        )
    if not candidates :
        raise ValueError (f"no alignment candidate satisfies direction constraint {constraint }")
    residual ,candidate_index ,rotation ,translation ,components =min (candidates ,key =lambda x :x [0 ])
    return rotation ,translation ,residual ,{
    "applied":True ,
    "axis":axis ,
    "axis_name":"XYZ"[axis ],
    "sign":sign ,
    "min_component":min_component ,
    "selected_candidate_index":candidate_index ,
    "selected_direction_components":components .tolist (),
    "passing_candidate_count":len (candidates ),
    "reason":"resolve near-symmetric geometry using known STEP entry-facing direction",
    }


def _cp_dicts (points :np .ndarray ,directions :np .ndarray ,names :Iterable [str ])->list [dict ]:
    result =[]
    for i ,(point ,direction ,name )in enumerate (zip (points ,directions ,names )):
        result .append (
        {
        "Index":i ,
        "Name":str (name ),
        "Point":{"X":float (point [0 ]),"Y":float (point [1 ]),"Z":float (point [2 ])},
        "InsertDirection":{
        "X":float (direction [0 ]),
        "Y":float (direction [1 ]),
        "Z":float (direction [2 ]),
        },
        }
        )
    return result 


def _abb_mesh (part_nr :str ,vertices :np .ndarray ,faces :np .ndarray ,cps :list [dict ])->dict :
    lo =vertices .min (axis =0 )
    hi =vertices .max (axis =0 )
    return {
    "PartNr":part_nr ,
    "Graphic3d":{
    "Points":[
    {"X":float (v [0 ]),"Y":float (v [1 ]),"Z":float (v [2 ])}
    for v in vertices 
    ],
    "Indices":faces .reshape (-1 ).astype (int ).tolist (),
    },
    "BoundingBox":{
    "Dimension":{
    "X":float (hi [0 ]-lo [0 ]),
    "Y":float (hi [1 ]-lo [1 ]),
    "Z":float (hi [2 ]-lo [2 ]),
    },
    "Location":{"X":float (lo [0 ]),"Y":float (lo [1 ]),"Z":float (lo [2 ])},
    },
    "ConnectionPoints":cps ,
    "LabelProvenance":{
    "source":"exact-catalog Phoenix Contact manufacturer JSON",
    "transfer":"signed-axis STEP/JSON frame alignment",
    "verification_status":"automatic QA passed; human sign-off pending",
    "human_gt":False ,
    },
    }


def _sample_render_points (vertices :np .ndarray ,faces :np .ndarray ,max_points :int =180_000 )->np .ndarray :
    rng =np .random .default_rng (20260716 )
    if len (faces ):
        face_idx =np .arange (len (faces ))
        if len (face_idx )>max_points //2 :
            face_idx =rng .choice (face_idx ,max_points //2 ,replace =False )
        centroids =vertices [faces [face_idx ]].mean (axis =1 )
        points =np .concatenate ([vertices ,centroids ],axis =0 )
    else :
        points =vertices 
    if len (points )>max_points :
        points =points [rng .choice (len (points ),max_points ,replace =False )]
    return points 


def _draw_projection (
points :np .ndarray ,
cp_points :np .ndarray ,
cp_dirs :np .ndarray ,
u_axis :np .ndarray ,
v_axis :np .ndarray ,
depth_axis :np .ndarray ,
title :str ,
width :int =560 ,
height :int =420 ,
)->Image .Image :
    image =np .full ((height ,width ,3 ),250 ,dtype =np .uint8 )
    margin_x ,margin_top ,margin_bottom =30 ,42 ,25 
    u =points @u_axis 
    v =points @v_axis 
    z =points @depth_axis 
    u_lo ,u_hi =float (u .min ()),float (u .max ())
    v_lo ,v_hi =float (v .min ()),float (v .max ())
    u_span =max (u_hi -u_lo ,1e-6 )
    v_span =max (v_hi -v_lo ,1e-6 )
    sx =(width -2 *margin_x )/u_span 
    sy =(height -margin_top -margin_bottom )/v_span 
    scale =min (sx ,sy )
    x0 =0.5 *(width -scale *u_span )-scale *u_lo 
    y0 =0.5 *(height +margin_top -margin_bottom +scale *v_span )+scale *v_lo 
    px =np .rint (x0 +scale *u ).astype (np .int64 )
    py =np .rint (y0 -scale *v ).astype (np .int64 )
    valid =(px >=1 )&(px <width -1 )&(py >=margin_top )&(py <height -margin_bottom )
    px ,py ,z =px [valid ],py [valid ],z [valid ]
    z_lo ,z_hi =float (z .min ()),float (z .max ())
    zn =(z -z_lo )/max (z_hi -z_lo ,1e-9 )
    order =np .argsort (zn )
    shade =(170 -85 *zn ).astype (np .uint8 )
    for offset_x ,offset_y in ((0 ,0 ),(1 ,0 ),(0 ,1 )):
        xx =np .clip (px [order ]+offset_x ,0 ,width -1 )
        yy =np .clip (py [order ]+offset_y ,0 ,height -1 )
        s =shade [order ]
        image [yy ,xx ,0 ]=s 
        image [yy ,xx ,1 ]=np .minimum (s +20 ,255 )
        image [yy ,xx ,2 ]=np .minimum (s +38 ,255 )

    panel =Image .fromarray (image ,mode ="RGB")
    draw =ImageDraw .Draw (panel )
    font =ImageFont .load_default ()
    draw .rectangle ((0 ,0 ,width -1 ,height -1 ),outline =(185 ,190 ,198 ),width =1 )
    draw .text ((12 ,13 ),title ,fill =(25 ,31 ,39 ),font =font )
    diagonal =float (np .linalg .norm (np .ptp (points ,axis =0 )))
    arrow_len =max (diagonal *0.16 ,4.0 )
    for i ,(point ,direction )in enumerate (zip (cp_points ,cp_dirs ),start =1 ):
        start_u ,start_v =float (point @u_axis ),float (point @v_axis )
        end =point +direction *arrow_len 
        end_u ,end_v =float (end @u_axis ),float (end @v_axis )
        x1 ,y1 =x0 +scale *start_u ,y0 -scale *start_v 
        x2 ,y2 =x0 +scale *end_u ,y0 -scale *end_v 
        radius =8 
        draw .ellipse ((x1 -radius ,y1 -radius ,x1 +radius ,y1 +radius ),fill =(225 ,38 ,45 ),outline =(92 ,0 ,0 ),width =2 )
        projected_len =math .hypot (x2 -x1 ,y2 -y1 )
        if projected_len >=4 :
            draw .line ((x1 ,y1 ,x2 ,y2 ),fill =(0 ,150 ,65 ),width =4 )
            angle =math .atan2 (y2 -y1 ,x2 -x1 )
            head =10 
            for delta in (2.55 ,-2.55 ):
                hx =x2 +head *math .cos (angle +delta )
                hy =y2 +head *math .sin (angle +delta )
                draw .line ((x2 ,y2 ,hx ,hy ),fill =(0 ,150 ,65 ),width =3 )
        else :
            depth_sign =float (direction @depth_axis )
            if depth_sign >=0 :
                draw .ellipse ((x1 -3 ,y1 -3 ,x1 +3 ,y1 +3 ),fill =(255 ,255 ,255 ))
            else :
                draw .line ((x1 -4 ,y1 -4 ,x1 +4 ,y1 +4 ),fill =(255 ,255 ,255 ),width =2 )
                draw .line ((x1 -4 ,y1 +4 ,x1 +4 ,y1 -4 ),fill =(255 ,255 ,255 ),width =2 )
        draw .text ((x1 +11 ,y1 -14 ),f"CP{i }",fill =(145 ,0 ,0 ),font =font )
    return panel 


def render_qa_png (
path :Path ,
title :str ,
vertices :np .ndarray ,
faces :np .ndarray ,
cp_points :np .ndarray ,
cp_dirs :np .ndarray ,
)->None :
    sample =_sample_render_points (vertices ,faces )
    views =[
    ("view +Z",(1 ,0 ,0 ),(0 ,1 ,0 ),(0 ,0 ,1 )),
    ("view -Z",(-1 ,0 ,0 ),(0 ,1 ,0 ),(0 ,0 ,-1 )),
    ("view +X",(0 ,1 ,0 ),(0 ,0 ,1 ),(1 ,0 ,0 )),
    ("view -X",(0 ,-1 ,0 ),(0 ,0 ,1 ),(-1 ,0 ,0 )),
    ("view +Y",(1 ,0 ,0 ),(0 ,0 ,1 ),(0 ,1 ,0 )),
    ("view -Y",(-1 ,0 ,0 ),(0 ,0 ,1 ),(0 ,-1 ,0 )),
    ]
    panels =[]
    for view_name ,u ,v ,depth in views :
        panels .append (
        _draw_projection (
        sample ,
        cp_points ,
        cp_dirs ,
        np .asarray (u ,dtype =float ),
        np .asarray (v ,dtype =float ),
        np .asarray (depth ,dtype =float ),
        f"{title } | {view_name }",
        )
        )
    sheet =Image .new ("RGB",(1120 ,1260 ),(235 ,238 ,242 ))
    for i ,panel in enumerate (panels ):
        sheet .paste (panel ,((i %2 )*560 ,(i //2 )*420 ))
    sheet .save (path ,optimize =True )


def render_contact_sheet (entries :list [dict ],output_path :Path )->None :
    thumb_w ,thumb_h =560 ,630 
    margin =20 
    canvas =Image .new ("RGB",(thumb_w *2 +margin *3 ,thumb_h *3 +margin *4 ),(230 ,233 ,238 ))
    draw =ImageDraw .Draw (canvas )
    font =ImageFont .load_default ()
    for i ,entry in enumerate (entries ):
        src =Path (entry ["files"]["qa_png"])
        with Image .open (src )as img :
            thumb =img .convert ("RGB")
            thumb .thumbnail ((thumb_w ,thumb_h -24 ),Image .Resampling .LANCZOS )
            x =margin +(i %2 )*(thumb_w +margin )
            y =margin +(i //2 )*(thumb_h +margin )
            draw .text ((x ,y ),f"{entry ['part_nr']} | {entry ['product_name']}",fill =(20 ,25 ,32 ),font =font )
            canvas .paste (thumb ,(x ,y +22 ))
    canvas .save (output_path ,optimize =True )


def validate_glb (path :Path )->None :
    with path .open ("rb")as fh :
        header =fh .read (12 )
    magic ,version ,length =struct .unpack ("<III",header )
    if magic !=0x46546C67 or version !=2 or length !=path .stat ().st_size :
        raise ValueError (f"invalid GLB header: {path }")


def make_readme (output_dir :Path ,entries :list [dict ])->None :
    lines =[
    "# WSCAD STEP - CP labeled review set (5 parts)",
    "",
    "This set uses STEP files from `code/_cad_eval_pxc` as geometry and the exact-catalog",
    "Phoenix Contact JSON files from `Desktop/JSON` as connection-point labels.",
    "Labels were rigidly transferred into each STEP frame and passed automatic geometry QA.",
    "",
    "**Status:** manufacturer-derived, AI-assisted labels. They are suitable for review and",
    "training experiments, but must not be reported as human ground truth until a person",
    "opens the QA PNG/GLB and signs them off.",
    "",
    "Red circles in the PNG mark CP locations; green arrows show outward insert direction.",
    "The GLB uses green spheres/arrows ten the complete gray STEP tessellation.",
    "",
    "| Part | Product | CP | Align residual (mm) | Max CP-nearest-vertex (mm) |",
    "|---|---|---:|---:|---:|",
    ]
    for entry in entries :
        lines .append (
        f"| {entry ['part_nr']} | {entry ['product_name']} | {entry ['cp_count']} | "
        f"{entry ['alignment']['mean_residual_mm']:.4f} | "
        f"{max (entry ['qa']['nearest_vertex_distance_mm']):.4f} |"
        )
    lines .extend (
    [
    "",
    "Each part folder contains:",
    "",
    "- `.stp`: unchanged source STEP geometry",
    "- `.cp_labels.json`: compact CP labels, directions, provenance, and alignment",
    "- `.labeled.json`: full STEP tessellation plus `ConnectionPoints` for training",
    "- `.QA.png`: six orthographic review views",
    "- `.QA.glb`: interactive 3D mesh with CP spheres and direction arrows",
    "- `.label_card.md`: per-part numeric review card",
    "",
    "`manifest.json` contains hashes and all automatic validation results.",
    "",
    ]
    )
    (output_dir /"README.md").write_text ("\n".join (lines ),encoding ="utf-8")


def process_part (spec :dict ,teacher_dir :Path ,output_dir :Path )->dict :
    part_nr =spec ["part_nr"]
    print (f"[{part_nr }] loading exact-catalog teacher JSON",flush =True )
    teacher_path =teacher_dir /f"{part_nr }.json"
    step_path =STEP_DIR /spec ["step_file"]
    if not teacher_path .is_file ():
        raise FileNotFoundError (teacher_path )
    if not step_path .is_file ():
        raise FileNotFoundError (step_path )
    if not _step_signature_ok (step_path ):
        raise ValueError (f"STEP signature failed: {step_path }")

    teacher =json_dataset .load_part_file (teacher_path )
    if teacher .part_nr !=part_nr :
        raise ValueError (f"teacher PartNr mismatch: {teacher .part_nr } != {part_nr }")
    if teacher .n_cps !=spec ["official_connection_count"]:
        raise ValueError (
        f"{part_nr }: JSON CP count {teacher .n_cps } != official count "
        f"{spec ['official_connection_count']}"
        )
    if len (teacher .cp_points )!=len (np .unique (np .round (teacher .cp_points ,6 ),axis =0 )):
        raise ValueError (f"{part_nr }: coincident CP labels require manual review")

    print (f"[{part_nr }] tessellating full STEP at {DEFL_MM } mm (no vertex cap)",flush =True )
    vertices ,faces =step_to_json .load_any_mesh (str (step_path ),deflection =DEFL_MM )
    vertices =np .asarray (vertices ,dtype =np .float64 )
    faces =np .asarray (faces ,dtype =np .int64 )
    if len (vertices )==0 or len (faces )==0 :
        raise ValueError (f"{part_nr }: blank STEP geometry")
    if not np .isfinite (vertices ).all ():
        raise ValueError (f"{part_nr }: non-finite STEP vertices")
    if faces .min ()<0 or faces .max ()>=len (vertices ):
        raise ValueError (f"{part_nr }: face indices outside vertex array")
    faces ,removed_faces =_remove_bad_faces (vertices ,faces )
    if len (faces )==0 :
        raise ValueError (f"{part_nr }: no non-degenerate faces")

    print (f"[{part_nr }] aligning JSON labels into STEP frame",flush =True )
    rotation ,translation ,residual ,orientation_selection =_align_with_direction_constraint (
    vertices ,
    teacher .vertices ,
    teacher .cp_directions ,
    spec .get ("step_direction_constraint"),
    sample =min (1000 ,len (teacher .vertices )),
    seed =0 ,
    )
    if residual >MAX_ALIGNMENT_RESIDUAL_MM :
        raise ValueError (
        f"{part_nr }: alignment residual {residual :.4f} mm exceeds "
        f"{MAX_ALIGNMENT_RESIDUAL_MM :.1f} mm"
        )
    if not np .allclose (rotation .T @rotation ,np .eye (3 ),atol =1e-8 ):
        raise ValueError (f"{part_nr }: alignment rotation is not orthonormal")
    if not np .isclose (np .linalg .det (rotation ),1.0 ,atol =1e-8 ):
        raise ValueError (f"{part_nr }: alignment rotation determinant is not +1")

    cp_points =(teacher .cp_points -translation )@rotation 
    cp_dirs =teacher .cp_directions @rotation 
    norms =np .linalg .norm (cp_dirs ,axis =1 ,keepdims =True )
    if np .any (norms <1e-9 ):
        raise ValueError (f"{part_nr }: zero CP direction")
    cp_dirs =cp_dirs /norms 
    roundtrip =cp_points @rotation .T +translation 
    roundtrip_error =np .linalg .norm (roundtrip -teacher .cp_points ,axis =1 )
    if float (roundtrip_error .max ())>=1e-6 :
        raise ValueError (f"{part_nr }: CP transform round-trip failed")

    from scipy .spatial import cKDTree 

    nearest_vertex_distance ,nearest_vertex_index =cKDTree (vertices ).query (cp_points ,k =1 )
    if float (nearest_vertex_distance .max ())>MAX_NEAREST_VERTEX_MM :
        raise ValueError (
        f"{part_nr }: CP nearest-vertex distance {nearest_vertex_distance .max ():.4f} mm "
        f"exceeds {MAX_NEAREST_VERTEX_MM :.1f} mm"
        )

    part_dir =output_dir /part_nr 
    part_dir .mkdir (parents =True ,exist_ok =True )
    copied_step =part_dir /f"{part_nr }.stp"
    shutil .copy2 (step_path ,copied_step )
    if sha256 (copied_step )!=sha256 (step_path ):
        raise ValueError (f"{part_nr }: copied STEP hash mismatch")

    cp_records =_cp_dicts (cp_points ,cp_dirs ,teacher .cp_names )
    compact_labels ={
    "schema_version":"wscad-step-cp-labels/1.0",
    "part_nr":part_nr ,
    "catalog":spec ["catalog"],
    "product_name":spec ["product_name"],
    "terminal_type":spec ["terminal_type"],
    "coordinate_frame":"copied STEP file",
    "units":"mm",
    "connection_points":cp_records ,
    "provenance":{
    "geometry_source":str (step_path ),
    "label_source":str (teacher_path ),
    "label_source_kind":"exact-catalog manufacturer JSON",
    "official_product_url":spec ["official_url"],
    "official_connection_count":spec ["official_connection_count"],
    "transfer_method":"cad_eval.align_frames signed-axis rigid alignment",
    "human_gt":False ,
    "human_review_required":True ,
    },
    "alignment":{
    "equation":"x_json = R @ x_step + t",
    "rotation_step_to_json":rotation .tolist (),
    "translation_step_to_json_mm":translation .tolist (),
    "mean_residual_mm":float (residual ),
    "max_roundtrip_error_mm":float (roundtrip_error .max ()),
    "orientation_selection":orientation_selection ,
    },
    "automatic_qa":{
    "status":"passed",
    "step_signature":True ,
    "full_tessellation_no_vertex_cap":True ,
    "finite_vertices":True ,
    "valid_face_indices":True ,
    "removed_degenerate_or_zero_area_faces":removed_faces ,
    "cp_count_matches_official":True ,
    "direction_norms":np .linalg .norm (cp_dirs ,axis =1 ).tolist (),
    "nearest_vertex_distance_mm":nearest_vertex_distance .tolist (),
    "nearest_vertex_index":nearest_vertex_index .astype (int ).tolist (),
    },
    "source_frame_labels":{
    "points_mm":teacher .cp_points .tolist (),
    "directions":teacher .cp_directions .tolist (),
    "names":list (teacher .cp_names ),
    },
    }
    label_path =part_dir /f"{part_nr }.cp_labels.json"
    write_json (label_path ,compact_labels )

    print (f"[{part_nr }] writing full labeled training JSON",flush =True )
    labeled_obj =_abb_mesh (part_nr ,vertices ,faces ,cp_records )
    labeled_path =part_dir /f"{part_nr }.labeled.json"
    write_json (labeled_path ,labeled_obj ,compact =True )
    parsed =json_dataset .load_part_file (labeled_path )
    if len (parsed .vertices )!=len (vertices )or len (parsed .faces )!=len (faces )or parsed .n_cps !=len (cp_points ):
        raise ValueError (f"{part_nr }: labeled JSON read-back mismatch")

    qa_cps =[
    {"entry_point":point .tolist (),"approach_vector":direction .tolist ()}
    for point ,direction in zip (cp_points ,cp_dirs )
    ]
    qa_glb =part_dir /f"{part_nr }.QA.glb"
    glb =export_glb ._GlbBuilder ()
    export_glb .build_scene (glb ,vertices ,faces ,ml_cps =[],cad_cps =qa_cps )
    glb .write (str (qa_glb ))
    validate_glb (qa_glb )

    qa_png =part_dir /f"{part_nr }.QA.png"
    render_qa_png (qa_png ,f"{part_nr } | {spec ['product_name']}",vertices ,faces ,cp_points ,cp_dirs )

    label_card =part_dir /f"{part_nr }.label_card.md"
    card_lines =[
    f"# {part_nr } - {spec ['product_name']}",
    "",
    f"- Product: {spec ['official_url']}",
    f"- Type: {spec ['terminal_type']}",
    f"- Official/JSON connection count: {spec ['official_connection_count']} / {len (cp_points )}",
    f"- Alignment residual: {residual :.6f} mm",
    f"- Full STEP mesh: {len (vertices )} vertices, {len (faces )} faces",
    "- Human-GT status: pending human visual sign-off",
    "",
    "| CP | Source name | X (mm) | Y (mm) | Z (mm) | Dir X | Dir Y | Dir Z | Nearest vertex (mm) |",
    "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i ,(point ,direction ,name ,distance )in enumerate (
    zip (cp_points ,cp_dirs ,teacher .cp_names ,nearest_vertex_distance ),start =1 
    ):
        card_lines .append (
        f"| {i } | {name or '(blank)'} | {point [0 ]:.6f} | {point [1 ]:.6f} | {point [2 ]:.6f} | "
        f"{direction [0 ]:.6f} | {direction [1 ]:.6f} | {direction [2 ]:.6f} | {distance :.6f} |"
        )
    label_card .write_text ("\n".join (card_lines )+"\n",encoding ="utf-8")

    entry ={
    "part_nr":part_nr ,
    "catalog":spec ["catalog"],
    "product_name":spec ["product_name"],
    "terminal_type":spec ["terminal_type"],
    "official_product_url":spec ["official_url"],
    "official_connection_count":spec ["official_connection_count"],
    "cp_count":len (cp_points ),
    "mesh":{
    "vertices":len (vertices ),
    "faces":len (faces ),
    "removed_degenerate_or_zero_area_faces":removed_faces ,
    "bbox_dimension_mm":np .ptp (vertices ,axis =0 ).tolist (),
    },
    "alignment":{
    "mean_residual_mm":float (residual ),
    "limit_mm":MAX_ALIGNMENT_RESIDUAL_MM ,
    "max_roundtrip_error_mm":float (roundtrip_error .max ()),
    "rotation_step_to_json":rotation .tolist (),
    "translation_step_to_json_mm":translation .tolist (),
    "orientation_selection":orientation_selection ,
    },
    "qa":{
    "status":"automatic_pass_human_signoff_pending",
    "step_signature":True ,
    "copied_step_hash_matches":True ,
    "full_tessellation_no_vertex_cap":True ,
    "labeled_json_readback":True ,
    "glb_header":True ,
    "direction_norms":np .linalg .norm (cp_dirs ,axis =1 ).tolist (),
    "nearest_vertex_distance_mm":nearest_vertex_distance .tolist (),
    "nearest_vertex_limit_mm":MAX_NEAREST_VERTEX_MM ,
    },
    "hashes":{
    "source_step_sha256":sha256 (step_path ),
    "copied_step_sha256":sha256 (copied_step ),
    "teacher_json_sha256":sha256 (teacher_path ),
    "labels_sha256":sha256 (label_path ),
    "labeled_json_sha256":sha256 (labeled_path ),
    "qa_glb_sha256":sha256 (qa_glb ),
    "qa_png_sha256":sha256 (qa_png ),
    },
    "files":{
    "step":str (copied_step ),
    "labels":str (label_path ),
    "labeled_json":str (labeled_path ),
    "qa_glb":str (qa_glb ),
    "qa_png":str (qa_png ),
    "label_card":str (label_card ),
    },
    }
    print (
    f"[{part_nr }] PASS: CP={len (cp_points )}, V={len (vertices )}, F={len (faces )}, "
    f"align={residual :.4f} mm, nearest={nearest_vertex_distance .max ():.4f} mm",
    flush =True ,
    )
    return entry 


def main ()->None :
    parser =argparse .ArgumentParser (description =__doc__ )
    parser .add_argument ("--teacher-dir",type =Path ,default =DEFAULT_TEACHER_DIR )
    parser .add_argument ("--out",type =Path ,default =DEFAULT_OUTPUT_DIR )
    args =parser .parse_args ()

    output_dir =args .out .resolve ()
    output_dir .mkdir (parents =True ,exist_ok =True )
    entries =[]
    for spec in PARTS :
        entries .append (process_part (spec ,args .teacher_dir .resolve (),output_dir ))

    step_hashes =[entry ["hashes"]["source_step_sha256"]for entry in entries ]
    if len (step_hashes )!=len (set (step_hashes )):
        raise ValueError ("duplicate STEP hashes in selected five-part set")
    if len (entries )!=5 :
        raise ValueError (f"expected five completed parts, got {len (entries )}")

    contact_sheet =output_dir /"QA_CONTACT_SHEET.png"
    render_contact_sheet (entries ,contact_sheet )
    manifest ={
    "schema_version":"wscad-step-cp-review-manifest/1.0",
    "status":"complete_automatic_qa_passed_human_signoff_pending",
    "scope":"five exact-catalog WSCAD STEP terminal parts",
    "geometry_source":str (STEP_DIR ),
    "label_source":str (args .teacher_dir .resolve ()),
    "output":str (output_dir ),
    "tessellation_deflection_mm":DEFL_MM ,
    "vertex_cap":None ,
    "label_semantics":"connection entry point plus outward insert direction",
    "human_gt":False ,
    "human_review_required":True ,
    "selection":"five geometrically and functionally diverse Phoenix Contact terminal parts",
    "global_qa":{
    "completed_parts":len (entries ),
    "expected_parts":5 ,
    "total_connection_points":sum (entry ["cp_count"]for entry in entries ),
    "all_step_hashes_unique":True ,
    "all_official_counts_match":all (
    entry ["cp_count"]==entry ["official_connection_count"]for entry in entries 
    ),
    "all_alignment_residuals_within_limit":all (
    entry ["alignment"]["mean_residual_mm"]<=MAX_ALIGNMENT_RESIDUAL_MM 
    for entry in entries 
    ),
    "all_cp_nearest_vertex_distances_within_limit":all (
    max (entry ["qa"]["nearest_vertex_distance_mm"])<=MAX_NEAREST_VERTEX_MM 
    for entry in entries 
    ),
    },
    "contact_sheet":str (contact_sheet ),
    "contact_sheet_sha256":sha256 (contact_sheet ),
    "parts":entries ,
    }
    write_json (output_dir /"manifest.json",manifest )
    make_readme (output_dir ,entries )
    print (f"COMPLETE -> {output_dir }",flush =True )


if __name__ =="__main__":
    main ()
