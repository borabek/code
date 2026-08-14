# -*- coding: utf-8 -*-
"""Build TRAINING labels from MANUFACTURER ConnectionPoints (no human labelling time).

WHY THIS IS THE STRONGEST REMAINING LEVER (evidence, 2026-07-22):
  * The 379-part manufacturer arbiter says the product is F1 0.752 ten PXC but **0.217 ten Weidmueller**
    (recall 0.138; it emits ZERO CPs ten 51% of WEI parts, vs 3% of PXC parts). That is a genuine
    cross-manufacturer domain gap: even before any confidence masking the model marks 2.9% of a WEI
    part's vertices as connection vs 9.0% of a PXC part's, and its top connection probability is 0.82
    vs 0.99. Lowering vertex_conf 0.7 -> 0.3 only lifts WEI to 0.319 -- post-processing cannot fix it,
    TRAINING DATA can.
  * Human partial labels have a built-in flaw: openings the annotator did NOT mark are supervised as
    NEGATIVES (train_seg_extra's masked BCE pushes p_pos down ten every unmarked vertex). Manufacturer
    ConnectionPoints are a COMPLETE list per part, so this corpus has no such false negatives -- which
    is very likely why going 77 -> 103 human parts did not help.

HOW: the manufacturer's point sits at the CONTACT, ~15mm INSIDE the opening (measured: tangential
offset from our prediction only 1.4mm, axial +14.8mm). So walk OUTWARD along its InsertDirection until
leaving the solid -> that exit point is the opening MOUTH = the thesis v_o = what we label. Then mark
the mesh vertices around it that face the same way and lie within an opening-sized radius.

SPLIT DISCIPLINE: parts are assigned train/test by hash of part id, per manufacturer, so WEI parts
used for training are never scored. Test parts are written nowhere.

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe labels_from_mfg.py \
         [--out _mfg_labels] [--test-frac 0.3] [--limit N] [--mfg WEI]
"""
import os ,sys ,glob ,json ,time ,hashlib ,argparse 
import numpy as np 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import thesis_remesh ,connector3d 
from cad_eval import align_frames 
from infer_step_cp import step_to_mesh 

CE =int (connector3d .CABLE_ENTRY )
DS ="_ds1/DataSet"


def save_obj (path ,V ,F ):
    with open (path ,"w")as f :
        for v in V :f .write (f"v {v [0 ]:.6f} {v [1 ]:.6f} {v [2 ]:.6f}\n")
        for t in F :f .write (f"f {t [0 ]+1 } {t [1 ]+1 } {t [2 ]+1 }\n")


def _ray_hits (V ,F ,o ,d ,eps =1e-9 ):
    """Moller-Trumbore, vectorised over all triangles: distances t>0 where ray that+t*d hits the mesh."""
    v0 =V [F [:,0 ]];e1 =V [F [:,1 ]]-v0 ;e2 =V [F [:,2 ]]-v0 
    pv =np .cross (d ,e2 );det =(e1 *pv ).sum (1 )
    ok =np .abs (det )>eps 
    inv =np .zeros_like (det );inv [ok ]=1.0 /det [ok ]
    tv =o -v0 
    u =(tv *pv ).sum (1 )*inv 
    qv =np .cross (tv ,e1 )
    v =(d *qv ).sum (1 )*inv 
    t =(e2 *qv ).sum (1 )*inv 
    hit =ok &(u >=-1e-6 )&(v >=-1e-6 )&(u +v <=1 +1e-6 )&(t >1e-4 )
    return t [hit ]


def mouth_from_contact (V ,c ,d ,F =None ,max_walk =60.0 ,step =0.4 ,tube =4.0 ,need =4 ):
    """The opening MOUTH = the point ten the insertion axis where the surrounding MATERIAL ENDS.

    Two earlier attempts failed the 5mm check against human-marked openings:
      * walk to the farthest vertex in a tube  -> overshoots to the part's outer extremity (median 8.8mm)
      * ray-cast, first triangle hit           -> median 10.0mm, 1/19 within 5mm.
        WHY ray-casting is wrong here: the wire opening is a HOLE, so a ray leaving the contact along
        the insertion axis passes straight THROUGH it without hitting anything; the first hit is an
        internal wall or a far-side face, never the mouth.
    What actually defines the mouth is the RIM: walking outward, the terminal's material surrounds the
    axis until the opening ends. So sample along the axis and take the LAST station that still has
    material around it (>= `need` vertices within `tube` mm perpendicular, near that station)."""
    d =np .asarray (d ,float );d =d /(np .linalg .norm (d )+1e-9 )
    V =np .asarray (V ,float )
    rel =V -c 
    along =rel @d 
    perp =np .linalg .norm (rel -along [:,None ]*d ,axis =1 )
    ring =perp <tube # a tube of material around the insertion axis
    if not ring .any ():
        return None 
    a_r =along [ring ]
    last =None 
    t =0.0 
    while t <=max_walk :
        n =int (np .count_nonzero (np .abs (a_r -t )<=step *2.0 ))
        if n >=need :
            last =t # material still surrounds the axis here
        elif last is not None and t -last >3.0 :
            break # material ended 3mm ago -> we are outside
        t +=step 
    if last is None :
        return None 
    return c +last *d 


def _adjacency (F ,n ):
    nb =[[]for _ in range (n )]
    for t in F :
        for i in range (3 ):
            nb [t [i ]].append (t [(i +1 )%3 ]);nb [t [i ]].append (t [(i +2 )%3 ])
    return [np .unique (x )for x in nb ]


def _cavity (V ,nb ,vn ,smooth =2 ):
    """Per-vertex concavity in [0,1]: 1 = deep recess, 0 = convex/flat.

    Same measure the annotator's own tool (label_tool.html computeCavity) shades with, so a generated
    label is grown over the SAME notion of "inside an opening" the human was looking at: average of
    dot(neighbour_direction, own_normal) -- positive when the neighbourhood sits IN FRONT of the
    surface, i.e. the vertex is inside a hollow. Smoothed so single-triangle noise does not speckle.
    """
    cv =np .zeros (len (V ))
    for i ,js in enumerate (nb ):
        if not len (js ):continue 
        d =V [js ]-V [i ]
        L =np .linalg .norm (d ,axis =1 )
        m =L >1e-9 
        if not m .any ():continue 
        cv [i ]=float (((d [m ]/L [m ,None ])@vn [i ]).mean ())
    for _ in range (smooth ):
        cv =np .array ([np .mean ([cv [i ]]+[cv [j ]for j in nb [i ]])if len (nb [i ])else cv [i ]
        for i in range (len (V ))])
    lo ,hi =cv .min (),cv .max ()
    return (cv -lo )/(hi -lo if hi >lo else 1.0 )


def grow_opening (V ,F ,vn ,nb ,cav ,mouth ,d ,max_mm =9.0 ,min_v =25 ,cav_drop =0.25 ):
    """Grow the label INTO the recess from the mouth, along the surface.

    Two earlier versions failed the IoU gate against human labels (0.05): a 6mm sphere paints a round
    patch, and growing over "outward-facing" vertices paints the FLAT FACE AROUND the hole. The human
    marks the hole ITSELF -- the recess walls. So the growth criterion is CONCAVITY: start at the
    mouth and keep vertices that are at least as hollow as the mouth region (minus cav_drop), which
    walks down into the opening and stops when it climbs back out onto the housing.
    """
    d =np .asarray (d ,float );d =d /(np .linalg .norm (d )+1e-9 )
    dist =np .linalg .norm (V -mouth ,axis =1 )
    seed =int (dist .argmin ())
    if dist [seed ]>4.0 :
        return None 
        # reference hollowness: the most concave vertex within a small ball of the mouth
    near =dist <=3.0 
    ref =float (cav [near ].max ())if near .any ()else float (cav [seed ])
    thr =ref -cav_drop 
    seen ={seed };stack =[seed ];out =[seed ]
    while stack :
        i =stack .pop ()
        for j in nb [i ]:
            j =int (j )
            if j in seen :continue 
            seen .add (j )
            if dist [j ]<=max_mm and cav [j ]>=thr :
                out .append (j );stack .append (j )
    return np .array (out ,int )if len (out )>=min_v else None 


def label_part (V ,F ,mouths ,dirs ,radius_mm ):
    """Per-vertex label from the manufacturer's CPs, grown into each opening (see grow_opening)."""
    L =np .zeros (len (V ),np .int64 )
    tri =V [F ]
    n =np .cross (tri [:,1 ]-tri [:,0 ],tri [:,2 ]-tri [:,0 ])
    n =n /(np .linalg .norm (n ,axis =1 ,keepdims =True )+1e-9 )
    vn =np .zeros_like (V )
    for k in range (3 ):
        np .add .at (vn ,F [:,k ],n )
    vn =vn /(np .linalg .norm (vn ,axis =1 ,keepdims =True )+1e-9 )
    nb =_adjacency (F ,len (V ))
    cav =_cavity (V ,nb ,vn )
    n_ok =0 
    for m ,d in zip (mouths ,dirs ):
        if m is None :continue 
        idx =grow_opening (V ,F ,vn ,nb ,cav ,m ,d ,max_mm =radius_mm +3.0 )
        if idx is None :continue # SKIP -- never approximate
        L [idx ]=CE ;n_ok +=1 
    return L if n_ok else None 


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--out",default ="_mfg_labels")
    ap .add_argument ("--test-frac",type =float ,default =0.3 )
    ap .add_argument ("--limit",type =int ,default =0 )
    ap .add_argument ("--mfg",default ="",help ="restrict to one manufacturer")
    ap .add_argument ("--radius-mm",type =float ,default =6.0 )
    ap .add_argument ("--target",type =int ,default =6000 )
    ap .add_argument ("--list",default ="",help ="sadece bu dosyadaki part no'lari (satir/bosluk ayrik)")
    a =ap .parse_args ()

    step ={os .path .basename (s ).split ("_")[1 ]:s for s in glob .glob ("all_wscad_stp/*.stp")}
    seen =set ()
    for d in ("_label_targets","_label_targets_2","_label_targets_3","_label_targets_4"):
        seen |={os .path .basename (os .path .normpath (p ))for p in glob .glob (d +"/*/")}
    import scheffler_dataset as ds 
    for sp in ("train","val"):
        seen |={s ["part_id"]for s in ds .load_split ("wscad_corpus_scheffler_exact",sp ,verify_hashes =False )}

    os .makedirs (a .out ,exist_ok =True )
    split ={}
    n_ok =n_skip =0 ;t0 =time .time ();n_cp =0 
    want =set (open (a .list ).read ().split ())if a .list else None 
    files =sorted (glob .glob (os .path .join (DS ,"*ElectricalTerminal*.json")))
    for f in files :
        head =os .path .basename (f ).split ("_")[0 ]
        mfg ,pid =(head .split (".",1 )+[""])[:2 ]
        if pid not in step or pid in seen :continue 
        if want is not None and pid not in want :continue 
        if a .mfg and mfg !=a .mfg :continue 
        # deterministic per-part split so a training part is NEVER scored
        h =int (hashlib .md5 (pid .encode ()).hexdigest ()[:8 ],16 )/0xFFFFFFFF 
        is_test =h <a .test_frac 
        try :
            j =json .load (open (f ,encoding ="utf-8-sig"))
            G =np .array ([[c ["Point"]["X"],c ["Point"]["Y"],c ["Point"]["Z"]]
            for c in j .get ("ConnectionPoints",[])],float )
            Gd =np .array ([[c ["InsertDirection"]["X"],c ["InsertDirection"]["Y"],c ["InsertDirection"]["Z"]]
            for c in j .get ("ConnectionPoints",[])],float )
            if not len (G ):continue 
            Vj =np .array ([[p ["X"],p ["Y"],p ["Z"]]for p in j ["Graphic3d"]["Points"]],float )
            Vr ,Fr =step_to_mesh (step [pid ])
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =a .target )
            V =np .ascontiguousarray (V ,float );F =np .ascontiguousarray (F ,np .int64 )
            R ,t ,_ =align_frames (Vr ,Vj )# STEP -> JSON
            Gs =(G -t )@R # manufacturer CPs INTO the mesh frame
            Gds =Gd @R 
            mouths =[mouth_from_contact (V ,Gs [i ],Gds [i ],F )for i in range (len (Gs ))]
            if all (m is None for m in mouths ):
                n_skip +=1 ;continue 
            L =label_part (V ,F ,mouths ,Gds ,a .radius_mm )
            if L is None or (L ==CE ).sum ()<20 :
                n_skip +=1 ;continue 
            d =os .path .join (a .out ,"test"if is_test else "train",pid )
            os .makedirs (d ,exist_ok =True )
            save_obj (os .path .join (d ,f"{pid }.obj"),V ,F )
            open (os .path .join (d ,f"{pid }.labels.txt"),"w").write ("\n".join (map (str ,L .tolist ())))
            json .dump ({"part_id":pid ,"mfg":mfg ,"n_mfg_cps":len (G ),
            "n_mouths":sum (m is not None for m in mouths ),
            "ce_vertices":int ((L ==CE ).sum ()),"source":"manufacturer ConnectionPoints"},
            open (os .path .join (d ,f"{pid }.provenance.json"),"w"),indent =1 )
            split [pid ]={"mfg":mfg ,"split":"test"if is_test else "train"}
            n_ok +=1 ;n_cp +=len (G )
            if n_ok %25 ==0 :print (f"  {n_ok } part ({time .time ()-t0 :.0f}s)",flush =True )
            if a .limit and n_ok >=a .limit :break 
        except Exception as e :
            n_skip +=1 
    json .dump (split ,open (os .path .join (a .out ,"split.json"),"w"),indent =1 )
    tr =sum (1 for v in split .values ()if v ["split"]=="train")
    from collections import Counter 
    print (f"\n{n_ok } part etiketlendi ({n_cp } manufacturer CP), {n_skip } atlandi, {time .time ()-t0 :.0f}s")
    print (f"  train {tr } | test {n_ok -tr }")
    print (f"  manufacturer dagilimi: {dict (Counter (v ['mfg']for v in split .values ()))}")
    print (f"  -> {a .out }/")


if __name__ =="__main__":
    main ()
