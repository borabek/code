# Extract connection openings DIRECTLY from a STEP B-rep -- NO mesh tessellation
# loss, NO ML -- using gmsh's CAD kernel so the coordinates match the tessellated
# mesh exactly (raw STEP placements are in local frames; gmsh applies the
# assembly transform, giving cylinders in the same frame as step_to_json's mesh).
#
# On these connectors the circular openings are real CAD features
# (CYLINDRICAL_SURFACE). gmsh reports each surface's type, centre of mass and
# bounding box; cylinders of terminal radius -> connection points (centre +
# radius). A hole is several cylindrical faces, so coincident ones are merged.
#
# This is the cleanest "real label" source when the STEP is available.
#
# Usage:
#   python step_openings.py part.stp --out openings.json
#   python step_openings.py part.stp --out that.json --rmin 1.3 --rmax 3.0
#   python step_openings.py "3d models" --out that.json
#   python step_openings.py part.stp --list            # radius histogram only

import os 
import json 
import glob 
import argparse 
import logging 

import numpy as np 

# Label-time face-merge ceiling. Kept FINER than the decoder's NMS ceiling ten
# purpose (4.0 vs 5.0mm) -- coarsening labels to match the decoder would merge
# 37 pairs of REAL terminals (measured ten the human GT, 8 parts with >100mm
# bodies) into single points. See the long note in cp_targets.
from cp_targets import LABEL_MERGE_CEILING_MM as MERGE_CEILING_MM ,merge_radius_mm 

logger =logging .getLogger (__name__ )


def _cyl_radius (ext ):
    """Radius of an axis-aligned cylinder from its bbox extents: the two CLOSEST
    extents are the two diameters (2r); the third is the height."""
    s =np .sort (np .asarray (ext ,float ))
    pair =(s [0 ],s [1 ])if (s [1 ]-s [0 ])<=(s [2 ]-s [1 ])else (s [1 ],s [2 ])
    return float ((pair [0 ]+pair [1 ])/4.0 )


def _cyl_axis_dim (ext ):
    """Bbox dimension along the cylinder axis = the 'odd-one-out' extent (height);
    the two close extents are the diameter."""
    o =np .argsort (np .asarray (ext ,float ))
    s =np .asarray (ext ,float )[o ]
    return int (o [2 ])if (s [1 ]-s [0 ])<=(s [2 ]-s [1 ])else int (o [0 ])


def _cyl_ambiguous (ext ,rel_tol =0.15 ):
    """True when _cyl_radius/_cyl_axis_dim's "closest pair = diameter" assumption
    is a poor fit: for a genuine axis-aligned cylinder two of the three bbox
    extents should be near-IDENTICAL (both = 2r) regardless of whether the axis
    (height) is longer or shorter than the diameter. If even the closest pair
    differs by more than rel_tol of their own scale, none of the three extents
    look like a clean diameter pair (e.g. a short/wide hole where height and
    diameter are close enough to confuse the heuristic) -- the picked radius/
    axis may be wrong. Used only to WARN; callers still get a best-effort value."""
    s =np .sort (np .asarray (ext ,float ))
    gap1 ,gap2 =s [1 ]-s [0 ],s [2 ]-s [1 ]
    a ,b =(s [0 ],s [1 ])if gap1 <=gap2 else (s [1 ],s [2 ])
    scale =max (abs (a ),abs (b ),1e-9 )
    return abs (a -b )>rel_tol *scale 


def auto_radius_range (radii ,fillet_floor =1.3 ,split_gap =1.0 ):
    """Auto-pick (rmin, rmax) for the terminal holes, so a new connector family
    needs no manual --rmin/--rmax. Edge fillets/rounds cluster below ~1 mm and are
    dropped (fillet_floor). A connector often has SEVERAL terminal hole sizes plus
    a few much larger mounting/structural holes; the terminal cluster is separated
    from those by a big jump in radius, so we keep everything from the floor up to
    the radius just below the largest gap (> split_gap mm). With no clear gap, keep
    all holes above the floor. Falls back to 1.0-5.0 mm if nothing is above it."""
    r =sorted (set (round (float (x ),1 )for x in radii if x >=fillet_floor ))
    if not r :
        return 1.0 ,5.0 
    if len (r )==1 :
        return max (0.0 ,r [0 ]-0.3 ),r [0 ]+0.3 
    gaps =[(r [i +1 ]-r [i ],i )for i in range (len (r )-1 )]
    biggest ,k =max (gaps )
    rmax =(r [k ]if biggest >split_gap else r [-1 ])+0.3 
    return fillet_floor -0.1 ,rmax 


def auto_radius_bands (radii ,fillet_floor =1.3 ,split_gap =1.0 ,rcap =8.0 ,
min_count =2 ):
    """Multi-band variant of auto_radius_range: a terminal part can have TWO
    real entry-size classes at before (e.g. NEOZED 3048357: 2.6mm wire bores AND
    6.0mm fuse-screw cartridges -- the single-band auto cuts at the first gap
    and silently loses the whole screw class, RESULTS 11k #3). Cluster the
    radii (gap > split_gap starts a new cluster) and accept every cluster with
    >= min_count members whose radius is within [fillet_floor, rcap];
    singleton radii are still treated as structural/viewing holes and dropped.
    Returns a list of (lo, hi) bands."""
    counts ={}
    for x in radii :
        if fillet_floor <=x <=rcap :
            key =round (float (x ),1 )
            counts [key ]=counts .get (key ,0 )+1 
    r =sorted (counts )
    if not r :
        return [(1.0 ,5.0 )]
    clusters ,cur =[],[r [0 ]]
    for a ,b in zip (r ,r [1 :]):
        if b -a >split_gap :
            clusters .append (cur )
            cur =[]
        cur .append (b )
    clusters .append (cur )
    bands =[(c [0 ]-0.3 ,c [-1 ]+0.3 )for c in clusters 
    if sum (counts [x ]for x in c )>=min_count ]
    return bands or [(1.0 ,5.0 )]


def extract_cylinders (path ,include_cones =True ):
    """All cylindrical (and, by default, conical) faces of a STEP -> (cyls,
    part_bbox) in the gmsh (== tessellated-mesh) coordinate frame.
    cyls = [{center, radius, axis_dim, ext, kind}]; part_bbox = (xmin..zmax).

    include_cones: conical faces are funnel lead-ins above wire-entry bores ten
    many terminal blocks; treating them like cylinders (same bbox radius/axis
    heuristics, kind='cone') lets the existing merge fold a funnel into its
    bore's opening instead of losing the entry entirely ten parts where only the
    funnel (not the bore) survives as a distinct CAD face."""
    import gmsh 
    gmsh .initialize ()
    try :
        gmsh .option .setNumber ("General.Terminal",0 )
        gmsh .open (path )
        wanted =("Cylinder","Cone")if include_cones else ("Cylinder",)
        cyls =[]
        for (d ,t )in gmsh .model .getEntities (2 ):
            typ =gmsh .model .getType (d ,t )
            if typ not in wanted :
                continue 
            com =np .array (gmsh .model .occ .getCenterOfMass (d ,t ),dtype =float )
            bb =gmsh .model .getBoundingBox (d ,t )
            ext =np .array ([bb [3 ]-bb [0 ],bb [4 ]-bb [1 ],bb [5 ]-bb [2 ]])
            if typ =="Cylinder"and _cyl_ambiguous (ext ):
                logger .warning (
                "%s: cylindrical face at %s has bbox extents %s with no clean "
                "diameter pair (radius/axis heuristic may be wrong for this "
                "short/wide or otherwise unusual hole)",
                os .path .basename (path ),com .round (1 ).tolist (),ext .round (2 ).tolist ())
            cyls .append ({"center":com ,"radius":_cyl_radius (ext ),
            "axis_dim":_cyl_axis_dim (ext ),"ext":ext ,
            "kind":"cone"if typ =="Cone"else "cyl"})
        bbox =np .array (gmsh .model .getBoundingBox (-1 ,-1 ),dtype =float )
        return cyls ,bbox 
    finally :
        gmsh .finalize ()


def extract_planar_faces (path ):
    """All PLANAR faces of a STEP -> ([{center, ext}], part_bbox) in the gmsh
    frame. Used by the planar-pocket rect detector (rectangular push-in wire
    slots have no cylindrical face -- only planar walls). Axis-aligned terminal
    blocks give axis-aligned wall planes, so the face normal is the thin bbox
    axis (argmin ext)."""
    import gmsh 
    gmsh .initialize ()
    try :
        gmsh .option .setNumber ("General.Terminal",0 )
        gmsh .open (path )
        planes =[]
        for (d ,t )in gmsh .model .getEntities (2 ):
            if gmsh .model .getType (d ,t )!="Plane":
                continue 
            com =np .array (gmsh .model .occ .getCenterOfMass (d ,t ),dtype =float )
            b =gmsh .model .getBoundingBox (d ,t )
            ext =np .array ([b [3 ]-b [0 ],b [4 ]-b [1 ],b [5 ]-b [2 ]])
            planes .append ({"center":com ,"ext":ext })
        bbox =np .array (gmsh .model .getBoundingBox (-1 ,-1 ),dtype =float )
        return planes ,bbox 
    finally :
        gmsh .finalize ()


def _planar_pocket_openings (planes ,bbox ,existing_ops ,merge_tol ,approach_dim =2 ):
    """Rectangular / push-in wire slots as PLANAR-WALL POCKETS -- the strong
    signal the corner-fillet _rect_openings lacks.

    A real push-in wire entry is a small rectangular CAVITY going into the part
    along the approach axis: bounded by ~4 small vertical planar walls (normal
    perpendicular to the approach axis) that enclose a slot-sized (1-8mm)
    rectangular footprint and have real depth (they are walls of a pocket, not
    cosmetic edge rounds). Requiring walls ten BOTH perpendicular axes (so they
    actually bound a rectangle, not a single flat lip) + a slot-sized footprint
    is what separates a genuine opening from housing planar faces. Verified ten
    PXC.3031238: the two GT wire entries are exactly such 4-wall pockets at z~23
    (~2.9x2.7mm footprint, ~9mm deep) that the cylinder path cannot see."""
    a =int (approach_dim )
    perp =[k for k in range (3 )if k !=a ]
    walls =[]
    for p in planes :
        ext =np .asarray (p ["ext"],float )
        ndim =int (np .argmin (ext ))
        if ndim ==a :# floor/ceiling face, not a side wall
            continue 
        depth =float (ext [a ])# extent ALONG the approach axis
        width =max (float (ext [k ])for k in perp if k !=ndim )# other in-plane span
        if not (3.0 <=depth <=20.0 ):# a real pocket wall, not a thin lip
            continue 
        if width >8.0 :# slot-sized wall, not the outer housing
            continue 
        walls .append ({"c":np .asarray (p ["center"],float ),"ndim":ndim ,
        "top":float (p ["center"][a ]+0.5 *depth )})
    n =len (walls )
    if n <3 :
        return []
    parent =list (range (n ))

    def find (x ):
        while parent [x ]!=x :
            parent [x ]=parent [parent [x ]];x =parent [x ]
        return x 

    for i in range (n ):# cluster walls of one pocket by
        for j in range (i +1 ,n ):# horizontal (perp-plane) proximity
            dh =float (np .hypot (*[walls [i ]["c"][k ]-walls [j ]["c"][k ]for k in perp ]))
            if dh <=6.0 :
                parent [find (i )]=find (j )
    groups ={}
    for i in range (n ):
        groups .setdefault (find (i ),[]).append (walls [i ])
    existing_pts =np .array ([o ["entry_point"]for o in existing_ops ],float )if existing_ops else np .zeros ((0 ,3 ))
    ops =[]
    for members in groups .values ():
    # must have walls facing ten BOTH perp axes -> they bound a rectangle
        if len (members )<3 or len (set (m ["ndim"]for m in members ))<2 :
            continue 
        C =np .array ([m ["c"]for m in members ])
        foot =[float (C [:,k ].max ()-C [:,k ].min ())for k in perp ]
        if not (min (foot )>=1.0 and max (foot )<=8.0 ):
            continue 
        c =C .mean (0 )
        c [a ]=max (m ["top"]for m in members )# opening sits at the pocket top
        if len (existing_pts )and np .linalg .norm (existing_pts -c ,axis =1 ).min ()<merge_tol :
            continue 
        if ops and min (np .linalg .norm (np .array ([o ["entry_point"]for o in ops ])-c ,
        axis =1 ))<merge_tol :
            continue # dedup against earlier pockets
        direction =np .zeros (3 )
        direction [a ]=_dir_sign (c ,a ,bbox )
        ops .append ({"entry_point":[float (x )for x in c ],
        "approach_vector":[float (x )for x in direction ],
        "radius_mm":float (0.5 *max (foot )),
        "n_faces":len (members ),"kind":"rect_pocket"})
    return ops 


def _dir_sign (c ,ad ,bbox ):
    """Insert-direction sign along axis `name` for an opening at centre `c`: a hole
    clearly nearer ONE face along its axis (blind hole/recess) is entered from
    that face; a symmetric through-hole points away from the bbox midpoint."""
    lo ,hi =bbox [ad ],bbox [ad +3 ]
    span =hi -lo 
    mid =0.5 *(lo +hi )
    d_lo ,d_hi =c [ad ]-lo ,hi -c [ad ]
    if span >0 and abs (d_hi -d_lo )>0.15 *span :
        return 1.0 if d_hi <d_lo else -1.0 # toward the nearer face
    return 1.0 if (c [ad ]-mid )>=0 else -1.0 


def validate_directions (ops ,V ,part_nr ="?"):
    """Flip any approach_vector that points INTO the part instead of out into
    free space, using the actual mesh -- the geometric ground truth the bbox
    heuristics in _dir_sign approximate.

    Check: walk a probe from the entry point a few mm along +dir and along
    -dir; the FREE-SPACE side (correct outward direction) is measurably farther
    from the mesh than the side that dives into the bore/housing. This is
    exactly the adjudication the bbox heuristic can't provide for a through-
    channel near the part's midpoint (measured: two wscad parts' matched CPs
    flipped 180 degrees between two labeling runs purely from heuristic
    ambiguity). Mutates ops in place; returns the number of flips."""
    if not ops or V is None or not len (V ):
        return 0 
    from scipy .spatial import cKDTree 
    tree =cKDTree (np .asarray (V ,float ))
    flips =0 
    for o in ops :
        c =np .asarray (o ["entry_point"],float )
        d =np .asarray (o ["approach_vector"],float )
        n =np .linalg .norm (d )
        if n <1e-8 :
            continue 
        d =d /n 
        probe =max (2.5 *float (o .get ("radius_mm",1.0 )),3.0 )
        d_out ,_ =tree .query (c +probe *d )
        d_in ,_ =tree .query (c -probe *d )
        # flip only ten a CLEAR verdict (20% relative margin) -- near-ties keep
        # the heuristic sign rather than thrashing ten tessellation noise
        if d_in >1.2 *d_out :
            o ["approach_vector"]=[float (-x )for x in d ]
            flips +=1 
    if flips :
        logger .info ("%s: mesh-clearance direction check flipped %d approach "
        "vector(s) that pointed into the part",part_nr ,flips )
    return flips 


def _pair_slot_halves (holes ,merge_tol ):
    """Fuse the two half-cylinder ends of a RACETRACK/oval wire slot into ONE
    opening. Measured ten real PXC push-in blocks (3211813/3211822): each wire
    entry is an oval slot whose CAD faces are two parallel half-cylinders of
    identical radius ~5.7mm apart (~2.1x r) -- farther than any sane merge_tol,
    so they surfaced as TWO offset detections, one scored TP and its twin FP,
    while the human GT sits exactly between them. Pairing rule: same axis, radii
    within 5%, centre distance in (merge_tol, 2.5x mean radius], offset mostly
    PERPENDICULAR to the axis (the slot lies in the face plane). 2.5x r keeps
    two DISTINCT adjacent terminals unpaired (physical pitch >= 2r + wall)."""
    used =[False ]*len (holes )
    out =[]
    for i ,a in enumerate (holes ):
        if used [i ]:
            continue 
        ra =float (np .mean ([m ["radius"]for m in a ["members"]]))
        ad_a =int (np .bincount ([m ["axis_dim"]for m in a ["members"]],
        minlength =3 ).argmax ())
        best_j =None 
        for j in range (i +1 ,len (holes )):
            if used [j ]:
                continue 
            b =holes [j ]
            rb =float (np .mean ([m ["radius"]for m in b ["members"]]))
            ad_b =int (np .bincount ([m ["axis_dim"]for m in b ["members"]],
            minlength =3 ).argmax ())
            if ad_a !=ad_b :
                continue 
            if abs (ra -rb )>0.05 *max (ra ,rb ):
                continue 
            delta =np .asarray (b ["center"],float )-np .asarray (a ["center"],float )
            d =float (np .linalg .norm (delta ))
            r_mean =0.5 *(ra +rb )
            if not (merge_tol <d <=2.5 *r_mean ):
                continue 
            if abs (delta [ad_a ])>0.5 *d :# offset along the axis, not a slot
                continue 
            best_j =j 
            break 
        if best_j is None :
            out .append (a )
        else :
            used [best_j ]=True 
            b =holes [best_j ]
            out .append ({"center":0.5 *(np .asarray (a ["center"],float )
            +np .asarray (b ["center"],float )),
            "members":a ["members"]+b ["members"],
            "slot":True })
    return out 


def _rect_openings (cyls ,rmin ,bbox ,existing_ops ,merge_tol ):
    """Rectangular / spring-clamp (push-in) wire openings from CORNER-FILLET
    clusters. These openings have no large cylindrical face at all -- the only
    cylinder-ish evidence is the small corner fillets (r ~0.3-1.2mm) rounding
    the slot's edges, which the radius band correctly rejects as terminal holes.
    Measured ten PXC.3031238 (both GT CPs missed by the cylinder path): the GT
    points sit ~2mm from pairs of r=0.5 fillet cylinders. Rule: cluster sub-band
    fillets (same axis, pairwise centre distance <= 10mm); a cluster of 2..8
    fillets whose extent is a plausible wire-slot size (1..14mm) and which is
    NOT within merge_tol of an already-detected opening becomes one candidate
    opening at the cluster centroid."""
    # Corner fillets of REAL push-in wire slots are small (measured r=0.5 ten
    # PXC.3031238); the 0.7-1.2mm rounds are housing cosmetics whose clusters
    # produced 9 false openings ten PXC.3211813 before this band was tightened.
    fillets =[c for c in cyls if 0.2 <=c ["radius"]<=min (0.65 ,rmin )]
    n =len (fillets )
    if n <2 :
        return []
    parent =list (range (n ))

    def find (x ):
        while parent [x ]!=x :
            parent [x ]=parent [parent [x ]]
            x =parent [x ]
        return x 

    for i in range (n ):
        for j in range (i +1 ,n ):
            if fillets [i ]["axis_dim"]!=fillets [j ]["axis_dim"]:
                continue 
            d =np .linalg .norm (fillets [i ]["center"]-fillets [j ]["center"])
            if d <=10.0 :
                parent [find (i )]=find (j )
    groups ={}
    for i in range (n ):
        groups .setdefault (find (i ),[]).append (fillets [i ])
    ops =[]
    existing_pts =np .array ([o ["entry_point"]for o in existing_ops ],float )if existing_ops else np .zeros ((0 ,3 ))
    for members in groups .values ():
    # A real rectangular/push-in wire slot is bounded by ~4 corner fillets
    # (the four rounded corners of the opening); cosmetic housing rounds
    # cluster in 2-3s. Requiring >=4 fillets drops the isolated 2-3-fillet
    # clusters that flooded the cylinder-only parts (10 of 14 FPs were ten
    # PXC.3211813/3211822, which have NO rect slots) while keeping the genuine
    # 4-corner slot ten PXC.3031238. Geometrically principled, not part-specific.
        if not (4 <=len (members )<=8 ):
            continue 
        centers =np .array ([m ["center"]for m in members ])
        ad =int (np .bincount ([m ["axis_dim"]for m in members ],minlength =3 ).argmax ())
        span =centers .max (0 )-centers .min (0 )
        perp =[span [k ]for k in range (3 )if k !=ad ]
        if not (1.0 <=max (perp )<=8.0 ):# plausible wire-slot width, not a
            continue # housing-wide cosmetic fillet run
            # the 4 corners must actually bound a rectangle: BOTH in-face extents
            # non-trivial (a co-linear run of fillets is an edge round, not a slot)
        if min (perp )<0.8 :
            continue 
        c =centers .mean (0 )
        if len (existing_pts )and np .linalg .norm (existing_pts -c ,axis =1 ).min ()<merge_tol :
            continue # already covered by a bore
        direction =np .zeros (3 )
        direction [ad ]=_dir_sign (c ,ad ,bbox )
        ops .append ({"entry_point":[float (x )for x in c ],
        "approach_vector":[float (x )for x in direction ],
        "radius_mm":float (0.5 *max (perp )),
        "n_faces":len (members ),"kind":"rect"})
    return ops 


def mouth_coord (members ,ad ,sign ,bbox ):
    """Coordinate of an opening's MOUTH along its approach axis `name`.

    A bore/funnel face physically ENDS at the mouth, so the face's own extreme
    along the approach axis IS the surface -- no ray-cast needed. Taking the
    extreme over the merged members lands ten the outermost (lead-in) face.

    This replaces the blind `centre + depth/2` push (RESULTS 11s), which mixed
    two frames -- the merged centre is a MEAN of member centres while `depth` is
    the DEEPEST member's extent -- and so overshot past the mouth to a point
    OUTSIDE the part whenever a shallow funnel merged with a deep bore
    (human-GT F1 32.5 -> 12.5, angle error 38.6 -> 90 degrees).

    Clamped to the part bbox: a mouth can never lie outside the part.
    """
    ends =[m ["center"][ad ]+0.5 *float (sign )*float (m ["ext"][ad ])
    for m in members ]
    x =max (ends )if sign >0 else min (ends )
    return float (np .clip (x ,bbox [ad ],bbox [ad +3 ]))


def _axis_key (v ):
    """Signed dominant axis of an approach vector: (0,'+') .. (2,'-')."""
    v =np .asarray (v ,dtype =float )
    i =int (np .argmax (np .abs (v )))
    return (i ,"+"if v [i ]>0 else "-")


def plausible_openings (ops ,topk =2 ,radius_buckets =2 ,part_nr ="?"):
    """Drop openings that cannot PHYSICALLY be wire entries.

    Two rules, both derived from what a terminal block IS. Neither is tuned ten the
    9-part human set -- that is exactly what poisoned the previous attempt
    (RESULTS 11s: filters fitted to 9 PXC parts deleted 73% of all CPs corpus-wide
    and left 25 of 92 parts with ZERO, and a terminal block always has at least two
    wire entries).

    1. DIRECTION, top-K. A wire goes in through ONE face -- or two, ten a
       feed-through block. Keep only openings whose approach axis is among the K
       most common in this part. Measured 2026-07-14 over corpus v8: 1119 of 2951
       parts (37.9%) carry "connection points" ten FOUR OR MORE faces, and those
       parts hold 53.6% of ALL ground truth. The extra faces are mounting, screw
       and vent holes.
       This is NOT `dir_consensus`, which keeps only the SINGLE dominant group and
       therefore decapitates every feed-through block (one of the ways 11s died).
       K=2 keeps both entry faces.

    2. RADIUS, dominant cluster. A product accepts one or two wire gauges, so its
       entries cluster tightly in radius. Bucket radii at 0.5mm and keep the
       fullest bucket plus the next `radius_buckets - 1` fullest. Measured: the
       control part 3260064 -- whose labels the model reproduces exactly -- has ONE
       radius cluster (1.60-1.63mm); part 1032400000 (labelled with 134 "CPs") has
       SIX, spanning 1.0 to 4.5mm. No single block accepts a 4.5x range of wire.

    Pure function over the `ops` records. `depth_mm` is deliberately NOT used: the
    rect / rect_pocket paths never emit it, and depth turned out not to separate the
    classes anyway (the 134-CP part's junk holes are 9mm deep ten average).

    SAFETY VALVE: a positive part is never emptied. If both rules together would
    leave nothing, the ORIGINAL openings are returned and a warning is logged -- a
    terminal block with zero entries is always a labelling failure, and silently
    producing one is how 11s destroyed 25 parts. The dry-run gate counts how often
    this fires.
    """
    if len (ops )<3 :# too few to infer a pattern -- leave it alone
        return list (ops )

    from collections import Counter 

    def _bucket (o ):
        return round (float (o ["radius_mm"])*2.0 )/2.0 

        # ORDER MATTERS. The radius rule is applied only to the openings that SURVIVED the
        # face rule, never to all of them: mounting and screw holes ten the minor faces skew
        # the global radius histogram, so a part with many identical screw holes would make
        # SCREW the dominant gauge and delete the real entries instead. Selecting the face
        # first means the gauge is inferred from the entries themselves.
    ax_counts =Counter (_axis_key (o ["approach_vector"])for o in ops )
    keep_axes ={a for a ,_ in ax_counts .most_common (max (1 ,int (topk )))}
    on_face =[o for o in ops if _axis_key (o ["approach_vector"])in keep_axes ]

    r_counts =Counter (_bucket (o )for o in on_face )
    keep_radii ={r for r ,_ in r_counts .most_common (max (1 ,int (radius_buckets )))}
    kept =[o for o in on_face if _bucket (o )in keep_radii ]

    if not kept :
        logger .warning ("%s: plausibility filter would drop ALL %d opening(s) -- "
        "keeping them; a terminal block cannot have zero entries",
        part_nr ,len (ops ))
        return list (ops )

    n_drop =len (ops )-len (kept )
    if n_drop :
        logger .info ("%s: plausibility filter dropped %d/%d opening(s) "
        "(axes kept: %s of %d; radius buckets kept: %s of %d)",
        part_nr ,n_drop ,len (ops ),
        sorted (keep_axes ),len (ax_counts ),
        sorted (keep_radii ),len (r_counts ))
    return kept 


def openings_from_step (path ,rmin =1.3 ,rmax =3.0 ,merge_tol =4.0 ,auto =False ,
drop_corner =0.0 ,slot_pairs =True ,rect_clusters =False ,
include_cones =True ,dir_consensus =False ,
min_depth =0.0 ,deep_axis =False ,auto_multi =False ,
entry_at_mouth =False ,plausible =False ,
plausible_topk =2 ,plausible_radius_buckets =2 ):
    """Cylinders of terminal radius -> merged openings (centre + axis + radius).

    auto=True picks the radius band per part (auto_radius_range). drop_corner>0
    (EXPERIMENTAL, opt-in): drop holes whose in-face position is within this
    fraction of the part edge -- likely mounting/corner holes. RISKY: connector
    terminals are often near an edge too, so it can drop real CPs; default off.

    slot_pairs: fuse racetrack/oval slot half-cylinder pairs (see
    _pair_slot_halves). rect_clusters: detect rectangular/push-in openings from
    corner-fillet clusters (see _rect_openings). include_cones: treat conical
    funnel lead-ins like cylinders. dir_consensus (opt-in): keep only openings
    whose approach direction belongs to the DOMINANT direction group when that
    group is a clear majority -- real wire entries ten a terminal strip share one
    face/axis, while side/bottom-facing cylinders (DIN-rail latches, test
    points) are false positives; measured ten PXC.3211813 the sideways
    detections were 3 of the 5 FPs.

    merge_tol is capped to 4% of the part's own bbox diagonal (the same
    nms_scale_frac used by cp_targets.decode_predictions for the eval/decode-side
    merge radius), so labeling and eval/decode use the SAME size-adaptive rule
    instead of three independently-chosen mm constants (this used to be a flat
    4.0mm here vs a flat 5.0mm for decode NMS / eval match radius -- close enough
    to accidentally either fuse two real close-together holes at label time or
    double-count one real hole as two at eval time, ten the same small part)."""
    cyls ,bbox =extract_cylinders (path ,include_cones =include_cones )
    part_diag =float (np .linalg .norm (bbox [3 :]-bbox [:3 ]))if len (bbox )==6 else 0.0 
    if part_diag >0 :
    # scale-adaptive, like the decoder, but with the LABEL ceiling (4.0mm)
        merge_tol =min (float (merge_tol ),
        merge_radius_mm (part_diag ,ceiling =MERGE_CEILING_MM ))
    if auto_multi :
        bands =auto_radius_bands ([c ["radius"]for c in cyls ])
        keep =[c for c in cyls 
        if any (lo <=c ["radius"]<=hi for lo ,hi in bands )]
        rmin ,rmax =bands [0 ][0 ],bands [-1 ][1 ]# for the drop-warning below
    else :
        if auto :
            rmin ,rmax =auto_radius_range ([c ["radius"]for c in cyls ])
        keep =[c for c in cyls if rmin <=c ["radius"]<=rmax ]
        # Visibility for the rmin/rmax band: cylinders outside it are SILENTLY absent
        # from the label set (not just filtered predictions) -- a real connector family
        # with terminal holes outside this band would lose that whole size class from
        # ground truth with no error anywhere. Warn (before per file) so it's noticeable
        # instead of looking like "the model never learned this hole size".
    dropped =[c ["radius"]for c in cyls if not (rmin <=c ["radius"]<=rmax )]
    if dropped :
        near_band =[r for r in dropped if (rmin -1.0 )<=r <=(rmax +1.0 )]
        if near_band :
            logger .warning (
            "%s: %d cylinder(s) with radius near but OUTSIDE [%.1f, %.1f]mm "
            "excluded from labels (radii: %s) -- if these are real terminal "
            "holes, widen --rmin/--rmax or use --auto",
            os .path .basename (path ),len (near_band ),rmin ,rmax ,
            sorted (round (r ,1 )for r in near_band ))
    holes =[]
    for c in keep :
        for h in holes :
            if np .linalg .norm (c ["center"]-h ["center"])<merge_tol :
                h ["members"].append (c )
                break 
        else :
            holes .append ({"center":c ["center"].copy (),"members":[c ]})
    if slot_pairs and len (holes )>=2 :
        holes =_pair_slot_halves (holes ,merge_tol )
    ops =[]
    for h in holes :
        c =np .mean ([m ["center"]for m in h ["members"]],axis =0 )
        r =float (np .mean ([m ["radius"]for m in h ["members"]]))
        # per-member bore depth = bbox extent along that member's own axis; the
        # opening's depth is its deepest member, so a funnel+bore merge keeps
        # the bore's depth rather than the shallow lead-in's.
        depths =[float (m ["ext"][m ["axis_dim"]])for m in h ["members"]]
        depth =max (depths )
        # opt-in FP filter (RESULTS 11k #1): real wire entries are DEEP bores;
        # LED windows, marking pockets and decorative recesses are shallow.
        if min_depth >0.0 and depth <min_depth :
            continue 
        if deep_axis :
        # axis = the DEEPEST member's axis instead of the member-count
        # majority: ten fuse blocks the wire bore (deep) merges with several
        # shallow cavity/LED faces and the count vote picks the wrong axis
        # -- the measured ang=90 failure ten ST 4-HESI*/PTME (RESULTS 11k #2).
            ad =int (h ["members"][int (np .argmax (depths ))]["axis_dim"])
        else :
            ad =int (np .bincount ([m ["axis_dim"]for m in h ["members"]],
            minlength =3 ).argmax ())
        face_dims =[i for i in range (3 )if i !=ad ]
        # #2 corner / mounting-hole filter (opt-in): skip holes hugging a face edge
        if drop_corner >0.0 :
            edge =False 
            for fd in face_dims :
                span =bbox [fd +3 ]-bbox [fd ]
                if span >0 and min (c [fd ]-bbox [fd ],bbox [fd +3 ]-c [fd ])/span <drop_corner :
                    edge =True 
            if edge :
                continue 
                # #3 insert-direction sign: see _dir_sign (bbox-midpoint fallback -- the
                # earlier mean-of-hole-centres proxy silently flipped signs ten parts with
                # an asymmetric hole layout).
        direction =np .zeros (3 )
        direction [ad ]=_dir_sign (c ,ad ,bbox )
        entry =c .copy ()
        if entry_at_mouth :
        # gmsh's getCenterOfMass for a bore face sits at MID-DEPTH, i.e. inside
        # the part -- but a connection point is where the wire ENTERS, and the
        # human ABB ground truth marks the opening's mouth. Measured ten the real
        # corpus: CAD-labelled CPs sat 0.62mm from the nearest surface (hugging
        # the bore wall) while human-labelled CPs sat ~3.05mm out (in the void at
        # the mouth) -- two conflicting conventions taught to one model.
        #
        # The mouth is taken from the MEMBER FACES' OWN outer ends, not by
        # pushing the merged centre out by depth/2. That blind push (RESULTS 11s)
        # mixed two frames -- `c` is the MEAN of member centres while `depth` is
        # the DEEPEST member's -- so ten a funnel+bore merge it overshot past the
        # mouth to a point outside the part (human-GT F1 32.5 -> 12.5, ang 90).
        # A face ENDS at the mouth, so its own extreme along the approach axis IS
        # the surface; taking the extreme over members lands ten the outermost
        # (lead-in) face and cannot overshoot. See mouth_coord().
            entry [ad ]=mouth_coord (h ["members"],ad ,direction [ad ],bbox )
        ops .append ({"entry_point":[float (x )for x in entry ],
        "approach_vector":[float (x )for x in direction ],
        "radius_mm":r ,"n_faces":len (h ["members"]),
        "depth_mm":depth ,
        "kind":"slot"if h .get ("slot")else "hole"})
    if rect_clusters :
        ops +=_rect_openings (cyls ,rmin ,bbox ,ops ,merge_tol )
        # planar-pocket path: catches rectangular push-in wire slots that have NO
        # cylindrical face at all (only planar walls) -- the cylinder detector's
        # structural blind spot. Approach axis = the dominant axis of the
        # cylinder openings already found (real wire entries share one face), or
        # the part's deepest axis (z) when there are none.
        if ops :
            approach_dim =int (np .bincount (
            [int (np .argmax (np .abs (o ["approach_vector"])))for o in ops ],
            minlength =3 ).argmax ())
        else :
            approach_dim =int (np .argmax (bbox [3 :]-bbox [:3 ]))if len (bbox )==6 else 2 
        try :
            planes ,_pbbox =extract_planar_faces (path )
            ops +=_planar_pocket_openings (planes ,bbox ,ops ,merge_tol ,
            approach_dim =approach_dim )
        except Exception as exc :# noqa: BLE001
            logger .warning ("%s: planar-pocket detection skipped (%s)",
            os .path .basename (path ),exc )
    if not ops :
        return []
        # opt-in FP filter: real wire entries ten a terminal strip share one dominant
        # approach direction; side/bottom-facing detections (DIN latches, test
        # points) are FPs. Keep only the STRICT-PLURALITY group: it must have >=2
        # members, beat the runner-up by >=1, and hold >=40% of all openings. A tie
        # (e.g. 2 top entries vs 2 bottom latches) filters nothing -- safe fallback.
    if dir_consensus and len (ops )>=3 :
        from collections import Counter 
        key =lambda o :tuple (int (round (x ))for x in o ["approach_vector"])
        counts =Counter (key (o )for o in ops ).most_common ()
        dom ,dom_n =counts [0 ]
        runner_n =counts [1 ][1 ]if len (counts )>1 else 0 
        if dom_n >=2 and dom_n >=runner_n +1 and dom_n >=0.4 *len (ops ):
            dropped =[o for o in ops if key (o )!=dom ]
            if dropped :
                logger .info ("%s: dir-consensus dropped %d opening(s) not facing "
                "the dominant %s direction",os .path .basename (path ),
                len (dropped ),dom )
            ops =[o for o in ops if key (o )==dom ]
            # opt-in PHYSICAL PLAUSIBILITY filter (see plausible_openings): drops openings
            # that cannot be wire entries -- holes ten the part's minor faces, and holes whose
            # radius sits outside the product's own wire-gauge cluster. Default OFF, so every
            # existing caller (and a v31 --resume) is byte-identical.
    if plausible :
        ops =plausible_openings (ops ,topk =plausible_topk ,
        radius_buckets =plausible_radius_buckets ,
        part_nr =os .path .basename (path ))
    return ops 


def run (source ,rmin =1.3 ,rmax =3.0 ,merge_tol =4.0 ,auto =False ,drop_corner =0.0 ,
slot_pairs =True ,rect_clusters =False ,include_cones =True ,
dir_consensus =False ,check_dirs =False ,min_depth =0.0 ,deep_axis =False ,
auto_multi =False ,entry_at_mouth =False ,plausible =False ,
plausible_topk =2 ,plausible_radius_buckets =2 ):
    if os .path .isdir (source ):
        files =sorted (glob .glob (os .path .join (source ,"**","*.st*p"),recursive =True ))
    else :
        files =[source ]
    parts =[]
    for f in files :
        if os .path .splitext (f )[1 ].lower ()not in (".stp",".step"):
            continue 
        part_nr =os .path .splitext (os .path .basename (f ))[0 ]
        try :# one bad STEP must not abort the batch
            ops =openings_from_step (f ,rmin =rmin ,rmax =rmax ,merge_tol =merge_tol ,
            auto =auto ,drop_corner =drop_corner ,
            slot_pairs =slot_pairs ,
            rect_clusters =rect_clusters ,
            include_cones =include_cones ,
            dir_consensus =dir_consensus ,
            min_depth =min_depth ,deep_axis =deep_axis ,
            auto_multi =auto_multi ,
            entry_at_mouth =entry_at_mouth ,
            plausible =plausible ,
            plausible_topk =plausible_topk ,
            plausible_radius_buckets =plausible_radius_buckets )
            if check_dirs and ops :
            # prediction path has no mesh in hand -- tessellate coarsely
            # (0.3mm is plenty for a clearance probe) just for the check
                import step_to_json as sj 
                V ,_ =sj .load_any_mesh (f ,deflection =0.3 )
                validate_directions (ops ,V ,part_nr =part_nr )
        except Exception as exc :# noqa: BLE001
            logger .warning ("  %-30s SKIPPED (unreadable STEP): %s",part_nr ,exc )
            continue 
        logger .info ("  %-30s %d opening(s)",part_nr ,len (ops ))
        parts .append ({"part_nr":part_nr ,"n_detected":len (ops ),
        "connection_points":ops })
    return {"detector":"step_cylinders_gmsh",
    "params":{"rmin":rmin ,"rmax":rmax ,"merge_tol":merge_tol ,
    "auto":auto ,"drop_corner":drop_corner ,
    "slot_pairs":slot_pairs ,"rect_clusters":rect_clusters ,
    "include_cones":include_cones ,
    "dir_consensus":dir_consensus ,
    "min_depth":min_depth ,"deep_axis":deep_axis },
    "parts":parts }


def to_abb_labels (openings ):
    """Detected openings -> ABB ConnectionPoints (for fine-tuning the ML model)."""
    return [{"Index":i ,"Name":f"OPEN-{i }",
    "Point":{"X":o ["entry_point"][0 ],"Y":o ["entry_point"][1 ],
    "Z":o ["entry_point"][2 ]},
    "InsertDirection":{"X":o ["approach_vector"][0 ],
    "Y":o ["approach_vector"][1 ],
    "Z":o ["approach_vector"][2 ]}}
    for i ,o in enumerate (openings )]


def _geom_sig (V ,n_ops ):
    """Coarse geometry fingerprint (bbox dims in mm + vertex + CP count) to catch
    near-identical re-downloads of the same part."""
    if not len (V ):
        return ("empty",n_ops )
    return (tuple (np .round (V .max (0 )-V .min (0 ),0 ).tolist ()),len (V ),n_ops )


def write_labeled_corpus (source ,out_dir ,rmin =1.3 ,rmax =3.0 ,merge_tol =MERGE_CEILING_MM ,
deflection =0.1 ,auto =False ,drop_corner =0.0 ,dedup =True ,
slot_pairs =True ,rect_clusters =False ,include_cones =True ,
dir_consensus =False ,min_depth =0.0 ,deep_axis =False ,
auto_multi =False ,entry_at_mouth =False ):
    """Turn STEP file(s) into a REAL LABELLED ABB-JSON corpus: the gmsh mesh plus
    the cylinder-extracted connection points as ground-truth ConnectionPoints.
    The mesh (step_to_json, gmsh tessellation) and the CPs (step_openings, gmsh
    OCC) share the gmsh coordinate frame, so the labels sit exactly ten the mesh.

    deflection default (0.1) MATCHES step_to_json.py's default so labeled training
    data gets the same tessellation fidelity as inference -- a coarser mesh here
    would under-sample small terminal holes (radius as low as ~1.3mm) at label time,
    silently starving the model of the geometric signal needed to learn them, while
    everything else in the pipeline sees the finer mesh. Pass a larger value only if
    you deliberately want a coarser/faster mesh and understand this tradeoff.

    This is the #2 bridge: it converts unlabelled .stp parts into data you can
    train / fine-tune `train_cp.py` ten. dedup=True skips a part whose geometry
    matches one already written (near-identical re-downloads would otherwise leak
    across the train/val split and waste training). Returns the written paths.

    min_depth / deep_axis / auto_multi are forwarded to openings_from_step exactly
    as in prediction mode: the FP-suppression + direction filters that the deploy
    config relies ten MUST be usable at label time too, or the corpus teaches the
    model the very errors the deploy path deletes (corpus v4 was built without
    them -- RESULTS 11q/11r)."""
    import step_to_json as sj 
    os .makedirs (out_dir ,exist_ok =True )
    if os .path .isdir (source ):
        files =sorted (glob .glob (os .path .join (source ,"**","*.st*p"),recursive =True ))
    else :
        files =[source ]
    written ,seen =[],{}
    for f in files :
        if os .path .splitext (f )[1 ].lower ()not in (".stp",".step"):
            continue 
        part_nr =os .path .splitext (os .path .basename (f ))[0 ]
        try :
            V ,F =sj .load_any_mesh (f ,deflection =deflection )
            ops =openings_from_step (f ,rmin =rmin ,rmax =rmax ,merge_tol =merge_tol ,
            auto =auto ,drop_corner =drop_corner ,
            slot_pairs =slot_pairs ,
            rect_clusters =rect_clusters ,
            include_cones =include_cones ,
            dir_consensus =dir_consensus ,
            min_depth =min_depth ,deep_axis =deep_axis ,
            auto_multi =auto_multi ,
            entry_at_mouth =entry_at_mouth )
            # the mesh is already in hand here, so the mesh-clearance direction
            # check is free -- labels get geometrically-verified signs instead
            # of bbox-heuristic ones (measured: heuristic flips 180deg between
            # runs ten through-channel parts)
            validate_directions (ops ,V ,part_nr =part_nr )
        except Exception as exc :# noqa: BLE001
            logger .warning ("  SKIPPED %s: %s",part_nr ,exc )
            continue 
        if dedup :
            sig =_geom_sig (V ,len (ops ))
            if sig in seen :
                logger .warning ("  %-30s DUPLICATE of %s -- skipped",part_nr ,seen [sig ])
                continue 
            seen [sig ]=part_nr 
        d =sj .mesh_to_abb_dict (part_nr ,V ,F )
        d ["ConnectionPoints"]=to_abb_labels (ops )
        out =os .path .join (out_dir ,part_nr +".json")
        with open (out ,"w",encoding ="utf-8")as fh :
            json .dump (d ,fh )
        logger .info ("  %-30s %d verts, %d labelled CP(s) -> %s",
        part_nr ,len (V ),len (ops ),os .path .basename (out ))
        written .append (out )
    return written 


def main (argv =None ):
    ap =argparse .ArgumentParser (
    description ="Extract connection openings from a STEP B-rep via gmsh "
    "(cylindrical faces) -> CP labels in the mesh frame")
    ap .add_argument ("source",help ="a .stp/.step file or a directory of them")
    ap .add_argument ("--out",help ="output predictions JSON")
    ap .add_argument ("--rmin",type =float ,default =1.3 ,help ="min hole radius (mm)")
    ap .add_argument ("--rmax",type =float ,default =3.0 ,help ="max hole radius (mm)")
    ap .add_argument ("--merge-tol",type =float ,default =MERGE_CEILING_MM ,
    help ="CEILING (mm) for merging cylindrical faces into one "
    "opening; the effective radius is scale-adaptive "
    "(min(ceiling, 0.04*bbox_diag)) and SHARED with the "
    "decoder's NMS radius, so labels and detections agree "
    "on what counts as one opening")
    ap .add_argument ("--list",action ="store_true",
    help ="print the radius histogram of all cylinders and exit")
    ap .add_argument ("--auto",action ="store_true",
    help ="auto-pick the terminal-hole radius band per part (drops "
    "edge fillets, keeps the most frequent hole radius) -- no "
    "manual --rmin/--rmax needed for a new connector family")
    ap .add_argument ("--drop-corner",type =float ,default =0.0 ,
    help ="EXPERIMENTAL (opt-in): drop holes within this fraction of "
    "the part edge (likely mounting holes). RISKY -- can also "
    "drop real edge terminals; default 0 (off)")
    ap .add_argument ("--keep-duplicates",action ="store_true",
    help ="--label-corpus: keep near-identical re-downloaded parts "
    "(default skips geometric duplicates)")
    ap .add_argument ("--label-corpus",metavar ="OUTDIR",
    help ="write a REAL labelled ABB-JSON corpus (mesh + extracted CPs) "
    "here, ready to train/fine-tune train_cp.py on")
    ap .add_argument ("--deflection",type =float ,default =0.1 ,
    help ="--label-corpus mesh tessellation fidelity (mm); matches "
    "step_to_json.py's default so labeled training data has the "
    "same fidelity as inference (a coarser mesh under-samples "
    "small terminal holes at label time)")
    ap .add_argument ("--no-slot-pairs",dest ="slot_pairs",action ="store_false",
    help ="disable racetrack/oval slot half-pair fusion (see "
    "_pair_slot_halves; on by default -- fixes push-in wire "
    "slots surfacing as two offset detections)")
    ap .add_argument ("--rect-slots",dest ="rect_clusters",action ="store_true",
    help ="OPT-IN: detect rectangular/push-in openings from corner-"
    "fillet clusters (_rect_openings). Catches spring-clamp "
    "wire slots invisible to the cylinder path (recovered a "
    "GT CP on PXC.3031238) but at heavy precision cost on the "
    "tiny 3-part eval (~10 extra FPs); default OFF until a "
    "bigger honest eval set exists to tune it on")
    ap .add_argument ("--no-cones",dest ="include_cones",action ="store_false",
    help ="ignore conical funnel lead-in faces (cones are treated "
    "like cylinders by default)")
    ap .add_argument ("--entry-at-mouth",action ="store_true",
    help ="put the CP at the opening's MOUTH (bore centre of mass pushed out by half the bore depth along the approach axis) instead of at the face centroid, which sits at mid-depth INSIDE the part. Matches the human ABB ground-truth convention -- measured: human CPs sit ~3.3mm from the nearest mesh vertex, CAD-labelled ones sat 0.3mm (RESULTS 11r A2)")
    ap .add_argument ("--auto-multi",action ="store_true",
    help ="multi-band radius auto-pick: accept EVERY radius "
    "cluster with >=2 members up to 8mm (a part can have "
    "wire bores AND fuse-screw cartridges at once, e.g. "
    "NEOZED 3048357 -- single-band --auto loses the whole "
    "screw class, RESULTS 11k #3)")
    ap .add_argument ("--plausible",action ="store_true",
    help ="OPT-IN physical-plausibility filter (see "
    "plausible_openings): keep only openings that CAN be wire "
    "entries -- those facing one of the part's 2 most common "
    "faces, AND whose radius sits in the product's own "
    "wire-gauge cluster. Measured 2026-07-14: 1119/2951 corpus "
    "parts carry 'CPs' on 4+ faces and hold 53.6% of all GT; "
    "those extra faces are mounting/screw/vent holes. Never "
    "empties a part (safety valve). Unlike --dir-consensus it "
    "keeps BOTH faces of a feed-through block.")
    ap .add_argument ("--plausible-topk",type =int ,default =2 ,
    dest ="plausible_topk",
    help ="how many approach faces --plausible keeps (default 2: "
    "front, plus the back on a feed-through block)")
    ap .add_argument ("--plausible-radius-buckets",type =int ,default =2 ,
    dest ="plausible_radius_buckets",
    help ="how many 0.5mm radius clusters --plausible keeps "
    "(default 2: a product accepts one or two wire gauges)")
    ap .add_argument ("--min-depth",type =float ,default =0.0 ,
    help ="OPT-IN precision filter: drop openings whose deepest "
    "member face is shallower than this many mm along its "
    "axis -- kills LED windows / marking pockets / "
    "decorative recesses, which are shallow, while real "
    "wire-entry bores are deep (RESULTS 11k #1)")
    ap .add_argument ("--deep-axis",action ="store_true",
    help ="OPT-IN direction fix: take the approach axis from the "
    "DEEPEST merged face instead of the member-count "
    "majority (fixes the ang=90 fuse-block failure where "
    "shallow cavity faces outvote the wire bore, "
    "RESULTS 11k #2)")
    ap .add_argument ("--dir-consensus",action ="store_true",
    help ="OPT-IN precision filter: keep only openings facing the "
    "dominant approach direction when it is a strict plurality "
    "(kills side/bottom-facing DIN-latch and test-point "
    "cylinders; risky on genuinely multi-face connectors). "
    "Measured on the 3-part human-GT eval: F1 57->73%%")
    ap .add_argument ("--check-dirs",action ="store_true",
    help ="verify each approach vector against the tessellated mesh "
    "(free-space clearance probe) and flip ones pointing into "
    "the part; slower (tessellates each STEP) but geometrically "
    "grounded. Always on for --label-corpus (mesh is free there)")
    ap .set_defaults (slot_pairs =True ,rect_clusters =False ,include_cones =True )
    args =ap .parse_args (argv )
    logging .basicConfig (level =logging .INFO ,format ="%(message)s")

    src =args .source 
    if args .label_corpus :
        written =write_labeled_corpus (src ,args .label_corpus ,rmin =args .rmin ,
        rmax =args .rmax ,merge_tol =args .merge_tol ,
        deflection =args .deflection ,
        auto =args .auto ,drop_corner =args .drop_corner ,
        dedup =not args .keep_duplicates ,
        slot_pairs =args .slot_pairs ,
        rect_clusters =args .rect_clusters ,
        include_cones =args .include_cones ,
        dir_consensus =args .dir_consensus ,
        min_depth =args .min_depth ,
        deep_axis =args .deep_axis ,
        auto_multi =args .auto_multi ,
        entry_at_mouth =args .entry_at_mouth )
        print (f"\nwrote {len (written )} labelled part(s) -> {args .label_corpus }")
        print ("fine-tune on them:  python train_cp.py %s --backbone knngraph "
        "--device cuda --split-group none --epochs 200 --run-name cp_real"
        %args .label_corpus )
        return 
    if args .list :
        f =src 
        if os .path .isdir (f ):
            f =sorted (glob .glob (os .path .join (f ,"**","*.st*p"),recursive =True ))[0 ]
        cyls ,_ =extract_cylinders (f )
        rad =np .round ([c ["radius"]for c in cyls ],1 )
        u ,ct =np .unique (rad ,return_counts =True )
        print ("cylinder radii (count x radius_mm):")
        for r ,c in zip (u ,ct ):
            print (f"  {c :3d} x {r :.1f}")
        return 

    res =run (src ,rmin =args .rmin ,rmax =args .rmax ,merge_tol =args .merge_tol ,
    auto =args .auto ,drop_corner =args .drop_corner ,
    slot_pairs =args .slot_pairs ,rect_clusters =args .rect_clusters ,
    include_cones =args .include_cones ,dir_consensus =args .dir_consensus ,
    check_dirs =args .check_dirs ,min_depth =args .min_depth ,
    deep_axis =args .deep_axis ,auto_multi =args .auto_multi ,
    entry_at_mouth =args .entry_at_mouth ,plausible =args .plausible ,
    plausible_topk =args .plausible_topk ,
    plausible_radius_buckets =args .plausible_radius_buckets )
    if args .out :
        with open (args .out ,"w",encoding ="utf-8")as fh :
            json .dump (res ,fh ,indent =2 )
    tot =sum (p ["n_detected"]for p in res ["parts"])
    print (f"\n{len (res ['parts'])} part(s), {tot } opening(s) -> {args .out or '(stdout)'}")


def _selftest ():
    """Build a STEP box with two drilled holes (gmsh) -> the detector finds two."""
    import tempfile 
    import shutil 
    import gmsh 
    work =tempfile .mkdtemp (prefix ="stepcp_")
    path =os .path .join (work ,"selftest_box.step")
    gmsh .initialize ()
    try :
        gmsh .option .setNumber ("General.Terminal",0 )
        box =gmsh .model .occ .addBox (0 ,0 ,0 ,40 ,20 ,8 )
        c1 =gmsh .model .occ .addCylinder (12 ,10 ,-1 ,0 ,0 ,10 ,2.0 )
        c2 =gmsh .model .occ .addCylinder (28 ,10 ,-1 ,0 ,0 ,10 ,2.0 )
        gmsh .model .occ .cut ([(3 ,box )],[(3 ,c1 ),(3 ,c2 )])
        gmsh .model .occ .synchronize ()
        gmsh .write (path )
    finally :
        gmsh .finalize ()
    try :
        ops =openings_from_step (path ,rmin =1.0 ,rmax =3.0 ,merge_tol =4.0 )
    finally :
        shutil .rmtree (work ,ignore_errors =True )
    assert len (ops )==2 ,f"expected 2 drilled holes, got {len (ops )}"
    print ("step_openings selftest OK: found",len (ops ),"hole(s) of r~2mm")


if __name__ =="__main__":
    import sys 
    if len (sys .argv )==1 :
        _selftest ()
    else :
        main ()
