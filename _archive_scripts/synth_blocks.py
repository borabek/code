"""Procedural synthetic terminal-block generator -> labelled corpus JSON.

WHY THIS EXISTS (the generalisation lever): the real corpus is only ~136 terminal
blocks. A from-scratch geometric detector trained ten that few parts memorises those
specific shapes and fails ten an UNSEEN part -- the user's core complaint. The fix is
not augmentation (re-poses the same 136) but genuine DIVERSITY: teach the model the
*concept* of a recessed cylindrical wire-entry opening, not 136 instances.

So we synthesise thousands of plausible terminal blocks with RANDOM geometry (pole
count, pitch, hole radius, recess depth, housing size, 1-2 rows) and EXACT, free CP
labels (we place the holes, so we know every connection point with zero labelling
error). Distractor screw holes (different radius, UNLABELLED) teach "not every hole
is a CP" -> precision. The model learns the general hole geometry -> generalises.

Two uses, which COMBINE (the standard small-data recipe):
  1. PRETRAIN the knngraph ten a large synthetic set -> weights that already know
     "concave cylindrical opening = CP".
  2. FINE-TUNE those weights ten the 136 real parts -> adapts to real detail.

Output JSON matches json_dataset.part_from_dict exactly:
  {"PartNr", "Graphic3d":{"Points":[{X,Y,Z}..],"Indices":[i,i,i,..]},
   "ConnectionPoints":[{"Point":{X,Y,Z},"InsertDirection":{X,Y,Z},"Name"}]}

Geometry is in MILLIMETRES (the whole pipeline's unit). The knngraph backbone keys ten
VERTICES (+ kNN PCA normals); faces are emitted too (valid for viewers / DiffusionNet).

Usage:
  python synth_blocks.py --out synth_corpus --n 2000 --seed 0
  python synth_blocks.py --out synth_preview --n 5 --seed 1   # quick visual check
"""
import os 
import json 
import argparse 
import logging 

import numpy as np 

logger =logging .getLogger (__name__ )


# ---------------------------------------------------------------------------
# one random parameter set (a "blueprint" for a terminal block)
# ---------------------------------------------------------------------------

def sample_params (rng ):
    """Draw one plausible terminal-block parameter set (all mm). Ranges chosen to
    span the real wscad/PXC variety: feed-through, multi-pole, double-deck, fine
    and coarse pitch, short and long strips."""
    n_rows =int (rng .integers (1 ,3 ))# 1 or 2 rows (double-deck)
    n_poles =int (rng .integers (2 ,17 ))# 2..16 wire-entry holes per row
    pitch =float (rng .uniform (3.5 ,12.0 ))# centre-to-centre spacing (X)
    r =float (rng .uniform (1.2 ,3.0 ))# CP wire-entry hole radius
    # recess depth into the body: a MIXTURE, not one range for every part. Measured
    # directly ten the real corpus (nearest-vertex dist / bbox diagonal): ABB median
    # 0.78% (94% of CPs under ~2%), wscad median 0.67% -- most real CPs sit almost
    # AT the surface (the hole radius itself explains the small residual distance).
    # Only ~6% of real ABB CPs are genuinely recessed (the 5-15% "socketed terminal"
    # tier cp_regressor.py logs about). A single depth range that recesses EVERY
    # synthetic CP (the previous fix) just swaps one systematic mismatch (always
    # 0%, depth never reached the label) for the opposite one (always 2-10%,
    # overshooting what ~94% of real CPs look like). Match the real mixture instead.
    deep_recess =bool (rng .random ()<0.12 )
    depth =float (rng .uniform (2.0 ,8.0 ))if deep_recess else float (rng .uniform (0.05 ,1.0 ))
    Lz =float (rng .uniform (14.0 ,45.0 ))# housing height (Z)
    Ly =float (rng .uniform (max (2.5 *r +2.0 ,5.0 ),22.0 ))# housing depth (Y)
    row_gap =float (rng .uniform (3.0 ,8.0 ))if n_rows ==2 else 0.0 
    margin =float (rng .uniform (2.0 ,5.0 ))# housing overhang past end holes
    # distractor screw/test holes: a DIFFERENT radius, UNLABELLED (teach precision)
    n_distract =int (rng .integers (0 ,max (1 ,n_poles //2 )+1 ))
    r_distract =float (rng .uniform (0.6 ,1.1 ))if rng .random ()<0.5 else float (rng .uniform (3.4 ,4.5 ))# clearly not the CP radius
    # block_face: which face the wire-entry holes open from.
    # "top" (60%): current XY-top, dir=[0,0,1]. "front" (25%): XZ-front, dir=[0,1,0].
    # "side" (15%): YZ-side, dir=[1,0,0]. Axis permutations applied post-build.
    block_face =str (rng .choice (["top","front","side"],
    p =[0.60 ,0.25 ,0.15 ]))
    return dict (n_rows =n_rows ,n_poles =n_poles ,pitch =pitch ,r =r ,depth =depth ,
    deep_recess =deep_recess ,
    Lz =Lz ,Ly =Ly ,row_gap =row_gap ,margin =margin ,
    n_distract =n_distract ,r_distract =r_distract ,
    block_face =block_face ,
    # mesh density: interior grid spacing and rim segment count
    grid_h =float (rng .uniform (1.2 ,2.2 )),rim_seg =int (rng .integers (14 ,26 )))


    # ---------------------------------------------------------------------------
    # build the mesh + CP labels for one parameter set
    # ---------------------------------------------------------------------------

def _circle (cx ,cy ,r ,n ,phase =0.0 ):
    a =phase +np .linspace (0.0 ,2 *np .pi ,n ,endpoint =False )
    return np .stack ([cx +r *np .cos (a ),cy +r *np .sin (a )],axis =1 )


def build_block (p ,rng ):
    """Return (vertices (N,3), faces (M,3), cp_points (K,3), cp_dirs (K,3)).

    Layout: housing box [0,Lx]x[0,Ly]x[0,Lz]; wire-entry holes are recessed pockets
    in the TOP face (z=Lz), axis -Z, opening up; CP at each opening centre, dir +Z.
    Distractor holes are pockets too but get NO CP label."""
    from scipy .spatial import Delaunay 

    # --- hole centres -------------------------------------------------------
    span =(p ["n_poles"]-1 )*p ["pitch"]
    Lx =span +2 *p ["margin"]
    ys =([p ["Ly"]/2 ]if p ["n_rows"]==1 
    else [p ["Ly"]/2 -p ["row_gap"]/2 ,p ["Ly"]/2 +p ["row_gap"]/2 ])
    xs =p ["margin"]+np .arange (p ["n_poles"])*p ["pitch"]
    cp_holes =[(float (x ),float (y ),p ["r"])for y in ys for x in xs ]

    # distractor holes: random spots not overlapping CP holes, different radius.
    # A big screw-hole radius may not fit a thin housing -> shrink it to fit, and
    # skip entirely if even that leaves no room (never crash ten a tight blueprint).
    rd =min (p ["r_distract"],(p ["Ly"]-2.0 )/2.0 ,(Lx -2.0 )/2.0 )
    dist_holes =[]
    dy_lo ,dy_hi =rd +1.0 ,p ["Ly"]-rd -1.0 
    dx_lo ,dx_hi =rd +1.0 ,Lx -rd -1.0 
    tries =0 
    while (len (dist_holes )<p ["n_distract"]and tries <200 
    and rd >0.4 and dy_hi >dy_lo and dx_hi >dx_lo ):
        tries +=1 
        dx =float (rng .uniform (dx_lo ,dx_hi ))
        dy =float (rng .uniform (dy_lo ,dy_hi ))
        if all ((dx -hx )**2 +(dy -hy )**2 >(hr +rd +1.5 )**2 
        for hx ,hy ,hr in cp_holes ):
            dist_holes .append ((dx ,dy ,rd ))
    holes =cp_holes +dist_holes # all pockets (CP + distractor)

    # --- 2D point set for the holed top face --------------------------------
    h =p ["grid_h"]
    pts ,rim_idx =[],[]# rim_idx[hole] -> [vert ids]
    # rectangle boundary
    nx =max (2 ,int (round (Lx /h )));ny =max (2 ,int (round (p ["Ly"]/h )))
    for t in np .linspace (0 ,Lx ,nx ):
        pts .append ((t ,0.0 ));pts .append ((t ,p ["Ly"]))
    for t in np .linspace (0 ,p ["Ly"],ny ):
        pts .append ((0.0 ,t ));pts .append ((Lx ,t ))
        # interior grid, skipping inside-hole points
    for gx in np .arange (h ,Lx ,h ):
        for gy in np .arange (h ,p ["Ly"],h ):
            if all ((gx -hx )**2 +(gy -hy )**2 >(hr *1.15 )**2 
            for hx ,hy ,hr in holes ):
                pts .append ((float (gx ),float (gy )))
                # rim rings (two concentric: tight rim + a slightly larger ring for density)
    for (hx ,hy ,hr )in holes :
        ring =_circle (hx ,hy ,hr ,p ["rim_seg"],phase =rng .uniform (0 ,1 ))
        start =len (pts )
        pts .extend (map (tuple ,ring ))
        rim_idx .append (list (range (start ,len (pts ))))
        outer =_circle (hx ,hy ,hr *1.6 ,max (8 ,p ["rim_seg"]//2 ),
        phase =rng .uniform (0 ,1 ))
        pts .extend (map (tuple ,outer ))
    P2 =np .asarray (pts ,dtype =np .float64 )

    # --- triangulate, drop triangles whose centroid is inside any hole ------
    tri =Delaunay (P2 )
    cent =P2 [tri .simplices ].mean (axis =1 )
    keep =np .ones (len (tri .simplices ),dtype =bool )
    for (hx ,hy ,hr )in holes :
        keep &=((cent [:,0 ]-hx )**2 +(cent [:,1 ]-hy )**2 )>(hr **2 )
    top_faces =tri .simplices [keep ]

    # --- assemble 3D vertices/faces -----------------------------------------
    V =[(x ,y ,p ["Lz"])for (x ,y )in P2 ]# top face lifted to z=Lz
    F =[tuple (int (i )for i in f )for f in top_faces ]

    # pocket floor height, shared by every hole's walls AND the CP label below --
    # real socketed-terminal CPs sit 5-15% of bbox diagonal BELOW the surface
    # (cp_regressor._prepare_sample_impl), not at the opening plane.
    zb =p ["Lz"]-min (p ["depth"],p ["Lz"]-1.0 )# pocket floor (keep >0)

    # pocket walls + bottom for every hole
    for hi ,(hx ,hy ,hr )in enumerate (holes ):
        rim =rim_idx [hi ]
        bottom =[]
        for vid in rim :
            x ,y ,_ =V [vid ]
            V .append ((x ,y ,zb ));bottom .append (len (V )-1 )
        m =len (rim )
        for j in range (m ):# wall quad -> 2 tris
            a ,b =rim [j ],rim [(j +1 )%m ]
            c ,d =bottom [(j +1 )%m ],bottom [j ]
            F .append ((a ,b ,c ));F .append ((a ,c ,d ))
        cxv =len (V );V .append ((hx ,hy ,zb ))# floor centre -> fan
        for j in range (m ):
            F .append ((bottom [j ],bottom [(j +1 )%m ],cxv ))

            # housing shell: 4 sides + bottom (coarse box, adds negative bulk)
    base =len (V )
    box =[(0 ,0 ,0 ),(Lx ,0 ,0 ),(Lx ,p ["Ly"],0 ),(0 ,p ["Ly"],0 ),
    (0 ,0 ,p ["Lz"]),(Lx ,0 ,p ["Lz"]),(Lx ,p ["Ly"],p ["Lz"]),
    (0 ,p ["Ly"],p ["Lz"])]
    V .extend (box )
    b =base 
    quads =[(b +0 ,b +1 ,b +2 ,b +3 ),# bottom
    (b +0 ,b +1 ,b +5 ,b +4 ),# sides
    (b +1 ,b +2 ,b +6 ,b +5 ),
    (b +2 ,b +3 ,b +7 ,b +6 ),
    (b +3 ,b +0 ,b +4 ,b +7 )]
    for (a ,bb ,c ,d )in quads :
        F .append ((a ,bb ,c ));F .append ((a ,c ,d ))
        # subdivide the 4 tall sides for extra negative-region density
    for (x0 ,y0 ,x1 ,y1 )in [(0 ,0 ,Lx ,0 ),(Lx ,0 ,Lx ,p ["Ly"]),
    (Lx ,p ["Ly"],0 ,p ["Ly"]),(0 ,p ["Ly"],0 ,0 )]:
        steps =max (2 ,int (np .hypot (x1 -x0 ,y1 -y0 )/h ))
        for s in np .linspace (0 ,1 ,steps ):
            for zz in np .arange (h ,p ["Lz"],h ):
                V .append ((x0 +s *(x1 -x0 ),y0 +s *(y1 -y0 ),float (zz )))

    Vn =np .asarray (V ,dtype =np .float64 )
    Fn =np .asarray (F ,dtype =np .int64 )

    # --- CP labels: pocket-floor centre (z=zb), outward +Z (only the CP holes).
    # For a genuinely "deep_recess" part, the true contact point is placed a bit
    # BELOW the tessellated floor (not exactly ten the floor-centre fan vertex) --
    # real recessed terminals are physically inside a socket cavity the coarse
    # mesh doesn't resolve down to (cp_targets.cp_surface_distances would read
    # exactly 0 otherwise, since a mesh vertex sits exactly at (hx,hy,zb) by
    # construction; that's fine for the shallow/at-surface majority, but wrong
    # for the minority that should look genuinely unresolved-by-the-mesh).
    if p .get ("deep_recess"):
        extra =min (1.1 *p ["depth"],0.6 *zb )# stay well clear of z=0 (housing floor)
        cp_z =zb -extra 
    else :
        cp_z =zb 
    cp_pts =np .array ([(hx ,hy ,cp_z )for (hx ,hy ,hr )in cp_holes ],
    dtype =np .float64 )
    cp_dir =np .tile (np .array ([0.0 ,0.0 ,1.0 ]),(len (cp_pts ),1 ))

    # recentre to origin (training normalises anyway, but keeps numbers tidy)
    c =Vn .mean (0 )
    Vn =Vn -c ;cp_pts =cp_pts -c 

    # Apply axis permutation to put CP holes ten the requested face.
    # "top"  (default): holes ten XY-top face, dir=[0,0,1] — no change.
    # "front": swap Y↔Z → holes open from XZ-front face, dir=[0,1,0].
    # "side" : swap X↔Z → holes open from YZ-side face, dir=[1,0,0].
    # (knngraph uses only vertices + kNN so face winding doesn't matter.)
    face =p .get ("block_face","top")
    if face =="front":
        perm =[0 ,2 ,1 ]
        Vn =Vn [:,perm ];cp_pts =cp_pts [:,perm ];cp_dir =cp_dir [:,perm ]
    elif face =="side":
        perm =[2 ,1 ,0 ]
        Vn =Vn [:,perm ];cp_pts =cp_pts [:,perm ];cp_dir =cp_dir [:,perm ]

    return Vn ,Fn ,cp_pts ,cp_dir 


    # ---------------------------------------------------------------------------
    # serialise to corpus JSON
    # ---------------------------------------------------------------------------

def to_corpus_dict (part_nr ,V ,F ,cp_pts ,cp_dir ):
    return {
    "PartNr":part_nr ,
    "Graphic3d":{
    "Points":[{"X":float (x ),"Y":float (y ),"Z":float (z )}
    for (x ,y ,z )in V ],
    "Indices":[int (i )for tri in F for i in tri ],
    },
    "ConnectionPoints":[
    {"Point":{"X":float (px ),"Y":float (py ),"Z":float (pz )},
    "InsertDirection":{"X":float (dx ),"Y":float (dy ),"Z":float (dz )},
    "Name":f"T{i +1 }"}
    for i ,((px ,py ,pz ),(dx ,dy ,dz ))in enumerate (zip (cp_pts ,cp_dir ))
    ],
    }


def generate (out_dir ,n ,seed =0 ,prefix ="SYNTH"):
    os .makedirs (out_dir ,exist_ok =True )
    rng =np .random .default_rng (seed )
    written =0 
    for i in range (n ):
        p =sample_params (rng )
        try :
            V ,F ,cp_pts ,cp_dir =build_block (p ,rng )
        except Exception as exc :# noqa: BLE001
            logger .warning ("skip part %d (%s)",i ,exc )
            continue 
        part_nr =f"{prefix }_{seed :02d}_{i :05d}"
        obj =to_corpus_dict (part_nr ,V ,F ,cp_pts ,cp_dir )
        with open (os .path .join (out_dir ,part_nr +".json"),"w",
        encoding ="utf-8")as fh :
            json .dump (obj ,fh )
        written +=1 
        if (i +1 )%200 ==0 :
            logger .info ("  generated %d/%d",i +1 ,n )
    logger .info ("wrote %d synthetic part(s) -> %s",written ,out_dir )
    return written 


def main ():
    ap =argparse .ArgumentParser (description ="procedural synthetic terminal blocks")
    ap .add_argument ("--out",required =True ,help ="output corpus directory")
    ap .add_argument ("--n",type =int ,default =2000 ,help ="number of parts")
    ap .add_argument ("--seed",type =int ,default =0 )
    ap .add_argument ("--prefix",default ="SYNTH",help ="PartNr prefix")
    args =ap .parse_args ()
    logging .basicConfig (level =logging .INFO ,format ="%(message)s")
    generate (args .out ,args .n ,seed =args .seed ,prefix =args .prefix )


if __name__ =="__main__":
    main ()
