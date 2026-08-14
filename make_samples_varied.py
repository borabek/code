# Generalised synthetic parts: connection points appear as SEVERAL distinct
# feature types -- a smooth round boss, a narrow tall pin, and a raised
# rectangular pad -- instead of the single boss type in make_samples_hard.py.
# Non-CP DECOYS (sharp cones, recessed pits, box-like blocks) and zero-CP
# negatives are kept for discrimination.
#
# WHY: a model trained ten one CP shape only generalises to that shape. Real
# connectors expose connection points as varied raised terminals / pads / posts,
# so teaching the detector "a CP is one of several raised feature shapes (but NOT
# a sharp cone / pit / plain box corner)" broadens what it recognises ten real
# `.stp` parts -- the point of plan "make the model more general".
#
# Still synthetic, but covers a WIDER family of connection-feature geometry.
# Reuses the primitives from make_samples_hard.py (no duplication).
#
# Usage:
#   python make_samples_varied.py ./synthetic_varied 500
#   (or via the unified CLI: python make_samples.py ./synthetic_varied 500 --mode varied)
import json 
import os 
import sys 

import numpy as np 

from make_samples_hard import (grid_plate ,add_boss ,add_cone ,add_pit ,add_block ,
add_rail ,_place )


def add_pin (V ,c ,radius ,height ):
    """A narrow tall CP feature (terminal post): a boss with small footprint and
    extra height. Returns (apex_idx, apex_pt)."""
    return add_boss (V ,c ,radius *0.5 ,height *1.6 )


def add_rect_pad (V ,c ,half ,height ):
    """A raised rectangular CP feature (flat terminal pad). CP at the centre of
    the flat top. Returns (apex_idx, apex_pt)."""
    x ,y =V [:,0 ],V [:,1 ]
    inside =(np .abs (x -c [0 ])<=half )&(np .abs (y -c [1 ])<=half )
    V [inside ,2 ]=np .maximum (V [inside ,2 ],height )
    vi =int (np .argmin (np .linalg .norm (V [:,:2 ]-np .asarray (c ),axis =1 )))
    return vi ,V [vi ].copy ()


def add_socket (V ,c ,radius ,depth ):
    """A RECESSED circular socket / opening (cable-entry style hole): vertices
    within `radius` are pushed DOWN into a cup, CP at the centre (the opening),
    outward approach +z (the tool enters from above). This matches the recessed
    circular openings that real connectors actually expose -- the feature the
    raised-only models (boss/pin/rect) miss. Returns (centre_idx, centre_pt)."""
    d =np .linalg .norm (V [:,:2 ]-np .asarray (c ),axis =1 )
    s =max (radius ,1e-6 )/2.0 
    V [:,2 ]-=depth *np .exp (-(d **2 )/(2.0 *s *s ))
    vi =int (np .argmin (d ))
    return vi ,V [vi ].copy ()


    # CP feature types. 'socket' is the recessed circular opening real connectors
    # show; the rest are raised terminals/pads/posts. Mixing raised AND recessed
    # teaches the model both, broadening real-part coverage.
CP_TYPES =("boss","pin","rect","socket")


def _add_cp (kind ,V ,c ,rng ):
    if kind =="boss":
        return add_boss (V ,c ,rng .uniform (10.0 ,16.0 ),rng .uniform (7.0 ,12.0 ))
    if kind =="pin":
        return add_pin (V ,c ,rng .uniform (10.0 ,16.0 ),rng .uniform (7.0 ,12.0 ))
    if kind =="socket":
        return add_socket (V ,c ,rng .uniform (10.0 ,16.0 ),rng .uniform (6.0 ,12.0 ))
    return add_rect_pad (V ,c ,rng .uniform (6.0 ,10.0 ),rng .uniform (6.0 ,11.0 ))


def make_part (idx ,rng ):
    w =float (rng .uniform (100 ,160 ))
    h =float (rng .uniform (100 ,160 ))
    nx =int (rng .integers (44 ,56 ))
    ny =int (rng .integers (44 ,56 ))
    V ,F =grid_plate (nx ,ny ,w ,h )

    negative =rng .random ()<0.12 
    k_cp =0 if negative else int (rng .integers (2 ,5 ))
    k_decoy =int (rng .integers (2 ,5 ))
    radius =16.0 
    centers =_place (rng ,w ,h ,k_cp +k_decoy ,min_sep =max (3.0 *radius ,40.0 ))
    cp_centers ,decoy_centers =centers [:k_cp ],centers [k_cp :]

    if rng .random ()<0.5 :
        add_block (V ,(rng .uniform (0.25 *w ,0.75 *w ),rng .uniform (0.25 *h ,0.75 *h )),
        rng .uniform (8 ,16 ),rng .uniform (8 ,16 ),rng .uniform (6 ,12 ))
        # long raised side RAIL (decoy) ten ~60% of parts, hugging an edge (outside the
        # CP margin so it never overlaps a real CP). This is the feature the model
        # over-fired ten for the real connectors -- here it is unlabelled background.
    if rng .random ()<0.6 :
        x0 =(rng .uniform (0.03 ,0.09 )if rng .random ()<0.5 else rng .uniform (0.91 ,0.97 ))*w 
        add_rail (V ,x0 ,rng .uniform (2.0 ,4.0 ),rng .uniform (6.0 ,12.0 ))
    for di ,c in enumerate (decoy_centers ):
        if di %2 ==0 :
            add_cone (V ,c ,rng .uniform (9 ,15 ),rng .uniform (7 ,13 ))
        else :
            add_pit (V ,c ,rng .uniform (9 ,15 ),rng .uniform (6 ,11 ))

    cps =[]
    for ci ,c in enumerate (cp_centers ):
        kind =str (rng .choice (CP_TYPES ))
        _vi ,p =_add_cp (kind ,V ,c ,rng )
        cps .append ({
        "Index":ci ,
        "Name":f"X{idx }-{ci }-{kind }",
        "Point":{"X":float (p [0 ]),"Y":float (p [1 ]),"Z":float (p [2 ])},
        "InsertDirection":{"X":0.0 ,"Y":0.0 ,"Z":1.0 },
        })

    zmin ,zmax =float (V [:,2 ].min ()),float (V [:,2 ].max ())
    return {
    "PartNr":f"SYNV.{idx :04d}",
    "Graphic3d":{
    "Points":[{"X":float (x ),"Y":float (y ),"Z":float (z )}
    for x ,y ,z in V ],
    "Indices":F .ravel ().tolist (),
    },
    "BoundingBox":{"Dimension":{"X":w ,"Y":h ,"Z":zmax -zmin },
    "Location":{"X":0.0 ,"Y":0.0 ,"Z":zmin }},
    "ConnectionPoints":cps ,
    }


def main (argv =None ):
    argv =sys .argv [1 :]if argv is None else argv 
    out =argv [0 ]if len (argv )>0 else "synthetic_varied"
    n =int (argv [1 ])if len (argv )>1 else 500 
    seed =int (argv [2 ])if len (argv )>2 else 0 
    os .makedirs (out ,exist_ok =True )
    rng =np .random .default_rng (seed )
    n_cps =n_neg =0 
    for i in range (n ):
        obj =make_part (i ,rng )
        if not obj ["ConnectionPoints"]:
            n_neg +=1 
        n_cps +=len (obj ["ConnectionPoints"])
        with open (os .path .join (out ,f"part_{i :04d}.json"),"w",encoding ="utf-8")as fh :
            json .dump (obj ,fh )
    print (f"wrote {n } parts ({n_cps } CPs of types {CP_TYPES }, {n_neg } negatives) "
    f"-> {os .path .abspath (out )}")


def _selftest ():
    """Parts parse cleanly and several CP feature types (raised + recessed) appear."""
    rng =np .random .default_rng (2 )
    import json_dataset as jd 
    seen =set ()
    for k in range (12 ):
        obj =make_part (k ,rng )
        for cp in obj ["ConnectionPoints"]:
            seen .add (cp ["Name"].split ("-")[-1 ])
        part =jd .part_from_dict (obj )# must parse without error
        assert part .faces .shape [1 ]==3 and part .n_vertices >0 
    assert len (seen )>=3 ,f"expected several CP types, saw {seen }"
    print ("make_samples_varied selftest OK: CP types seen =",sorted (seen ))


if __name__ =="__main__":
    if len (sys .argv )==1 :
        _selftest ()
    else :
        main ()
