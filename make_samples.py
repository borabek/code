# Generate VALID synthetic ABB-style part JSONs (mesh + connection points).
# Single CLI entry point with three generators, chosen via --mode:
#   flat -> legacy plate with CPs at indistinguishable spots (NOT learnable by a
#           geometric model; kept only for comparison -- the model collapses ten it)
#   feat -> CP = a distinctive raised boss apex (LEARNABLE; make_samples_feat.py)
#   hard -> feat + non-CP decoys (cones/pits/box-blocks) + negatives so the model
#           also learns to DISCRIMINATE (make_samples_hard.py). DEFAULT.
# Synthetic -- good to prove the pipeline + emit real metrics, NOT a substitute
# for the real 479-file corpus.
#
#   python make_samples.py ./synth 400              # default mode=hard
#   python make_samples.py ./synth 400 --mode feat
#   python make_samples.py ./abb_samples 6 --mode flat
import json ,os ,sys 
import argparse 
import numpy as np 

# the learnable / discriminative generators (today's fix lives here); this file
# is the single CLI entry point and dispatches to them via --mode.
import make_samples_feat as _feat 
import make_samples_hard as _hard 
import make_samples_varied as _varied 


def grid_plate (nx ,ny ,w ,h ):
    """Triangulated open plate: (V, F). z=0 plane, size w x h mm."""
    xs =np .linspace (0 ,w ,nx )
    ys =np .linspace (0 ,h ,ny )
    X ,Y =np .meshgrid (xs ,ys )
    V =np .column_stack ([X .ravel (),Y .ravel (),np .zeros (X .size )])
    F =[]
    for j in range (ny -1 ):
        for i in range (nx -1 ):
            a =j *nx +i 
            b =a +1 
            c =a +nx 
            d =c +1 
            F .append ([a ,b ,c ])
            F .append ([b ,d ,c ])
    return V ,np .asarray (F ,dtype =np .int64 )


def make_part (idx ,rng ):
    nx =int (rng .integers (24 ,34 ))
    ny =int (rng .integers (24 ,34 ))
    w =float (rng .uniform (80 ,140 ))
    h =float (rng .uniform (80 ,140 ))
    V ,F =grid_plate (nx ,ny ,w ,h )
    # 2-3 connection points ten the plate, snapped to nearby vertices,
    # insert direction outward (+z) from the top face.
    k =int (rng .integers (2 ,4 ))
    cps =[]
    for c in range (k ):
        px =float (rng .uniform (0.15 *w ,0.85 *w ))
        py =float (rng .uniform (0.15 *h ,0.85 *h ))
        vi =int (np .argmin (np .linalg .norm (V [:,:2 ]-[px ,py ],axis =1 )))
        p =V [vi ]
        cps .append ({
        "Index":c ,
        "Name":f"X{idx }-{c }",
        "Point":{"X":float (p [0 ]),"Y":float (p [1 ]),"Z":float (p [2 ])},
        "InsertDirection":{"X":0.0 ,"Y":0.0 ,"Z":1.0 },
        })
    obj ={
    "PartNr":f"SYN.{idx :03d}",
    "Graphic3d":{
    "Points":[{"X":float (x ),"Y":float (y ),"Z":float (z )}for x ,y ,z in V ],
    "Indices":F .ravel ().tolist (),
    },
    "BoundingBox":{"Dimension":{"X":w ,"Y":h ,"Z":0.0 },
    "Location":{"X":0.0 ,"Y":0.0 ,"Z":0.0 }},
    "ConnectionPoints":cps ,
    }
    return obj 


    # the three generators, selectable via --mode:
    #   flat -> CPs at indistinguishable spots ten a flat plate (legacy; NOT learnable
    #           by a geometric model -- it collapses. Kept for reference/comparison.)
    #   feat -> CP = a distinctive raised boss apex (learnable; make_samples_feat)
    #   hard -> feat + non-CP decoys (cones/pits/box-blocks) + negatives, so the
    #           model also learns to DISCRIMINATE (make_samples_hard). Default.
_GENERATORS ={"flat":make_part ,"feat":_feat .make_part ,
"hard":_hard .make_part ,"varied":_varied .make_part }


def main (argv =None ):
    ap =argparse .ArgumentParser (
    description ="Generate synthetic ABB-style parts (mesh + connection points)")
    ap .add_argument ("out",nargs ="?",default ="abb_samples",help ="output directory")
    ap .add_argument ("n",nargs ="?",type =int ,default =6 ,help ="number of parts")
    ap .add_argument ("--mode",choices =list (_GENERATORS ),default ="hard",
    help ="flat=legacy non-learnable, feat=learnable bosses, "
    "hard=learnable+discriminative (default)")
    ap .add_argument ("--seed",type =int ,default =0 )
    args =ap .parse_args (sys .argv [1 :]if argv is None else argv )

    gen =_GENERATORS [args .mode ]
    os .makedirs (args .out ,exist_ok =True )
    rng =np .random .default_rng (args .seed )
    width =3 if args .mode =="flat"else 4 
    n_cps =0 
    for i in range (args .n ):
        obj =gen (i ,rng )
        n_cps +=len (obj ["ConnectionPoints"])
        with open (os .path .join (args .out ,f"part_{i :0{width }d}.json"),"w",
        encoding ="utf-8")as fh :
            json .dump (obj ,fh )
    print ("done (mode=%s): %d parts, %d connection points -> %s"
    %(args .mode ,args .n ,n_cps ,os .path .abspath (args .out )))


def _selftest ():
    """Every --mode generator builds a valid, loadable part."""
    import json_dataset as jd 
    rng =np .random .default_rng (0 )
    for mode ,gen in _GENERATORS .items ():
        part =jd .part_from_dict (gen (0 ,rng ))
        assert part .n_vertices >0 and part .faces .shape [1 ]==3 ,mode 
    print ("make_samples selftest OK: flat/feat/hard generators all valid")


if __name__ =="__main__":
    if len (sys .argv )==1 :
        _selftest ()
    else :
        main ()
