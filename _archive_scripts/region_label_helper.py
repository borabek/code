# -*- coding: utf-8 -*-
"""Region-based labelling aid so a NON-INTERACTIVE agent can label a mesh the way a human does in
label_tool.html: over-segment the mesh into geometric regions (neutral -- normals only, no model),
render them numbered from several views, and later apply a {region_id: class} decision map.

Purpose (2026-07-20): the user cannot label (no electrical domain knowledge). Before assuming *I*
can, this is used for a BLIND SELF-TEST: label expert-labelled parts without looking at their
ground truth, then score my labels against the experts'. High agreement -> my labels are usable;
low -> proven unusable, an electrical engineer is required. Measure, don't assume.

  python region_label_helper.py segment <obj_or_part>   -> regions + multi-view PNG + regions.json
  python region_label_helper.py apply <part> '{"3":1,...}'  -> writes labels.txt from a decision map
"""
import os ,sys ,json 
import numpy as np 
import matplotlib ;matplotlib .use ("Agg");import matplotlib .pyplot as plt 
from mpl_toolkits .mplot3d .art3d import Poly3DCollection 

OUT ="results/_agent_label"


def load_obj (p ):
    V ,F =[],[]
    for ln in open (p ):
        if ln .startswith ("v "):V .append ([float (x )for x in ln .split ()[1 :4 ]])
        elif ln .startswith ("f "):F .append ([int (t .split ("/")[0 ])-1 for t in ln .split ()[1 :4 ]])
    return np .asarray (V ,float ),np .asarray (F ,int )


def face_normals (V ,F ):
    n =np .cross (V [F [:,1 ]]-V [F [:,0 ]],V [F [:,2 ]]-V [F [:,0 ]])
    L =np .linalg .norm (n ,axis =1 ,keepdims =True );L [L <1e-12 ]=1 
    return n /L 


def vertex_normals (V ,F ):
    n =np .zeros_like (V );fn =np .cross (V [F [:,1 ]]-V [F [:,0 ]],V [F [:,2 ]]-V [F [:,0 ]])
    for k in range (3 ):np .add .at (n ,F [:,k ],fn )
    L =np .linalg .norm (n ,axis =1 ,keepdims =True );L [L <1e-12 ]=1 
    return n /L 


def cavity_of (V ,F ):
    """Same neutral concavity measure as label_tool.html (1 = deep recess)."""
    N =vertex_normals (V ,F );nbrs =[[]for _ in range (len (V ))]
    for a ,b ,c in F :
        nbrs [a ]+=[b ,c ];nbrs [b ]+=[a ,c ];nbrs [c ]+=[a ,b ]
    cv =np .zeros (len (V ))
    for i ,nb in enumerate (nbrs ):
        if not nb :continue 
        d =V [nb ]-V [i ];L =np .linalg .norm (d ,axis =1 );ok =L >1e-9 
        if ok .any ():cv [i ]=float (np .mean ((d [ok ]/L [ok ,None ])@N [i ]))
    for _ in range (2 ):
        t =cv .copy ()
        for i ,nb in enumerate (nbrs ):
            if nb :t [i ]=(cv [i ]+cv [nb ].sum ())/(1 +len (nb ))
        cv =t 
    return (cv -cv .min ())/max (float (cv .max ()-cv .min ()),1e-12 )


def segment_regions (V ,F ,cav =None ,min_faces =25 ,cav_thr =0.55 ):
    """Over-segment into units a human would click. Dihedral flood-fill alone FAILS ten these meshes
    (thesis remeshing smooths the sharp edges away -> everything merges into one region), so key each
    face by (dominant axis-aligned normal direction, is-it-recessed) and take connected components of
    equal keys. That separates 'top face' from 'the recess inside the top face' from 'left side',
    which is exactly the granularity of a wire opening / clamp mouth. Neutral: geometry only."""
    fn =face_normals (V ,F )
    if cav is None :cav =cavity_of (V ,F )
    fcav =cav [F ].mean (1 )
    bucket =np .argmax (np .abs (fn ),axis =1 )*2 +(fn [np .arange (len (F )),np .argmax (np .abs (fn ),axis =1 )]<0 )
    key =bucket *2 +(fcav >cav_thr )
    edge ={}
    for fi ,(a ,b ,c )in enumerate (F ):
        for u ,v in ((a ,b ),(b ,c ),(c ,a )):
            edge .setdefault ((min (u ,v ),max (u ,v )),[]).append (fi )
    adj =[[]for _ in range (len (F ))]
    for fs in edge .values ():
        if len (fs )==2 :
            adj [fs [0 ]].append (fs [1 ]);adj [fs [1 ]].append (fs [0 ])
    reg =np .full (len (F ),-1 ,int );rid =0 
    for s in range (len (F )):
        if reg [s ]!=-1 :continue 
        stack =[s ];reg [s ]=rid 
        while stack :
            f =stack .pop ()
            for g in adj [f ]:
                if reg [g ]==-1 and key [g ]==key [f ]:
                    reg [g ]=rid ;stack .append (g )
        rid +=1 
        # absorb tiny regions into the largest adjacent region
    for _ in range (3 ):
        counts =np .bincount (reg ,minlength =rid )
        small ={r for r in range (rid )if 0 <counts [r ]<min_faces }
        if not small :break 
        for f in range (len (F )):
            if reg [f ]in small :
                cand =[reg [g ]for g in adj [f ]if reg [g ]not in small ]
                if cand :reg [f ]=max (set (cand ),key =cand .count )
                # renumber compactly
    uniq ={r :i for i ,r in enumerate (sorted (set (reg .tolist ())))}
    reg =np .array ([uniq [r ]for r in reg ])
    return reg ,fn 


def render (V ,F ,reg ,cav ,out ,title ):
    """Two rows: (top) pure CAVITY shading -- shows where the recesses/openings physically are;
    (bottom) the numbered REGION map. Views are axis-aligned to the part's own bbox so the big face
    is seen flat-ten, plus the two edge faces (a terminal block is a thin slice: the wire openings
    live ten the edges/recesses, so both matter)."""
    nreg =reg .max ()+1 
    rng_ =np .random .RandomState (7 )
    cols =rng_ .rand (nreg ,3 )*0.55 +0.4 
    ext =V .max (0 )-V .min (0 );ctr =V .mean (0 );rr =ext .max ()
    thin =int (np .argmin (ext ))
    # look down each principal axis: thin axis first (the informative "profile" view)
    axviews ={0 :(0 ,0 ),1 :(0 ,-90 ),2 :(90 ,-90 )}
    order =[thin ]+[i for i in range (3 )if i !=thin ]
    fcav =cav [F ].mean (1 )
    fig =plt .figure (figsize =(16 ,10.5 ))
    for col ,ax_i in enumerate (order ):
        ev ,az =axviews [ax_i ]
        for row in range (2 ):
            ax =fig .add_subplot (2 ,3 ,row *3 +col +1 ,projection ="3d")
            if row ==0 :
                g =1 -0.85 *fcav # cavity only: deep recess = dark
                fc =np .stack ([g ,g ,g ],1 )
            else :
                fc =np .clip (cols [reg ]*(1 -0.35 *fcav )[:,None ],0 ,1 )
            ax .add_collection3d (Poly3DCollection (V [F ],facecolors =fc ,edgecolors ="none"))
            for st ,lo ,hi in [(ax .set_xlim ,ctr [0 ]-rr /2 ,ctr [0 ]+rr /2 ),(ax .set_ylim ,ctr [1 ]-rr /2 ,ctr [1 ]+rr /2 ),
            (ax .set_zlim ,ctr [2 ]-rr /2 ,ctr [2 ]+rr /2 )]:st (lo ,hi )
            try :ax .set_box_aspect (ext )
            except Exception :pass 
            ax .view_init (elev =ev ,azim =az );ax .set_axis_off ()
            if row ==1 :
                for r in np .argsort (-np .bincount (reg ,minlength =nreg ))[:22 ]:
                    m =reg ==r 
                    if not m .any ():continue 
                    c =V [F [m ]].reshape (-1 ,3 ).mean (0 )
                    ax .text (c [0 ],c [1 ],c [2 ],str (r ),fontsize =7 ,color ="k",
                    bbox =dict (fc ="white",alpha =0.8 ,pad =0.5 ,lw =0 ))
            lbl ="XYZ"[ax_i ]
            ax .set_title (("CAVITY down "if row ==0 else "REGIONS down ")+lbl +
            (" (thin axis = profile)"if ax_i ==thin else ""),fontsize =9 )
    fig .suptitle (title +f"   bbox {np .round (ext ,1 )} mm",fontsize =11 )
    plt .tight_layout (rect =[0 ,0 ,1 ,0.96 ]);plt .savefig (out ,dpi =105 ,bbox_inches ="tight");plt .close ()


def region_table (V ,F ,reg ,cav ):
    nreg =reg .max ()+1 ;ext =V .max (0 )-V .min (0 );lo =V .min (0 )
    rows =[]
    for r in range (nreg ):
        m =reg ==r 
        if not m .any ():continue 
        vi =np .unique (F [m ])
        c =V [vi ].mean (0 );rel =(c -lo )/np .maximum (ext ,1e-9 )
        rows .append ({"region":int (r ),"n_faces":int (m .sum ()),"n_verts":int (len (vi )),
        "centroid":[round (float (x ),1 )for x in c ],
        "rel_pos_xyz":[round (float (x ),2 )for x in rel ],
        "cavity_mean":round (float (cav [vi ].mean ()),2 ),
        "area_frac":round (float (m .sum ()/len (F )),3 )})
    rows .sort (key =lambda d :-d ["n_faces"])
    return rows 


def main ():
    os .makedirs (OUT ,exist_ok =True )
    cmd =sys .argv [1 ]
    if cmd =="segment":
        src =sys .argv [2 ]
        pid =os .path .basename (src ).replace (".obj","")
        V ,F =load_obj (src )
        cav =cavity_of (V ,F )
        reg ,_ =segment_regions (V ,F ,cav )
        render (V ,F ,reg ,cav ,f"{OUT }/{pid }_regions.png",f"{pid }: {reg .max ()+1 } regions (neutral geometry)")
        rows =region_table (V ,F ,reg ,cav )
        json .dump ({"part":pid ,"src":src ,"n_verts":int (len (V )),"n_regions":int (reg .max ()+1 ),
        "regions":rows ,"face_region":reg .tolist ()},
        open (f"{OUT }/{pid }_regions.json","w"))
        print (f"{pid }: {len (V )} verts, {reg .max ()+1 } regions -> {OUT }/{pid }_regions.png")
        print (f"{'reg':>4} {'faces':>6} {'verts':>6} {'cav':>5}  rel_xyz (0..1 in bbox)")
        for d in rows [:20 ]:
            print (f"{d ['region']:>4} {d ['n_faces']:>6} {d ['n_verts']:>6} {d ['cavity_mean']:>5.2f}  {d ['rel_pos_xyz']}")
    elif cmd =="apply":
        pid =sys .argv [2 ];dmap ={int (k ):int (v )for k ,v in json .loads (sys .argv [3 ]).items ()}
        meta =json .load (open (f"{OUT }/{pid }_regions.json"))
        V ,F =load_obj (meta ["src"]);reg =np .array (meta ["face_region"])
        lab =np .zeros (len (V ),int )# default Housing
        for r ,cl in dmap .items ():
            vi =np .unique (F [reg ==r ]);lab [vi ]=cl 
        out =f"{OUT }/{pid }.mylabels.txt"
        open (out ,"w").write ("\n".join (map (str ,lab .tolist ()))+"\n")
        print (f"{pid }: wrote {len (lab )} labels -> {out }")
        print ("  class counts:",{c :int ((lab ==c ).sum ())for c in range (5 )})
    else :
        print (__doc__ )


if __name__ =="__main__":
    main ()
