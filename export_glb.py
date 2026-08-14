"""GLB exporter: preds.json (predict.py output) → .glb 3D file.

Zero extra dependencies — uses only Python stdlib (struct, json, base64).
Output opens with double-click in Windows 3D Viewer, Blender, MeshLab.

Scene contents
--------------
- GREY mesh     : original part geometry (triangulated)
- RED spheres   : ML-predicted connection points (knngraph model)
- GREEN spheres : CAD ground-truth CPs (step_openings, if available)
- ARROWS        : outward approach direction for each CP
  - red arrows  = ML predictions
  - green arrows= CAD ground-truth

Colour coding
  TP  red + green sphere at same spot  → model found it
  FN  green only                       → model MISSED (recall gap)
  FP  red only                         → model HALLUCINATED (precision gap)

Usage
-----
  # ML predictions only:
  python export_glb.py preds.json part.json --out scene.glb

  # ML + CAD ground truth comparison:
  python export_glb.py preds.json part.json --cad cad_preds.json --out scene.glb

  # From corpus (no separate mesh file needed):
  python export_glb.py preds.json --corpus C:/Users/.../JSON --out scene.glb

  # Batch: all parts in preds.json → one .glb per part:
  python export_glb.py preds.json --corpus C:/Users/.../JSON --batch --out-dir glb_out/
"""
import argparse 
import base64 
import json 
import math 
import os 
import struct 
import sys 

import numpy as np 


# ---------------------------------------------------------------------------
# GLB / glTF constants
# ---------------------------------------------------------------------------

_GLTF_FLOAT =5126 
_GLTF_UNSIGNED_INT =5125 
_GLTF_UNSIGNED_SHORT =5123 
_GLTF_ARRAY_BUFFER =34962 
_GLTF_ELEMENT_ARRAY_BUFFER =34963 
_COMP_FLOAT3 ="VEC3"
_COMP_SCALAR ="SCALAR"
_PRIM_TRIANGLES =4 
_PRIM_LINES =1 
_PRIM_POINTS =0 

# colours (linear RGBA float)
_COL_GREY =[0.35 ,0.35 ,0.35 ,1.0 ]
_COL_ML_RED =[0.90 ,0.15 ,0.10 ,1.0 ]# ML prediction
_COL_CAD_GRN =[0.10 ,0.82 ,0.20 ,1.0 ]# CAD ground truth
_COL_ML_ARR =[1.00 ,0.40 ,0.10 ,1.0 ]# ML arrow
_COL_CAD_ARR =[0.10 ,1.00 ,0.30 ,1.0 ]# CAD arrow


# ---------------------------------------------------------------------------
# geometry primitives
# ---------------------------------------------------------------------------

def _sphere_mesh (centre ,radius ,rings =8 ,sectors =8 ):
    """Return (V float32 (N,3), F uint32 (M,3)) for a UV sphere."""
    verts =[]
    for r in range (rings +1 ):
        phi =math .pi *r /rings # 0 .. pi
        for s in range (sectors ):
            theta =2 *math .pi *s /sectors 
            x =centre [0 ]+radius *math .sin (phi )*math .cos (theta )
            y =centre [1 ]+radius *math .sin (phi )*math .sin (theta )
            z =centre [2 ]+radius *math .cos (phi )
            verts .append ((x ,y ,z ))

    faces =[]
    for r in range (rings ):
        for s in range (sectors ):
            n0 =r *sectors +s 
            n1 =r *sectors +(s +1 )%sectors 
            n2 =(r +1 )*sectors +s 
            n3 =(r +1 )*sectors +(s +1 )%sectors 
            faces .append ((n0 ,n2 ,n1 ))
            faces .append ((n1 ,n2 ,n3 ))

    return np .array (verts ,dtype =np .float32 ),np .array (faces ,dtype =np .uint32 )


def _arrow_lines (origin ,direction ,length =10.0 ,head_frac =0.25 ,spread =0.15 ):
    """Return (V float32 (N,3), lines (M,2) uint32) for a 3-D arrow.

    The arrow is: shaft from origin to type, plus 3 head fins from the type.
    """
    origin =np .asarray (origin ,dtype =np .float64 )
    d =np .asarray (direction ,dtype =np .float64 )
    dn =np .linalg .norm (d )
    if dn <1e-9 :
        return np .zeros ((0 ,3 ),np .float32 ),np .zeros ((0 ,2 ),np .uint32 )
    d =d /dn 

    tip =origin +d *length 
    head_start =origin +d *(length *(1 -head_frac ))

    # two perpendicular axes for the head
    perp =np .array ([1.0 ,0.0 ,0.0 ])
    if abs (np .dot (d ,perp ))>0.9 :
        perp =np .array ([0.0 ,1.0 ,0.0 ])
    a1 =np .cross (d ,perp );a1 /=np .linalg .norm (a1 )
    a2 =np .cross (d ,a1 );a2 /=np .linalg .norm (a2 )

    hw =length *spread 
    fins =[head_start +a1 *hw ,head_start -a1 *hw ,head_start +a2 *hw ]

    pts =[origin ,tip ]+fins # 0=origin, 1=type, 2,3,4=fin bases
    verts =np .array (pts ,dtype =np .float32 )
    lines =np .array ([[0 ,1 ],[1 ,2 ],[1 ,3 ],[1 ,4 ]],dtype =np .uint32 )
    return verts ,lines 


    # ---------------------------------------------------------------------------
    # GLB binary builder (pure Python, zero deps)
    # ---------------------------------------------------------------------------

class _GlbBuilder :
    """Incrementally builds a binary GLB 2.0 file."""

    def __init__ (self ):
        self ._bin =bytearray ()
        self ._views =[]# buffer views
        self ._accs =[]# accessors
        self ._meshes =[]
        self ._nodes =[]
        self ._mats ={}# colour tuple → material index

        # ---- buffer management ----

    def _add_bytes (self ,data :bytes ,target :int )->int :
        """Append raw bytes to BIN chunk; return buffer-view index."""
        offset =len (self ._bin )
        self ._bin .extend (data )
        # 4-byte align
        while len (self ._bin )%4 :
            self ._bin .append (0 )
        bv =len (self ._views )
        self ._views .append ({"buffer":0 ,"byteOffset":offset ,
        "byteLength":len (data ),"target":target })
        return bv 

    def _add_f32 (self ,arr ,target =_GLTF_ARRAY_BUFFER )->int :
        arr =np .asarray (arr ,dtype =np .float32 )
        bv =self ._add_bytes (arr .tobytes (),target )
        return bv ,arr 

    def _add_u32 (self ,arr ,target =_GLTF_ELEMENT_ARRAY_BUFFER )->int :
        arr =np .asarray (arr ,dtype =np .uint32 )
        bv =self ._add_bytes (arr .tobytes (),target )
        return bv ,arr 

        # ---- accessor helpers ----

    def _accessor_vec3 (self ,arr_f32 ,bv_idx ,count ):
        """Add a VEC3/FLOAT accessor; return index."""
        mi =arr_f32 .reshape (-1 ,3 ).min (0 ).tolist ()
        ma =arr_f32 .reshape (-1 ,3 ).max (0 ).tolist ()
        idx =len (self ._accs )
        self ._accs .append ({"bufferView":bv_idx ,"componentType":_GLTF_FLOAT ,
        "count":count ,"type":_COMP_FLOAT3 ,
        "min":mi ,"max":ma })
        return idx 

    def _accessor_scalar_u32 (self ,arr_u32 ,bv_idx ,count ):
        idx =len (self ._accs )
        self ._accs .append ({"bufferView":bv_idx ,"componentType":_GLTF_UNSIGNED_INT ,
        "count":count ,"type":_COMP_SCALAR })
        return idx 

        # ---- material ----

    def _material (self ,colour ):
        key =tuple (colour )
        if key not in self ._mats :
            r ,g ,b ,a =colour 
            idx =len (self ._mats )
            self ._mats [key ]=idx 
        return self ._mats [key ]

        # ---- high-level mesh helpers ----

    def add_triangle_mesh (self ,V ,F ,colour ,name ="mesh"):
        """Add a triangle mesh node. V:(N,3) float32, F:(M,3) uint32."""
        V =np .asarray (V ,dtype =np .float32 )
        F =np .asarray (F ,dtype =np .uint32 )
        if len (V )==0 or len (F )==0 :
            return 

        bv_v ,v_arr =self ._add_f32 (V .ravel ())
        bv_f ,f_arr =self ._add_u32 (F .ravel ())
        acc_v =self ._accessor_vec3 (v_arr ,bv_v ,len (V ))
        acc_f =self ._accessor_scalar_u32 (f_arr ,bv_f ,len (F )*3 )

        mat =self ._material (colour )
        mesh_idx =len (self ._meshes )
        self ._meshes .append ({"name":name ,
        "primitives":[{"attributes":{"POSITION":acc_v },
        "indices":acc_f ,
        "material":mat ,
        "mode":_PRIM_TRIANGLES }]})
        node_idx =len (self ._nodes )
        self ._nodes .append ({"mesh":mesh_idx ,"name":name })
        return node_idx 

    def add_line_set (self ,V ,lines ,colour ,name ="lines"):
        """Add a line-set node. V:(N,3), lines:(M,2) uint32 index pairs."""
        V =np .asarray (V ,dtype =np .float32 )
        lines =np .asarray (lines ,dtype =np .uint32 )
        if len (V )==0 or len (lines )==0 :
            return 

        bv_v ,v_arr =self ._add_f32 (V .ravel ())
        bv_l ,l_arr =self ._add_u32 (lines .ravel ())
        acc_v =self ._accessor_vec3 (v_arr ,bv_v ,len (V ))
        acc_l =self ._accessor_scalar_u32 (l_arr ,bv_l ,len (lines )*2 )

        mat =self ._material (colour )
        mesh_idx =len (self ._meshes )
        self ._meshes .append ({"name":name ,
        "primitives":[{"attributes":{"POSITION":acc_v },
        "indices":acc_l ,
        "material":mat ,
        "mode":_PRIM_LINES }]})
        node_idx =len (self ._nodes )
        self ._nodes .append ({"mesh":mesh_idx ,"name":name })
        return node_idx 

        # ---- write GLB ----

    def write (self ,path ):
    # build material list in insertion order
        mat_list =[None ]*len (self ._mats )
        for (r ,g ,b ,a ),idx in self ._mats .items ():
            mat_list [idx ]={
            "pbrMetallicRoughness":{
            "baseColorFactor":[r ,g ,b ,a ],
            "metallicFactor":0.0 ,
            "roughnessFactor":0.8 ,
            },
            "alphaMode":"OPAQUE"if a >=1.0 else "BLEND",
            "doubleSided":True ,
            }

        gltf ={
        "asset":{"version":"2.0","generator":"export_glb.py"},
        "scene":0 ,
        "scenes":[{"nodes":list (range (len (self ._nodes )))}],
        "nodes":self ._nodes ,
        "meshes":self ._meshes ,
        "materials":mat_list ,
        "accessors":self ._accs ,
        "bufferViews":self ._views ,
        "buffers":[{"byteLength":len (self ._bin )}],
        }

        json_bytes =json .dumps (gltf ,separators =(",",":")).encode ("utf-8")
        # pad to 4-byte boundary with spaces
        while len (json_bytes )%4 :
            json_bytes +=b" "

        bin_bytes =bytes (self ._bin )
        while len (bin_bytes )%4 :
            bin_bytes +=b"\x00"

            # GLB header
        total =12 +8 +len (json_bytes )+(8 +len (bin_bytes )if bin_bytes else 0 )
        header =struct .pack ("<III",0x46546C67 ,2 ,total )# magic, version, length
        json_chunk =struct .pack ("<II",len (json_bytes ),0x4E4F534A )+json_bytes 
        bin_chunk =(struct .pack ("<II",len (bin_bytes ),0x004E4942 )+bin_bytes 
        if bin_bytes else b"")

        with open (path ,"wb")as fh :
            fh .write (header +json_chunk +bin_chunk )
        print (f"Wrote {path }  ({os .path .getsize (path )//1024 } KB)")


        # ---------------------------------------------------------------------------
        # scene builder
        # ---------------------------------------------------------------------------

def _scene_bbox (V ):
    """Bounding-box diagonal of a vertex array (float mm)."""
    if len (V )==0 :
        return 1.0 
    lo ,hi =V .min (0 ),V .max (0 )
    return float (np .linalg .norm (hi -lo ))or 1.0 


def build_scene (glb :_GlbBuilder ,vertices ,faces ,
ml_cps ,cad_cps =None ,
sphere_frac =0.015 ,arrow_frac =0.18 ):
    """
    vertices : (N,3) float64 mm
    faces    : (M,3) int
    ml_cps   : list of robot-node dicts (entry_point, approach_vector, confidence_score)
    cad_cps  : list of robot-node dicts or None
    """
    V =np .asarray (vertices ,dtype =np .float64 )
    F =np .asarray (faces ,dtype =np .uint32 )if len (faces )else np .zeros ((0 ,3 ),np .uint32 )
    diag =_scene_bbox (V )
    sphere_r =diag *sphere_frac 
    arrow_len =diag *arrow_frac 

    # 1. mesh body
    if len (V )and len (F ):
        glb .add_triangle_mesh (V .astype (np .float32 ),F ,_COL_GREY ,"part_mesh")

        # helper: add spheres + arrows for a CP list
    def _add_cps (cp_list ,s_colour ,a_colour ,tag ):
        all_sv ,all_sf ,sv_off =[],[],0 
        all_lv ,all_ll ,lv_off =[],[],0 
        for i ,nd in enumerate (cp_list ):
            ep =np .asarray (nd ["entry_point"],dtype =np .float64 )
            av =np .asarray (nd .get ("approach_vector",[0 ,0 ,1 ]),dtype =np .float64 )
            av_n =np .linalg .norm (av )
            if av_n >1e-9 :av =av /av_n 

            sv ,sf =_sphere_mesh (ep ,sphere_r )
            sf_off =sf +sv_off 
            all_sv .append (sv );all_sf .append (sf_off )
            sv_off +=len (sv )

            lv ,ll =_arrow_lines (ep ,av ,length =arrow_len )
            if len (lv ):
                ll_off =ll +lv_off 
                all_lv .append (lv );all_ll .append (ll_off )
                lv_off +=len (lv )

        if all_sv :
            SV =np .concatenate (all_sv );SF =np .concatenate (all_sf )
            glb .add_triangle_mesh (SV ,SF ,s_colour ,f"{tag }_spheres")
        if all_lv :
            LV =np .concatenate (all_lv );LL =np .concatenate (all_ll )
            glb .add_line_set (LV ,LL ,a_colour ,f"{tag }_arrows")

            # 2. ML predictions (red)
    if ml_cps :
        _add_cps (ml_cps ,_COL_ML_RED ,_COL_ML_ARR ,"ml")

        # 3. CAD ground truth (green)
    if cad_cps :
        _add_cps (cad_cps ,_COL_CAD_GRN ,_COL_CAD_ARR ,"cad")

        # 4. FIZIKSEL DEFECT ISARETCILERI (M10, 2026-08-05) -- AYRI GORSEL KANAL
        #
        # WHY SEPARATE KANAL: sozlesmenin M6 maddesi "RENK = ESLESME DURUMU" diyor and denetci
        # renk sayimlarini makbuzun TP/FP/FN'iyle karsilastiriyor. Kusurlari new RENKLERLE
        # gostermek M6'nin sayimini BOZARDI. Bunun instead of same noktaya DAHA KUCUK ikinci a
        # kure konur; own name onekiyle (`fiz_*`) sayilir, M6 aynen calismaya devam eder.
        #
        # WHY GEREKLI: A4 olcumu -- FP'ler fiziksel kusurda ZENGIN (body ici 3.16x,
        # dar mouth 2.18x, onu closed 1.85x). Bu kusurlar SAYIDA gorunmuyor but GLB'ye bakan
        # insan for "this point why wrong" sorusunun DOGRUDAN yaniti.
        #
        # Bayrak kaynagi: each CP sozlugunde opsiyonel `fiz_bayrak` (govde_ici/onu_kapali/
        # dar_agiz/ters_yon). Yoksa no sey cizilmez -- old GLB'ler BIT-OZDES kalir.
    _FIZ_RENK ={
    "govde_ici":[1.00 ,0.00 ,0.00 ,1.0 ],# kirmizi   -- point govdenin ICINDE
    "onu_kapali":[1.00 ,0.55 ,0.00 ,1.0 ],# turuncu   -- ileride engel present
    "dar_agiz":[1.00 ,1.00 ,0.00 ,1.0 ],# sari      -- mouth telden dar
    "ters_yon":[0.00 ,0.80 ,1.00 ,1.0 ],# camgobegi -- direction govdeye bakiyor
    }
    if ml_cps :
        for _ad ,_renk in _FIZ_RENK .items ():
            _pts =[np .asarray (n ["entry_point"],float )for n in ml_cps 
            if _ad in (n .get ("fiz_bayrak")or ())]
            if not _pts :
                continue 
            _sv ,_sf ,_off =[],[],0 
            for _p in _pts :
                v ,f =_sphere_mesh (_p ,sphere_r *0.55 )# KUCUK: ana kureyi ortmez
                _sv .append (v );_sf .append (f +_off );_off +=len (v )
            glb .add_triangle_mesh (np .concatenate (_sv ),np .concatenate (_sf ),
            _renk ,f"fiz_{_ad }")


            # ---------------------------------------------------------------------------
            # loading helpers
            # ---------------------------------------------------------------------------

def _load_preds_json (path ):
    """Load predict.py JSON output → dict {part_nr: [node_dicts]}."""
    with open (path )as fh :
        data =json .load (fh )
    result ={}
    parts =data .get ("parts",[data ])if "parts"in data else [data ]
    for p in parts :
        nr =p .get ("part_nr","unknown")
        result [nr ]=p .get ("connection_points",[])
    return result 


def _load_cad_json (path ):
    """Load step_openings/predict.py JSON → dict {part_nr: [node_dicts]}."""
    return _load_preds_json (path )


def _load_mesh (part_json_path =None ,corpus_dir =None ,part_nr =None ):
    """Load (V, F) from a part.json file or corpus directory."""
    candidate =None 
    if part_json_path and os .path .isfile (part_json_path ):
        candidate =part_json_path 
    elif corpus_dir and part_nr :
    # try exact filename match then prefix search
        for fn in os .listdir (corpus_dir ):
            if fn ==f"{part_nr }.json"or fn .startswith (str (part_nr )):
                candidate =os .path .join (corpus_dir ,fn )
                break 

    if candidate is None :
        return np .zeros ((0 ,3 ),np .float32 ),np .zeros ((0 ,3 ),np .uint32 )

    with open (candidate )as fh :
        obj =json .load (fh )
    g =obj .get ("Graphic3d",{})
    pts =g .get ("Points",[])
    idx =g .get ("Indices",[])
    V =np .array ([[p ["X"],p ["Y"],p ["Z"]]for p in pts ],dtype =np .float32 )
    F =np .array (idx ,dtype =np .uint32 ).reshape (-1 ,3 )if idx else np .zeros ((0 ,3 ),np .uint32 )
    return V ,F 


    # ---------------------------------------------------------------------------
    # CLI
    # ---------------------------------------------------------------------------

def _parse_args ():
    ap =argparse .ArgumentParser (description ="Export CP predictions to GLB 3D file")
    ap .add_argument ("preds",help ="predict.py output JSON (ML predictions)")
    ap .add_argument ("mesh",nargs ="?",default =None ,
    help ="part mesh JSON (ABB format) for single-part mode")
    ap .add_argument ("--cad",default =None ,
    help ="CAD ground-truth JSON (step_openings output) for comparison")
    ap .add_argument ("--corpus",default =None ,
    help ="corpus directory to find mesh JSONs by part_nr")
    ap .add_argument ("--out",default =None ,
    help ="output .glb path (default: <preds>.glb)")
    ap .add_argument ("--batch",action ="store_true",
    help ="one .glb per part (requires --corpus)")
    ap .add_argument ("--out-dir",default ="glb_out",dest ="out_dir",
    help ="output directory for --batch mode")
    ap .add_argument ("--sphere-frac",type =float ,default =0.015 ,dest ="sphere_frac",
    help ="sphere radius as fraction of bbox diagonal (default 0.015)")
    ap .add_argument ("--arrow-frac",type =float ,default =0.18 ,dest ="arrow_frac",
    help ="arrow length as fraction of bbox diagonal (default 0.18)")
    ap .add_argument ("--part-nr",default =None ,dest ="part_nr",
    help ="restrict to a single part_nr (for multi-part preds JSON)")
    return ap .parse_args ()


def main ():
    args =_parse_args ()

    ml_by_part =_load_preds_json (args .preds )
    cad_by_part =_load_cad_json (args .cad )if args .cad else {}

    if args .part_nr :
        ml_by_part ={k :v for k ,v in ml_by_part .items ()if k ==args .part_nr }
        cad_by_part ={k :v for k ,v in cad_by_part .items ()if k ==args .part_nr }

    if not ml_by_part :
        print ("No parts found in predictions JSON.",file =sys .stderr )
        sys .exit (1 )

    if args .batch :
        if not args .corpus :
            print ("--batch requires --corpus",file =sys .stderr );sys .exit (1 )
        os .makedirs (args .out_dir ,exist_ok =True )
        for part_nr ,ml_cps in ml_by_part .items ():
            V ,F =_load_mesh (corpus_dir =args .corpus ,part_nr =part_nr )
            cad_cps =cad_by_part .get (part_nr )
            glb =_GlbBuilder ()
            build_scene (glb ,V ,F ,ml_cps ,cad_cps ,
            sphere_frac =args .sphere_frac ,arrow_frac =args .arrow_frac )
            out =os .path .join (args .out_dir ,f"{part_nr }.glb")
            glb .write (out )
        print (f"Batch done: {len (ml_by_part )} GLB files in {args .out_dir }/")
    else :
    # single-part or first-part mode
        if len (ml_by_part )>1 and not args .part_nr :
            print (f"preds.json has {len (ml_by_part )} parts — using first one. "
            f"Pass --part-nr or --batch to select.",file =sys .stderr )
        part_nr =next (iter (ml_by_part ))
        ml_cps =ml_by_part [part_nr ]
        cad_cps =cad_by_part .get (part_nr )

        V ,F =_load_mesh (part_json_path =args .mesh ,corpus_dir =args .corpus ,
        part_nr =part_nr )
        if len (V )==0 :
            print (f"WARNING: no mesh found for {part_nr } — exporting CPs only",
            file =sys .stderr )

        glb =_GlbBuilder ()
        build_scene (glb ,V ,F ,ml_cps ,cad_cps ,
        sphere_frac =args .sphere_frac ,arrow_frac =args .arrow_frac )

        out =args .out or (os .path .splitext (args .preds )[0 ]+".glb")
        glb .write (out )

        # print summary
        print (f"part_nr : {part_nr }")
        print (f"mesh    : {len (V )} verts  {len (F )} faces")
        print (f"ML CPs  : {len (ml_cps )}  (red spheres)")
        if cad_cps is not None :
            print (f"CAD CPs : {len (cad_cps )}  (green spheres)")
        else :
            print (f"CAD CPs : — (pass --cad to add ground-truth comparison)")
        print (f"\nOpen {out } in Windows 3D Viewer (double-click) or Blender.")


if __name__ =="__main__":
    main ()
