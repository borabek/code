# -*- coding: utf-8 -*-
"""Thesis-prescribed uniform remeshing (Masterarbeit_Scheffler 5.1.2): normalise EVERY
mesh -- coarse JSON export AND fine WSCAD STEP -- to a uniform ~6000-vertex 2-manifold via
isotropic remeshing, so their tessellation (and thus the kNN/intrinsic features) MATCH.
This is what closes the JSON->STEP domain gap that flat midpoint-subdivision did NOT.
"""
import os ,hashlib 
import numpy as np 
import pymeshlab 


REMESH_CACHE =os .environ .get ("CP_REMESH_CACHE","results/remesh_cache")


def _cache_key (V ,F ,target ,iterations ):
    hh =hashlib .sha256 ()
    hh .update (np .ascontiguousarray (np .asarray (V ,np .float64 ).round (6 )).tobytes ())
    hh .update (np .ascontiguousarray (np .asarray (F ,np .int64 )).tobytes ())
    hh .update (f"|{int (target )}|{int (iterations )}|v1".encode ())
    return hh .hexdigest ()[:24 ]


ADAPTIVE_DENSITY =float (os .environ .get ("CP_ADAPTIVE_DENSITY","0"))# 0 = closed (sabit hedef)


def adaptive_target (V ,F ,density =0.790 ,lo =4000 ,hi =20000 ):
    """Tezin izotropik kurali UNIFORM YOGUNLUK der; sabit vertex SAYISI parts different
    boyuttayken bunu ihlal eder (measured: very-CP parcalari %51 more large but same 6000 vertex ->
    yogunluk 0.531 vs 0.790 /mm2). Hedef alanla olceklenir."""
    import trimesh as _tm 
    area =float (_tm .Trimesh (np .asarray (V ,float ),np .asarray (F ,np .int64 ),process =False ).area )
    return int (np .clip (area *density ,lo ,hi ))


def remesh_uniform (V ,F ,target =6000 ,iterations =8 ):
    """Isotropic-remesh (V,F) to ~target vertices with near-uniform edge length.

    DETERMINIZM (2026-07-28): pymeshlab'in izotropik remesh'i SURECLER ARASI DETERMINISTIK DEGIL --
    same STEP, ayri sureclerde different vertex kumesi ureti (4 surecte 3 same 1 different; measured
    `S1_mesh` hash'i with). Bu, asagi akista different CP sayisina path aciyordu (WEI.1547610000: 4/5/4).
    Cozum: sonucu DISKE onbellekle -- first hesap kazanir and that part for sonsuza up to same mesh
    returns. Yan fayda: tekrar kosularda remesh maliyeti sifir.
    Kapatmak for: CP_REMESH_CACHE=0 (that zaman old, oynak davranis geri gelir).
    Onbellek float32/int32 saklar; DEGERLER HER IKI YOLDA DA AYNI olsun diye hesaplanan sonuc da
    same yuvarlama uygulanarak dondurulur (otherwise first run sonrakilerden different olurdu).
    """
    if ADAPTIVE_DENSITY >0 :
        target =adaptive_target (V ,F ,ADAPTIVE_DENSITY )
    use_cache =REMESH_CACHE not in ("0","","off")
    if use_cache :
        key =_cache_key (V ,F ,target ,iterations )
        fp =os .path .join (REMESH_CACHE ,key +".npz")
        if os .path .exists (fp ):
            z =np .load (fp )
            return np .asarray (z ["V"],np .float64 ),np .asarray (z ["F"],np .int64 )
    V =np .asarray (V ,float );F =np .asarray (F ,np .int32 )
    ms =pymeshlab .MeshSet ()
    ms .add_mesh (pymeshlab .Mesh (vertex_matrix =V ,face_matrix =F ))
    ms .meshing_remove_duplicate_vertices ()
    ms .meshing_remove_unreferenced_vertices ()
    V0 =ms .current_mesh ().vertex_matrix ();F0 =ms .current_mesh ().face_matrix ()
    pct =1.3 
    best =None 
    for _ in range (4 ):
        m2 =pymeshlab .MeshSet ()
        m2 .add_mesh (pymeshlab .Mesh (vertex_matrix =V0 ,face_matrix =F0 ))
        m2 .meshing_isotropic_explicit_remeshing (
        targetlen =pymeshlab .PercentageValue (pct ),iterations =iterations )
        n =m2 .current_mesh ().vertex_number ()
        best =m2 
        if n ==0 :
            break 
        if 0.7 *target <=n <=1.4 *target :
            break 
        pct *=(n /target )**0.5 
    cm =best .current_mesh ()
    Vo =np .asarray (cm .vertex_matrix (),np .float32 )# onbellekle AYNI yuvarlama (see docstring)
    Fo =np .asarray (cm .face_matrix (),np .int32 )
    if use_cache :
        os .makedirs (REMESH_CACHE ,exist_ok =True )
        tmp =fp +f".tmp{os .getpid ()}"# atomik yazim: half file okunmasin
        np .savez_compressed (tmp ,V =Vo ,F =Fo )
        os .replace (tmp +".npz"if os .path .exists (tmp +".npz")else tmp ,fp )
    return np .asarray (Vo ,np .float64 ),np .asarray (Fo ,np .int64 )


if __name__ =="__main__":
    import glob 
    from scipy .spatial import cKDTree 
    import json_dataset as jd ,step_to_json as sj ,cad_eval as ce 

    def stats (V ,cp ,tag ):
        V =np .asarray (V ,float )
        d ,_ =cKDTree (V ).query (V ,k =2 )
        cpd =cKDTree (V ).query (np .asarray (cp ,float ))[0 ]
        print (f"  {tag :14} verts={len (V ):5d}  NN-spacing={np .median (d [:,1 ]):.2f}mm  "
        f"CP->vtx med={np .median (cpd ):.1f}mm max={cpd .max ():.1f}")

    p =next (p for p in jd .iter_parts (r"C:\Users\DE00024082\Desktop\JSON")
    if str (p .part_nr )=="PXC.3031238")
    _ ,gt ,gd =jd .dedup_connection_points (p )
    Vj =np .asarray (p .vertices ,float );Fj =np .asarray (p .faces ,int )
    Vs ,Fs =sj .load_any_mesh (glob .glob ("_cad_eval_pxc/*3031238*.stp")[0 ],deflection =0.3 )
    Vs =np .asarray (Vs ,float )
    R ,t ,_ =ce .align_frames (Vs ,Vj );Vs_j =Vs @R .T +t 

    print ("=== RAW (before remesh) ===")
    stats (Vj ,gt ,"JSON raw");stats (Vs_j ,gt ,"STEP raw")
    print ("=== REMESHED (~6000 uniform, same pipeline) ===")
    Vjr ,Fjr =remesh_uniform (Vj ,Fj )
    Vsr ,Fsr =remesh_uniform (Vs ,Fs );Vsr_j =Vsr @R .T +t 
    stats (Vjr ,gt ,"JSON remesh");stats (Vsr_j ,gt ,"STEP remesh")
    print ("\n>>> remesh sonrasi NN-spacing yakinsa -> tessellation NORMALIZE oldu")
