# -*- coding: utf-8 -*-
"""Tests for the CP derivation + seal gate (pure numpy / logic, no GPU)."""
import os ,sys ,json 
import numpy as np 
import pytest 

sys .path .insert (0 ,os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))))
import connector3d 
import cp_openings 

CE =int (connector3d .CABLE_ENTRY );CT =int (connector3d .CONTACT )


def grid_patch (center ,label ,k =4 ,step =0.5 ,depth =0.0 ):
    """A k x k planar grid, optionally with its inner 2x2 core pushed back by `depth` (mm) --
    a flat patch (depth=0) has thesis insertion_depth_mm == 0.0 exactly (v_o == v_s, verified);
    a notched patch (depth>0) has insertion_depth_mm == depth exactly. Used to test the cp-v3
    Contact insertion-depth gate (flat screw-heads/pads vs real recessed connection openings)."""
    cx ,cy ,cz =center 
    verts =[]
    for iy in range (k ):
        for ix in range (k ):
            z =cz -depth if (depth and 0 <ix <k -1 and 0 <iy <k -1 )else cz 
            verts .append ([cx +ix *step ,cy +iy *step ,z ])
    verts =np .array (verts ,float )
    faces =[]
    for iy in range (k -1 ):
        for ix in range (k -1 ):
            a =iy *k +ix ;b =a +1 ;c =a +k ;d =c +1 
            faces +=[[a ,b ,c ],[b ,d ,c ]]
    return verts ,np .array (faces ,int ),np .full (len (verts ),label ,int )


def mesh_from (patches ):
    """patches: list of (center, label) or (center, label, depth). Returns V, F, labels with a
    housing backing plane so build_fragments (which skips HOUSING) still yields one component
    per patch."""
    Vs ,Fs ,Ls =[],[],[]
    off =0 
    for p in patches :
        center ,label =p [0 ],p [1 ];depth =p [2 ]if len (p )>2 else 0.0 
        v ,f ,l =grid_patch (center ,label ,depth =depth )
        Vs .append (v );Fs .append (f +off );Ls .append (l );off +=len (v )
    return np .vstack (Vs ),np .vstack (Fs ),np .concatenate (Ls )


def test_cableentry_and_deep_contact_are_default ():
# cp-v3 (2026-07-20, arbiter-selected): CableEntry always counts; a Contact patch with REAL
# insertion depth (a recessed clamp/contact opening, not a flat pad) counts too by default.
    V ,F ,L =mesh_from ([((0 ,0 ,0 ),CE ),((20 ,0 ,0 ),CE ),((40 ,0 ,0 ),CT ,2.0 )])
    cps =cp_openings .connection_points (V ,F ,L )
    assert len (cps )==3 
    assert sum (c ["source_label"]==CE for c in cps )==2 
    assert sum (c ["source_label"]==CT for c in cps )==1 


def test_flat_contact_excluded_by_depth_gate ():
# cp-v2 fallacy correction: Contact is NOT excluded by class, it's excluded when FLAT (no real
# insertion depth -> screw-head/pad, not a connection opening). depth=0 (default grid_patch)
# -> insertion_depth_mm == 0.0 exactly (verified) -> gated out by the default ct_depth_min_mm.
    V ,F ,L =mesh_from ([((0 ,0 ,0 ),CT ),((20 ,0 ,0 ),CT )])
    assert cp_openings .connection_points (V ,F ,L )==[]
    # disabling the gate proves it WAS the gate, not a class exclusion (cp-v2 behaviour)
    assert len (cp_openings .connection_points (V ,F ,L ,ct_depth_min_mm =0.0 ))==2 


def test_same_class_never_merged ():
# two Contact patches 6 mm apart, real depth (else the depth gate would zero both and the
# merge logic under test would never run), dedupe 10 mm -> still 2 (same class never merges)
    V ,F ,L =mesh_from ([((0 ,0 ,0 ),CT ,2.0 ),((6 ,0 ,0 ),CT ,2.0 )])
    cps =cp_openings .connection_points (V ,F ,L ,classes =(CE ,CT ),dedupe_mm =10.0 )
    assert len (cps )==2 


def test_cross_class_dedupe_merges_and_toggle ():
# a CableEntry + a Contact (real depth) 5 mm apart: dedupe 10 -> 1 (same terminal); dedupe 0 -> 2
    V ,F ,L =mesh_from ([((0 ,0 ,0 ),CE ),((5 ,0 ,0 ),CT ,2.0 )])
    assert len (cp_openings .connection_points (V ,F ,L ,classes =(CE ,CT ),dedupe_mm =10.0 ))==1 
    assert len (cp_openings .connection_points (V ,F ,L ,classes =(CE ,CT ),dedupe_mm =0.0 ))==2 


def test_cluster_mm_merges_one_cp_per_terminal ():
# cp-v3.1: a real terminal has several connection features each segmented separately. Two
# patches 6mm apart (same terminal) must collapse to ONE CP at their centroid when clustered,
# while a third 40mm away (a different terminal) must stay separate.
    V ,F ,L =mesh_from ([((0 ,0 ,0 ),CE ),((6 ,0 ,0 ),CE ),((40 ,0 ,0 ),CE )])
    assert len (cp_openings .connection_points (V ,F ,L ,cluster_mm =0.0 ))==3 # off = unchanged
    cps =cp_openings .connection_points (V ,F ,L ,cluster_mm =10.0 )
    assert len (cps )==2 ,"the two 6mm-apart features are one terminal; the 40mm one is separate"
    merged =[c for c in cps if c .get ("n_merged",1 )==2 ]
    assert len (merged )==1 and merged [0 ]["n_verts"]==32 ,"merged CP sums its members' vertices"


def test_cluster_keeps_cableentry_label_and_is_gt_safe ():
# a merged cluster containing a CableEntry reports CableEntry (it outranks Contact), and
# clustering is OFF by default so GT derivation is untouched.
    V ,F ,L =mesh_from ([((0 ,0 ,0 ),CE ),((5 ,0 ,0 ),CT ,2.0 )])
    assert len (cp_openings .connection_points (V ,F ,L ,dedupe_mm =0.0 ))==2 # default: no cluster
    cps =cp_openings .connection_points (V ,F ,L ,dedupe_mm =0.0 ,cluster_mm =10.0 )
    assert len (cps )==1 and cps [0 ]["source_label"]==CE 


def test_cp_v2_still_reachable_for_provenance ():
# the superseded CableEntry-only definition (cp-v2; arbiter F1 0.000 ten manufacturer PXC CPs)
# stays reachable by explicit classes=(CABLE_ENTRY,) for provenance/comparison
    V ,F ,L =mesh_from ([((0 ,0 ,0 ),CE ),((20 ,0 ,0 ),CE ),((40 ,0 ,0 ),CT ,2.0 )])
    cps =cp_openings .connection_points (V ,F ,L ,classes =(CE ,))
    assert len (cps )==2 and all (c ["source_label"]==CE for c in cps )


def test_gt_is_stable_to_prediction_knobs ():
# GT (no probs) count must not change when min_conf/vertex_conf are passed (they need probs)
    V ,F ,L =mesh_from ([((0 ,0 ,0 ),CE ),((20 ,0 ,0 ),CE )])
    base =len (cp_openings .connection_points (V ,F ,L ,min_v =1 ))
    assert base ==len (cp_openings .connection_points (V ,F ,L ,min_v =1 ,min_conf =0.9 ,vertex_conf =0.9 ))
    assert base ==2 


def test_seal_gate_rejects_pending_accepts_signed ():
    import cp_seal_score 
    pending ={"review_status":"PENDING","seal":{"reviewer":None ,"reviewed_utc":None ,"signature":None },
    "cps":[{"id":1 ,"decision":None }],"source_sha":{}}
    assert cp_seal_score .check (pending ,"X","val"),"pending must produce blockers"
    sealed ={"review_status":"SEALED",
    "seal":{"reviewer":"me","reviewed_utc":"2026-07-18T00:00:00Z","signature":"sig"},
    "cps":[{"id":1 ,"decision":"accept"},{"id":2 ,"decision":"reject"}],
    "source_sha":{}}
    assert cp_seal_score .check (sealed ,"X","val")==[],"fully sealed must have no blockers"


def test_review_package_writes_pending (tmp_path ):
# the emitted confirm JSON must NOT auto-confirm
    rec ={"review_status":"PENDING","cps":[{"id":1 ,"decision":None }],
    "seal":{"signature":None }}
    assert rec ["review_status"]=="PENDING"
    assert all (c ["decision"]is None for c in rec ["cps"])
    assert rec ["seal"]["signature"]is None 


    # --- axis-snap direction regression (cp_openings._snap_axis) ---
def test_axis_snap_unit_and_axis_aligned ():
    V ,F ,L =mesh_from ([((0 ,0 ,0 ),CE ),((20 ,0 ,0 ),CE )])
    for c in cp_openings .connection_points (V ,F ,L ):# axis_snap default True
        d =np .asarray (c ["direction"],float )
        assert abs (np .linalg .norm (d )-1.0 )<1e-6 ,"direction must be a unit vector"
        assert int ((np .abs (d )>0.999 ).sum ())==1 ,"exactly one axis component is +/-1"
        assert int ((np .abs (d )<1e-6 ).sum ())==2 ,"the other two components are 0 (only +/-X/Y/Z)"


def test_snap_axis_orients_outward ():
# geometry-independent: _snap_axis flips the snapped axis to agree with the outward reference
    assert np .allclose (cp_openings ._snap_axis ([0.9 ,0.1 ,0.0 ],[5.0 ,0.0 ,0.0 ]),[1.0 ,0.0 ,0.0 ])
    assert np .allclose (cp_openings ._snap_axis ([0.9 ,0.1 ,0.0 ],[-5.0 ,0.0 ,0.0 ]),[-1.0 ,0.0 ,0.0 ])


def test_snap_axis_keeps_a_genuine_tilt ():
    """Gercek egim YUVARLANMAZ -- only small titreme temizlenir.

    Bu test eskiden tersini dogruluyordu: 11.7 derece egik a direction eksene yuvarlanmali diyordu.
    Olcum bunu curuttu (8534 manufacturer ConnectionPoint): yonlerin only %75.6'si eksene hizali,
    **%19.1'i 10 dereceden extra egik** (most extra 43.1), and this single ureticiye ozgu not
    (PXC %16.9, WEI %21.7) -- klemenste tel acili a huniden girer, egim GERCEKTIR.
    Her yonu yuvarlamak eslesen CP'lerin %18.2'sinde ekseni 15-90 derece saptiriyordu and error
    part basina ya hep ya never goruluyordu (93 parcanin 19'u tamamen wrong).
    Esik 10 derece: korpusta 1-10 derece bandi tum CP'lerin only %5.3'u (titreme bandi),
    real egimler 22-43 derecede kumeleniyor.
    """
    # Esik ACIKCA verilir: urun varsayilani 90 (always yuvarla), because 10 derece MEASURED and
    # kaybetti (-0.0248 robot-hazir F1). Test MEKANIZMAYI korur, varsayilani not.
    tilt =np .array ([0.0 ,0.2 ,0.97 ])/np .linalg .norm ([0.0 ,0.2 ,0.97 ])# eksene 11.7 derece
    out =cp_openings ._snap_axis (tilt ,[0.0 ,0.0 ,3.0 ],snap_max_deg =10.0 )
    assert np .allclose (out ,tilt ,atol =1e-6 ),"gercek egim korunmali, eksene yuvarlanmamali"
    # sign yine de disari cevrilir
    flipped =cp_openings ._snap_axis (tilt ,[0.0 ,0.0 ,-3.0 ],snap_max_deg =10.0 )
    assert np .allclose (flipped ,-tilt ,atol =1e-6 )
    # small titreme (3 derece) still temizlenir
    wob =np .array ([0.05 ,0.0 ,1.0 ])/np .linalg .norm ([0.05 ,0.0 ,1.0 ])
    assert np .allclose (cp_openings ._snap_axis (wob ,[0.0 ,0.0 ,1.0 ],snap_max_deg =10.0 ),
    [0.0 ,0.0 ,1.0 ])
    # urun varsayilaninda (90) egim de yuvarlanir -- dagitilan davranis budur
    assert np .allclose (cp_openings ._snap_axis (tilt ,[0.0 ,0.0 ,3.0 ]),[0.0 ,0.0 ,1.0 ])
    # output each durumda unit vektor
    for v in (out ,flipped ):
        assert abs (np .linalg .norm (v )-1.0 )<1e-9 


def test_axis_snap_false_not_forced_to_axis ():
# with axis_snap=False the direction is NOT forced onto an axis (old behaviour preserved)
    V ,F ,L =mesh_from ([((0 ,0 ,0 ),CE ),((20 ,0 ,0 ),CE )])
    raw =[np .asarray (c ["direction"],float )for c in cp_openings .connection_points (V ,F ,L ,axis_snap =False )]
    snapped =[np .asarray (c ["direction"],float )for c in cp_openings .connection_points (V ,F ,L ,axis_snap =True )]
    assert all (abs (np .linalg .norm (d )-1.0 )<1e-6 for d in raw ),"raw directions still unit"
    assert all (int ((np .abs (d )>0.999 ).sum ())==1 for d in snapped ),"snapped are axis-aligned"
    # _snap_axis in isolation: a near-axis noisy vector snaps to that axis, oriented by outward ref
    e =cp_openings ._snap_axis (np .array ([0.1 ,0.95 ,-0.05 ]),np .array ([0.0 ,-1.0 ,0.0 ]))
    assert np .allclose (e ,[0.0 ,-1.0 ,0.0 ])
